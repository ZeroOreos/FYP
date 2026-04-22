from __future__ import annotations

from pathlib import Path
import math

from PIL import Image
import random
import io
import tarfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm

from Training.attacks import AttackResult, generate_attack_batch
from Training.config import AttackPolicy
from Training.losses import classification_accuracy, embedding_consistency_loss
from Training.recognizers import SurrogateWrapper, TrainableRecognizer


_TO_TENSOR = transforms.ToTensor()
_TAR_CACHE: dict[Path, tarfile.TarFile] = {}
_DEFAULT_FAR_TARGETS = (1e-2, 1e-3, 1e-4, 1e-5)


def _distributed_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def _reduce_scalar_sums(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    if _distributed_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [float(item) for item in tensor.tolist()]


def _reduce_member_totals(
    totals: dict[str, dict[str, float]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    if not _distributed_enabled():
        return totals
    names = sorted(totals)
    for name in names:
        bucket = totals[name]
        keys = ["loss", "accuracy", "correct", "samples", "batches"]
        reduced = _reduce_scalar_sums([float(bucket[key]) for key in keys], device)
        for key, value in zip(keys, reduced):
            bucket[key] = value
    return totals


def _empty_member_totals() -> dict[str, dict[str, float]]:
    return {}


def _accumulate_member_metrics(
    totals: dict[str, dict[str, float]],
    logits_by_member: dict[str, torch.Tensor],
    labels: torch.Tensor | dict[str, torch.Tensor],
) -> None:
    for name, logits in logits_by_member.items():
        bucket = totals.setdefault(
            name,
            {
                "loss": 0.0,
                "accuracy": 0.0,
                "correct": 0.0,
                "samples": 0.0,
                "batches": 0.0,
            },
        )
        member_labels = labels[name] if isinstance(labels, dict) else labels
        loss = F.cross_entropy(logits, member_labels)
        predictions = logits.argmax(dim=1)
        bucket["loss"] += float(loss.item())
        bucket["accuracy"] += classification_accuracy(logits, member_labels)
        bucket["correct"] += float((predictions == member_labels).sum().item())
        bucket["samples"] += float(member_labels.numel())
        bucket["batches"] += 1.0


def _finalize_member_metrics(totals: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for name, bucket in totals.items():
        batches = max(1.0, bucket["batches"])
        finalized[name] = {
            "loss": bucket["loss"] / batches,
            "accuracy": bucket["accuracy"] / batches,
            "correct": bucket["correct"],
            "samples": bucket["samples"],
        }
    return finalized


def _member_logits_from_training_outputs(
    outputs: dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    member_outputs = outputs.get("member_outputs")
    if not isinstance(member_outputs, dict):
        return {}
    member_logits: dict[str, torch.Tensor] = {}
    for name, item in member_outputs.items():
        predict_logits = item.get("predict_logits")
        if isinstance(predict_logits, torch.Tensor):
            member_logits[name] = predict_logits
    return member_logits


def _member_loss_labels_from_training_outputs(
    outputs: dict[str, torch.Tensor | dict[str, dict[str, torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    member_outputs = outputs.get("member_outputs")
    if not isinstance(member_outputs, dict):
        return {}
    member_labels: dict[str, torch.Tensor] = {}
    for name, item in member_outputs.items():
        loss_labels = item.get("loss_labels")
        if isinstance(loss_labels, torch.Tensor):
            member_labels[name] = loss_labels
    return member_labels


def _calibration_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 15,
) -> dict[str, float | list[dict[str, float]]]:
    if probs.numel() == 0:
        return {
            "ece": 0.0,
            "brier": 0.0,
            "nll": 0.0,
            "mean_confidence": 0.0,
            "reliability_bins": [],
        }

    confidence, predictions = probs.max(dim=1)
    correctness = (predictions == labels).float()
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).to(dtype=probs.dtype)
    brier = float(((probs - one_hot) ** 2).sum(dim=1).mean().item())
    nll = float(F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels).item())

    edges = torch.linspace(0.0, 1.0, steps=bins + 1, device=probs.device)
    ece = torch.zeros((), dtype=probs.dtype, device=probs.device)
    reliability_bins: list[dict[str, float]] = []
    total = max(1, labels.numel())
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index + 1 == bins:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum().item())
        if count == 0:
            reliability_bins.append(
                {
                    "bin_start": float(lower.item()),
                    "bin_end": float(upper.item()),
                    "count": 0.0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                }
            )
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidence[mask].mean()
        ece = ece + (mask.float().mean() * torch.abs(bin_acc - bin_conf))
        reliability_bins.append(
            {
                "bin_start": float(lower.item()),
                "bin_end": float(upper.item()),
                "count": float(count),
                "accuracy": float(bin_acc.item()),
                "confidence": float(bin_conf.item()),
            }
        )

    return {
        "ece": float(ece.item()),
        "brier": brier,
        "nll": nll,
        "mean_confidence": float(confidence.mean().item()),
        "reliability_bins": reliability_bins,
    }


def _collect_pairwise_error_stats(
    logits_by_member: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict[str, object]:
    member_names = sorted(logits_by_member)
    if len(member_names) < 2:
        return {
            "pairwise_error_correlation_mean": 0.0,
            "pairwise_error_correlation_by_pair": {},
        }

    error_vectors: dict[str, torch.Tensor] = {}
    for name, logits in logits_by_member.items():
        error_vectors[name] = (logits.argmax(dim=1) != labels).float()

    correlations: dict[str, float] = {}
    values: list[float] = []
    for left_index, left_name in enumerate(member_names):
        left_errors = error_vectors[left_name]
        left_centered = left_errors - left_errors.mean()
        left_std = left_centered.pow(2).mean().sqrt()
        for right_name in member_names[left_index + 1:]:
            right_errors = error_vectors[right_name]
            right_centered = right_errors - right_errors.mean()
            right_std = right_centered.pow(2).mean().sqrt()
            if float(left_std.item()) == 0.0 or float(right_std.item()) == 0.0:
                corr = 0.0
            else:
                corr = float(((left_centered * right_centered).mean() / (left_std * right_std)).item())
            key = f"{left_name}__{right_name}"
            correlations[key] = corr
            values.append(corr)

    return {
        "pairwise_error_correlation_mean": float(sum(values) / max(1, len(values))),
        "pairwise_error_correlation_by_pair": correlations,
    }


def _ensemble_analysis(
    *,
    logits_by_member: dict[str, torch.Tensor],
    fused_logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    if not logits_by_member:
        return {}

    member_names = sorted(logits_by_member)
    member_accuracies = {
        name: classification_accuracy(logits, labels)
        for name, logits in logits_by_member.items()
    }
    fused_accuracy = classification_accuracy(fused_logits, labels)
    mean_member_logits = torch.stack([logits_by_member[name] for name in member_names], dim=0).mean(dim=0)
    mean_single_accuracy = classification_accuracy(mean_member_logits, labels)
    best_single_name = max(member_accuracies, key=member_accuracies.get)
    best_single_accuracy = member_accuracies[best_single_name]

    leave_one_out: dict[str, dict[str, float]] = {}
    if len(member_names) > 1:
        for omitted in member_names:
            kept = [logits_by_member[name] for name in member_names if name != omitted]
            loo_logits = torch.stack(kept, dim=0).mean(dim=0)
            loo_accuracy = classification_accuracy(loo_logits, labels)
            leave_one_out[omitted] = {
                "accuracy": loo_accuracy,
                "gain_vs_leave_one_out": fused_accuracy - loo_accuracy,
            }

    analysis: dict[str, object] = {
        "fused_accuracy": fused_accuracy,
        "best_single_name": best_single_name,
        "best_single_accuracy": best_single_accuracy,
        "mean_single_accuracy": float(sum(member_accuracies.values()) / max(1, len(member_accuracies))),
        "mean_member_logits_accuracy": mean_single_accuracy,
        "gain_vs_best_single": fused_accuracy - best_single_accuracy,
        "gain_vs_mean_single": fused_accuracy - (float(sum(member_accuracies.values()) / max(1, len(member_accuracies)))),
        "member_accuracies": member_accuracies,
        "leave_one_out": leave_one_out,
    }
    analysis.update(_collect_pairwise_error_stats(logits_by_member, labels))
    return analysis


def _load_image_from_reference(reference: Path | str) -> Image.Image:
    ref = str(reference)
    if "::" not in ref:
        return Image.open(Path(ref)).convert("RGB")

    shard_str, member_name = ref.split("::", 1)
    shard_path = Path(shard_str).resolve()
    tar_handle = _TAR_CACHE.get(shard_path)
    if tar_handle is None:
        tar_handle = tarfile.open(shard_path, mode="r:gz")
        _TAR_CACHE[shard_path] = tar_handle
    file_obj = tar_handle.extractfile(member_name)
    if file_obj is None:
        raise RuntimeError(f"Failed to read {member_name} from shard {shard_path}")
    return Image.open(io.BytesIO(file_obj.read())).convert("RGB")


@torch.no_grad()
def evaluate_clean(
    model: TrainableRecognizer,
    loader,
    device: torch.device,
    progress_desc: str | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    total_samples = 0
    total_correct = 0
    member_totals = _empty_member_totals()
    all_probs = []
    all_labels = []
    fused_member_logits: list[dict[str, torch.Tensor]] = []
    batches = 0
    distributed_eval = _distributed_enabled()

    iterator = loader
    if progress_desc is not None:
        iterator = tqdm(loader, desc=progress_desc, unit="batch", leave=False, dynamic_ncols=True, smoothing=0.1)

    for images, labels, _, _ in iterator:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if hasattr(model, "predict_eval_outputs"):
            eval_outputs = model.predict_eval_outputs(images)
            logits = eval_outputs["predict_logits"]
            member_logits = eval_outputs.get("member_logits")
        else:
            logits = model.predict_logits(images)
            member_logits = model.predict_member_logits(images) if hasattr(model, "predict_member_logits") else None
        if not distributed_eval:
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
        if isinstance(member_logits, dict) and member_logits:
            _accumulate_member_metrics(member_totals, member_logits, labels)
            if not distributed_eval:
                fused_member_logits.append({name: tensor.detach().cpu() for name, tensor in member_logits.items()})
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        total_loss += float(loss.item())
        total_acc += classification_accuracy(logits, labels)
        total_correct += int((predictions == labels).sum().item())
        total_samples += int(labels.numel())
        total_batches += 1
        batches += 1

    if batches == 0:
        return {"loss": 0.0, "accuracy": 0.0, "samples": 0.0, "correct": 0.0}
    total_loss, total_batches, total_samples, total_correct = _reduce_scalar_sums(
        [total_loss, float(total_batches), float(total_samples), float(total_correct)],
        device,
    )
    member_totals = _reduce_member_totals(member_totals, device)
    accuracy = float(total_correct) / float(max(1.0, total_samples))
    ensemble_metrics: dict[str, object] = {}
    calibration: dict[str, object] = {}
    if not distributed_eval and fused_member_logits:
        merged_logits: dict[str, list[torch.Tensor]] = {}
        for batch_payload in fused_member_logits:
            for name, tensor in batch_payload.items():
                merged_logits.setdefault(name, []).append(tensor)
        logits_by_member = {name: torch.cat(chunks, dim=0) for name, chunks in merged_logits.items()}
        probs = torch.cat(all_probs, dim=0) if all_probs else torch.empty(0)
        stacked_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
        fused_logits = torch.log(probs.clamp_min(1e-12))
        ensemble_metrics = _ensemble_analysis(
            logits_by_member=logits_by_member,
            fused_logits=fused_logits,
            labels=stacked_labels,
        )
        calibration = _calibration_metrics(probs, stacked_labels)
    elif not distributed_eval:
        probs = torch.cat(all_probs, dim=0) if all_probs else torch.empty(0)
        stacked_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
        calibration = _calibration_metrics(probs, stacked_labels) if all_probs else _calibration_metrics(torch.empty((0, 1)), torch.empty(0, dtype=torch.long))
    return {
        "loss": total_loss / max(1.0, total_batches),
        "accuracy": accuracy,
        "samples": float(total_samples),
        "correct": float(total_correct),
        "member_metrics": _finalize_member_metrics(member_totals),
        "calibration": calibration,
        "ensemble": ensemble_metrics,
    }


def evaluate_robust(
    *,
    model: TrainableRecognizer,
    loader,
    device: torch.device,
    policy: AttackPolicy | None,
    surrogates: dict[str, SurrogateWrapper],
    image_size: int,
    attack_chunk_size: int | None = None,
    progress_desc: str | None = None,
) -> dict[str, float]:
    if policy is None:
        return {"loss": 0.0, "accuracy": 0.0, "consistency": 0.0, "cached_hits": 0.0}

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_consistency = 0.0
    total_cached_hits = 0.0
    total_samples = 0
    total_correct = 0
    total_clean_correct = 0
    total_success_from_clean_correct = 0
    member_totals = _empty_member_totals()
    all_probs = []
    all_labels = []
    fused_member_logits: list[dict[str, torch.Tensor]] = []
    batches = 0
    distributed_eval = _distributed_enabled()

    iterator = loader
    if progress_desc is not None:
        iterator = tqdm(loader, desc=progress_desc, unit="batch", leave=False, dynamic_ncols=True, smoothing=0.1)

    for images, labels, _, rel_paths in iterator:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        attack_result: AttackResult = generate_attack_batch(
            policy=policy,
            images=images,
            labels=labels,
            rel_paths=list(rel_paths),
            image_size=image_size,
            target_model=model,
            surrogates=surrogates,
            device=device,
            attack_chunk_size=attack_chunk_size,
        )

        with torch.no_grad():
            if hasattr(model, "predict_eval_outputs"):
                clean_outputs = model.predict_eval_outputs(images)
                clean_embeddings = clean_outputs["embeddings"]
                clean_predict_logits = clean_outputs["predict_logits"]
            else:
                clean_embeddings = model.forward_embeddings(images)
                clean_predict_logits = model.predict_logits_from_embeddings(clean_embeddings)
            adv_outputs = model(attack_result.images, labels)
            adv_logits = adv_outputs["logits"]
            adv_loss_labels = adv_outputs.get("loss_labels", labels)
            adv_embeddings = adv_outputs["embeddings"]
            if hasattr(model, "predict_eval_outputs"):
                adv_eval_outputs = model.predict_eval_outputs(attack_result.images)
                adv_predict_logits = adv_eval_outputs["predict_logits"]
                member_logits = adv_eval_outputs.get("member_logits") or {}
            else:
                adv_predict_logits = adv_outputs["predict_logits"]
                member_logits = _member_logits_from_training_outputs(adv_outputs)
            member_loss_labels = _member_loss_labels_from_training_outputs(adv_outputs)
            if member_logits:
                _accumulate_member_metrics(
                    member_totals,
                    member_logits,
                    labels if hasattr(model, "predict_eval_outputs") else (member_loss_labels or labels),
                )
                if not distributed_eval:
                    fused_member_logits.append({name: tensor.detach().cpu() for name, tensor in member_logits.items()})

        loss = F.cross_entropy(adv_logits, adv_loss_labels)
        if not distributed_eval:
            all_probs.append(torch.softmax(adv_predict_logits, dim=1).detach().cpu())
            all_labels.append(labels.detach().cpu())
        clean_predictions = clean_predict_logits.argmax(dim=1)
        adv_predictions = adv_predict_logits.argmax(dim=1)
        clean_correct_mask = clean_predictions == labels
        adv_correct_mask = adv_predictions == labels
        total_loss += float(loss.item())
        total_acc += classification_accuracy(adv_predict_logits, labels)
        total_consistency += float(embedding_consistency_loss(clean_embeddings, adv_embeddings).item())
        total_cached_hits += float(attack_result.cached_hits)
        total_correct += int(adv_correct_mask.sum().item())
        total_clean_correct += int(clean_correct_mask.sum().item())
        total_success_from_clean_correct += int((clean_correct_mask & ~adv_correct_mask).sum().item())
        total_samples += int(labels.numel())
        batches += 1

    if batches == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0,
            "cached_hits": 0.0,
            "samples": 0.0,
            "correct": 0.0,
            "clean_correct": 0.0,
            "attack_success_rate": 0.0,
        }
    (
        total_loss,
        total_consistency,
        total_cached_hits,
        total_samples,
        total_correct,
        total_clean_correct,
        total_success_from_clean_correct,
        batches,
    ) = _reduce_scalar_sums(
        [
            total_loss,
            total_consistency,
            total_cached_hits,
            float(total_samples),
            float(total_correct),
            float(total_clean_correct),
            float(total_success_from_clean_correct),
            float(batches),
        ],
        device,
    )
    member_totals = _reduce_member_totals(member_totals, device)
    asr = float(total_success_from_clean_correct) / float(max(1.0, total_clean_correct))
    accuracy = float(total_correct) / float(max(1.0, total_samples))
    ensemble_metrics: dict[str, object] = {}
    calibration: dict[str, object] = {}
    if not distributed_eval and fused_member_logits:
        merged_logits: dict[str, list[torch.Tensor]] = {}
        for batch_payload in fused_member_logits:
            for name, tensor in batch_payload.items():
                merged_logits.setdefault(name, []).append(tensor)
        logits_by_member = {name: torch.cat(chunks, dim=0) for name, chunks in merged_logits.items()}
        probs = torch.cat(all_probs, dim=0) if all_probs else torch.empty(0)
        stacked_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
        fused_logits = torch.log(probs.clamp_min(1e-12))
        ensemble_metrics = _ensemble_analysis(
            logits_by_member=logits_by_member,
            fused_logits=fused_logits,
            labels=stacked_labels,
        )
        calibration = _calibration_metrics(probs, stacked_labels)
    elif not distributed_eval:
        probs = torch.cat(all_probs, dim=0) if all_probs else torch.empty(0)
        stacked_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
        calibration = _calibration_metrics(probs, stacked_labels) if all_probs else _calibration_metrics(torch.empty((0, 1)), torch.empty(0, dtype=torch.long))
    return {
        "loss": total_loss / max(1.0, batches),
        "accuracy": accuracy,
        "consistency": total_consistency / max(1.0, batches),
        "cached_hits": total_cached_hits / max(1.0, batches),
        "samples": float(total_samples),
        "correct": float(total_correct),
        "clean_correct": float(total_clean_correct),
        "attack_success_rate": asr,
        "member_metrics": _finalize_member_metrics(member_totals),
        "calibration": calibration,
        "ensemble": ensemble_metrics,
    }


def evaluate_robust_all(
    *,
    model: TrainableRecognizer,
    loader,
    device: torch.device,
    policies: list[AttackPolicy],
    surrogates: dict[str, SurrogateWrapper],
    image_size: int,
    attack_chunk_size: int | None = None,
    progress_prefix: str | None = None,
    distributed: bool = False,
) -> dict[str, object]:
    enabled = [policy for policy in policies if policy.enabled]
    if not enabled:
        return {
            "average": {
                "loss": 0.0,
                "accuracy": 0.0,
                "consistency": 0.0,
                "cached_hits": 0.0,
                "samples": 0.0,
                "correct": 0.0,
                "clean_correct": 0.0,
                "attack_success_rate": 0.0,
            },
            "by_attack": {},
        }

    by_attack: dict[str, dict[str, float]] = {}
    for policy in enabled:
        by_attack[policy.name] = evaluate_robust(
            model=model,
            loader=loader,
            device=device,
            policy=policy,
            surrogates=surrogates,
            image_size=image_size,
            attack_chunk_size=attack_chunk_size,
            progress_desc=None if progress_prefix is None else f"{progress_prefix}:{policy.name}",
        )

    average = {
        "loss": sum(metrics["loss"] for metrics in by_attack.values()) / len(by_attack),
        "accuracy": sum(metrics["accuracy"] for metrics in by_attack.values()) / len(by_attack),
        "consistency": sum(metrics["consistency"] for metrics in by_attack.values()) / len(by_attack),
        "cached_hits": sum(metrics["cached_hits"] for metrics in by_attack.values()) / len(by_attack),
        "samples": sum(metrics["samples"] for metrics in by_attack.values()) / len(by_attack),
        "correct": sum(metrics["correct"] for metrics in by_attack.values()) / len(by_attack),
        "clean_correct": sum(metrics["clean_correct"] for metrics in by_attack.values()) / len(by_attack),
        "attack_success_rate": sum(metrics["attack_success_rate"] for metrics in by_attack.values()) / len(by_attack),
    }
    return {"average": average, "by_attack": by_attack}


def choose_eval_policy(policies: list[AttackPolicy], preferred_name: str | None) -> AttackPolicy | None:
    enabled = [policy for policy in policies if policy.enabled]
    if not enabled:
        return None
    if preferred_name is not None:
        for policy in enabled:
            if policy.name == preferred_name:
                return policy
    rng = random.Random(0)
    return enabled[rng.randrange(len(enabled))]


def _best_threshold_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    if scores.numel() == 0:
        return 0.0, 0.0
    sorted_scores = torch.unique(scores)
    best_acc = 0.0
    best_threshold = 0.0
    for threshold in sorted_scores:
        predictions = (scores >= threshold).to(torch.int64)
        acc = float((predictions == labels).float().mean().item())
        if acc >= best_acc:
            best_acc = acc
            best_threshold = float(threshold.item())
    return best_acc, best_threshold


def _eer(scores: torch.Tensor, labels: torch.Tensor) -> float:
    if scores.numel() == 0:
        return 0.0
    thresholds = torch.unique(scores)
    best_gap = None
    best_eer = 1.0
    positive = labels == 1
    negative = labels == 0
    pos_count = max(1, int(positive.sum().item()))
    neg_count = max(1, int(negative.sum().item()))
    for threshold in thresholds:
        predictions = scores >= threshold
        far = float((predictions[negative]).float().mean().item()) if negative.any() else 0.0
        frr = float((~predictions[positive]).float().mean().item()) if positive.any() else 0.0
        gap = abs(far - frr)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_eer = 0.5 * (far + frr)
    return best_eer


def _roc_curve(scores: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.numel() == 0:
        zeros = torch.zeros(1, dtype=torch.float32, device=scores.device)
        return zeros, zeros, zeros
    thresholds = torch.sort(torch.unique(scores), descending=True).values
    positive = labels == 1
    negative = labels == 0
    pos_count = max(1, int(positive.sum().item()))
    neg_count = max(1, int(negative.sum().item()))
    tprs = []
    fprs = []
    for threshold in thresholds:
        predictions = scores >= threshold
        true_positive = int((predictions & positive).sum().item())
        false_positive = int((predictions & negative).sum().item())
        tprs.append(true_positive / pos_count)
        fprs.append(false_positive / neg_count)
    fpr_tensor = torch.as_tensor([0.0, *fprs, 1.0], dtype=torch.float32, device=scores.device)
    tpr_tensor = torch.as_tensor([0.0, *tprs, 1.0], dtype=torch.float32, device=scores.device)
    threshold_tensor = torch.as_tensor(
        [float("inf"), *[float(item.item()) for item in thresholds], float("-inf")],
        dtype=torch.float32,
        device=scores.device,
    )
    return fpr_tensor, tpr_tensor, threshold_tensor


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    fpr, tpr, _ = _roc_curve(scores, labels)
    if fpr.numel() <= 1:
        return 0.0
    return float(torch.trapz(tpr, fpr).item())


def _tar_at_far(
    scores: torch.Tensor,
    labels: torch.Tensor,
    far_targets: tuple[float, ...] = _DEFAULT_FAR_TARGETS,
) -> dict[str, float | bool | None]:
    if scores.numel() == 0:
        return {
            f"tar@far={target:g}": 0.0 for target in far_targets
        } | {
            f"threshold@far={target:g}": None for target in far_targets
        } | {
            f"far_supported@{target:g}": False for target in far_targets
        }

    positive = labels == 1
    negative = labels == 0
    neg_count = int(negative.sum().item())
    thresholds = torch.sort(torch.unique(scores), descending=True).values
    metrics: dict[str, float | bool | None] = {}
    for target in far_targets:
        supported = neg_count >= max(1, math.ceil(1.0 / target))
        best_tar = 0.0
        best_threshold: float | None = None
        for threshold in thresholds:
            predictions = scores >= threshold
            far = float((predictions[negative]).float().mean().item()) if negative.any() else 0.0
            tar = float((predictions[positive]).float().mean().item()) if positive.any() else 0.0
            if far <= target and (best_threshold is None or tar >= best_tar):
                best_tar = tar
                best_threshold = float(threshold.item())
        metrics[f"tar@far={target:g}"] = best_tar
        metrics[f"threshold@far={target:g}"] = best_threshold
        metrics[f"far_supported@{target:g}"] = supported
    return metrics


def verification_metrics_from_scores(
    scores: torch.Tensor,
    labels: torch.Tensor,
    far_targets: tuple[float, ...] = _DEFAULT_FAR_TARGETS,
) -> dict[str, float | bool | None]:
    best_acc, best_threshold = _best_threshold_accuracy(scores, labels)
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    metrics: dict[str, float | bool | None] = {
        "pairs": float(scores.numel()),
        "positive_pairs": float(positive_scores.numel()),
        "negative_pairs": float(negative_scores.numel()),
        "best_accuracy": best_acc,
        "best_threshold": best_threshold,
        "eer": _eer(scores, labels),
        "roc_auc": _roc_auc(scores, labels),
        "mean_positive_cosine": float(positive_scores.mean().item()) if positive_scores.numel() else 0.0,
        "mean_negative_cosine": float(negative_scores.mean().item()) if negative_scores.numel() else 0.0,
    }
    metrics.update(_tar_at_far(scores, labels, far_targets))
    return metrics


@torch.no_grad()
def evaluate_verification_pairs(
    *,
    model: TrainableRecognizer,
    pairs_path: Path,
    device: torch.device,
) -> dict[str, float]:
    pair_data = dict(**__import__("numpy").load(pairs_path, allow_pickle=True))
    img1_paths = [Path(item) for item in pair_data["img1_paths"].tolist()]
    img2_paths = [Path(item) for item in pair_data["img2_paths"].tolist()]
    labels = torch.as_tensor(pair_data["labels"].astype("int64"))
    unique_paths = sorted({path for path in img1_paths + img2_paths})
    embedding_cache: dict[Path, torch.Tensor] = {}
    model.eval()

    for path in unique_paths:
        image = _load_image_from_reference(path)
        tensor = _TO_TENSOR(image).unsqueeze(0).to(device, non_blocking=device.type == "cuda")
        embedding_cache[path] = model.forward_embeddings(tensor).squeeze(0).detach().cpu()

    scores = []
    for path_a, path_b in zip(img1_paths, img2_paths):
        score = F.cosine_similarity(
            F.normalize(embedding_cache[path_a], dim=0).unsqueeze(0),
            F.normalize(embedding_cache[path_b], dim=0).unsqueeze(0),
            dim=1,
        )
        scores.append(score.squeeze(0))
    score_tensor = torch.stack(scores)
    return verification_metrics_from_scores(score_tensor, labels)
