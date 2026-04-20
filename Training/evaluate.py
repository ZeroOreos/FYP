from __future__ import annotations

from pathlib import Path
import math

from PIL import Image
import random
import io
import tarfile

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm

from Training.attacks import AttackResult, generate_attack_batch
from Training.config import AttackPolicy
from Training.losses import classification_accuracy, embedding_consistency_loss
from Training.recognizers import SurrogateWrapper, TrainableRecognizer


_TO_TENSOR = transforms.ToTensor()
_TAR_CACHE: dict[Path, tarfile.TarFile] = {}
_DEFAULT_FAR_TARGETS = (1e-2, 1e-3, 1e-4)


def _empty_member_totals() -> dict[str, dict[str, float]]:
    return {}


def _accumulate_member_metrics(
    totals: dict[str, dict[str, float]],
    logits_by_member: dict[str, torch.Tensor],
    labels: torch.Tensor,
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
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        bucket["loss"] += float(loss.item())
        bucket["accuracy"] += classification_accuracy(logits, labels)
        bucket["correct"] += float((predictions == labels).sum().item())
        bucket["samples"] += float(labels.numel())
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
    batches = 0

    iterator = loader
    if progress_desc is not None:
        iterator = tqdm(loader, desc=progress_desc, unit="batch", leave=False, dynamic_ncols=True, smoothing=0.1)

    for images, labels, _, _ in iterator:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        logits = model.predict_logits(images)
        if hasattr(model, "predict_member_logits"):
            _accumulate_member_metrics(member_totals, model.predict_member_logits(images), labels)
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
    return {
        "loss": total_loss / batches,
        "accuracy": total_acc / batches,
        "samples": float(total_samples),
        "correct": float(total_correct),
        "member_metrics": _finalize_member_metrics(member_totals),
    }


def evaluate_robust(
    *,
    model: TrainableRecognizer,
    loader,
    device: torch.device,
    policy: AttackPolicy | None,
    surrogates: dict[str, SurrogateWrapper],
    image_size: int,
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
    batches = 0

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
        )

        with torch.no_grad():
            clean_embeddings = model.forward_embeddings(images)
            clean_predict_logits = model.predict_logits_from_embeddings(clean_embeddings)
            adv_logits, adv_embeddings = model.forward_logits(attack_result.images, labels)
            adv_predict_logits = model.predict_logits_from_embeddings(adv_embeddings)
            if hasattr(model, "predict_member_logits_from_embeddings"):
                _accumulate_member_metrics(member_totals, model.predict_member_logits_from_embeddings(adv_embeddings), labels)

        loss = F.cross_entropy(adv_logits, labels)
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
    asr = float(total_success_from_clean_correct) / float(max(1, total_clean_correct))
    return {
        "loss": total_loss / batches,
        "accuracy": total_acc / batches,
        "consistency": total_consistency / batches,
        "cached_hits": total_cached_hits / batches,
        "samples": float(total_samples),
        "correct": float(total_correct),
        "clean_correct": float(total_clean_correct),
        "attack_success_rate": asr,
        "member_metrics": _finalize_member_metrics(member_totals),
    }


def evaluate_robust_all(
    *,
    model: TrainableRecognizer,
    loader,
    device: torch.device,
    policies: list[AttackPolicy],
    surrogates: dict[str, SurrogateWrapper],
    image_size: int,
    progress_prefix: str | None = None,
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
