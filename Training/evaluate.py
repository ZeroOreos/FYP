from __future__ import annotations

from pathlib import Path
import gc
import math

from PIL import Image
import random
import io
import tarfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from tqdm.auto import tqdm

from Training.attacks import AttackResult, generate_attack_batch
from Training.config import AttackPolicy
from Training.losses import classification_accuracy, embedding_consistency_loss
from Training.recognizers import ExternalRecognizerEnsemble, SurrogateWrapper, TrainableRecognizer


_TAR_CACHE: dict[Path, tarfile.TarFile] = {}
_DEFAULT_FAR_TARGETS = (1e-2, 1e-3, 1e-4, 1e-5)
_VERIFICATION_BATCH_SIZE_CUDA = 512
_VERIFICATION_BATCH_SIZE_CPU = 64
_VERIFICATION_NUM_WORKERS_CUDA = 4
_VERIFICATION_NUM_WORKERS_CPU = 2
_VERIFICATION_PREFETCH_FACTOR = 4
_VERIFICATION_SCORE_CHUNK = 65536


class _NullProgress:
    def update(self, _: int | float = 1) -> None:
        return

    def set_description(self, _: str) -> None:
        return

    def close(self) -> None:
        return


class _VerificationImageDataset(Dataset):
    def __init__(self, paths: list[Path], *, image_size: int) -> None:
        self.paths = paths
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = _load_image_from_reference(self.paths[index])
        target_size = (self.image_size, self.image_size)
        if image.size != target_size:
            image = image.resize(target_size, Image.BILINEAR)
        return transforms.functional.to_tensor(image), index


def _unwrap_model(model: TrainableRecognizer):
    return getattr(model, "module", model)


def _verification_image_size(model: TrainableRecognizer) -> int:
    unwrapped = _unwrap_model(model)
    backbone_name = str(getattr(unwrapped, "backbone_name", "")).lower()
    if backbone_name.startswith("iresnet"):
        return 112
    if backbone_name.startswith("resnet"):
        return 160
    return 112


def _verification_num_workers(device: torch.device) -> int:
    if device.type != "cuda":
        return _VERIFICATION_NUM_WORKERS_CPU
    if _distributed_enabled():
        world_size = max(1, dist.get_world_size())
        return max(1, min(_VERIFICATION_NUM_WORKERS_CUDA, 8 // world_size))
    return _VERIFICATION_NUM_WORKERS_CUDA


def _clear_verification_device_caches(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()


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


def _empty_ensemble_totals() -> dict[str, object]:
    return {}


def _accumulate_ensemble_totals(
    totals: dict[str, object],
    logits_by_member: dict[str, torch.Tensor],
    fused_logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if not logits_by_member:
        return
    member_names = sorted(logits_by_member)
    if any(logits_by_member[name].shape != fused_logits.shape for name in member_names):
        return

    samples = float(labels.numel())
    totals["samples"] = float(totals.get("samples", 0.0)) + samples
    totals["fused_correct"] = float(totals.get("fused_correct", 0.0)) + float((fused_logits.argmax(dim=1) == labels).sum().item())

    mean_member_logits = torch.stack([logits_by_member[name] for name in member_names], dim=0).mean(dim=0)
    totals["mean_member_logits_correct"] = float(totals.get("mean_member_logits_correct", 0.0)) + float(
        (mean_member_logits.argmax(dim=1) == labels).sum().item()
    )

    member_correct = totals.setdefault("member_correct", {})
    error_sums = totals.setdefault("error_sums", {})
    pair_error_products = totals.setdefault("pair_error_products", {})
    leave_one_out_correct = totals.setdefault("leave_one_out_correct", {})
    assert isinstance(member_correct, dict)
    assert isinstance(error_sums, dict)
    assert isinstance(pair_error_products, dict)
    assert isinstance(leave_one_out_correct, dict)

    member_errors: dict[str, torch.Tensor] = {}
    for name in member_names:
        predictions = logits_by_member[name].argmax(dim=1)
        correct = predictions == labels
        errors = (~correct).float()
        member_correct[name] = float(member_correct.get(name, 0.0)) + float(correct.sum().item())
        error_sums[name] = float(error_sums.get(name, 0.0)) + float(errors.sum().item())
        member_errors[name] = errors

    if len(member_names) > 1:
        for omitted in member_names:
            kept_logits = [logits_by_member[name] for name in member_names if name != omitted]
            loo_logits = torch.stack(kept_logits, dim=0).mean(dim=0)
            leave_one_out_correct[omitted] = float(leave_one_out_correct.get(omitted, 0.0)) + float(
                (loo_logits.argmax(dim=1) == labels).sum().item()
            )
        for left_index, left_name in enumerate(member_names):
            for right_name in member_names[left_index + 1:]:
                key = f"{left_name}__{right_name}"
                pair_error_products[key] = float(pair_error_products.get(key, 0.0)) + float(
                    (member_errors[left_name] * member_errors[right_name]).sum().item()
                )


def _reduce_nested_float_dict(values: dict[str, float], device: torch.device) -> dict[str, float]:
    if not values or not _distributed_enabled():
        return values
    keys = sorted(values)
    reduced = _reduce_scalar_sums([float(values[key]) for key in keys], device)
    return {key: value for key, value in zip(keys, reduced)}


def _reduce_ensemble_totals(totals: dict[str, object], device: torch.device) -> dict[str, object]:
    if not totals or not _distributed_enabled():
        return totals
    scalar_keys = ["samples", "fused_correct", "mean_member_logits_correct"]
    reduced_scalars = _reduce_scalar_sums([float(totals.get(key, 0.0)) for key in scalar_keys], device)
    for key, value in zip(scalar_keys, reduced_scalars):
        totals[key] = value
    for nested_key in ("member_correct", "error_sums", "pair_error_products", "leave_one_out_correct"):
        nested = totals.get(nested_key)
        if isinstance(nested, dict):
            totals[nested_key] = _reduce_nested_float_dict({str(k): float(v) for k, v in nested.items()}, device)
    return totals


def _finalize_ensemble_totals(totals: dict[str, object]) -> dict[str, object]:
    samples = float(totals.get("samples", 0.0)) if totals else 0.0
    member_correct = totals.get("member_correct", {}) if totals else {}
    if samples <= 0 or not isinstance(member_correct, dict) or not member_correct:
        return {}

    member_names = sorted(str(name) for name in member_correct)
    member_accuracies = {
        name: float(member_correct.get(name, 0.0)) / samples
        for name in member_names
    }
    fused_accuracy = float(totals.get("fused_correct", 0.0)) / samples
    mean_member_logits_accuracy = float(totals.get("mean_member_logits_correct", 0.0)) / samples
    best_single_name = max(member_accuracies, key=member_accuracies.get)
    best_single_accuracy = member_accuracies[best_single_name]
    mean_single_accuracy = float(sum(member_accuracies.values()) / max(1, len(member_accuracies)))

    leave_one_out: dict[str, dict[str, float]] = {}
    raw_leave_one_out = totals.get("leave_one_out_correct", {})
    if isinstance(raw_leave_one_out, dict):
        for name in sorted(str(item) for item in raw_leave_one_out):
            accuracy = float(raw_leave_one_out.get(name, 0.0)) / samples
            leave_one_out[name] = {
                "accuracy": accuracy,
                "gain_vs_leave_one_out": fused_accuracy - accuracy,
            }

    error_sums = totals.get("error_sums", {})
    pair_error_products = totals.get("pair_error_products", {})
    pairwise: dict[str, float] = {}
    pairwise_values: list[float] = []
    if isinstance(error_sums, dict) and isinstance(pair_error_products, dict):
        for key, product_sum in pair_error_products.items():
            left_name, right_name = str(key).split("__", 1)
            left_mean = float(error_sums.get(left_name, 0.0)) / samples
            right_mean = float(error_sums.get(right_name, 0.0)) / samples
            product_mean = float(product_sum) / samples
            covariance = product_mean - left_mean * right_mean
            left_var = max(0.0, left_mean * (1.0 - left_mean))
            right_var = max(0.0, right_mean * (1.0 - right_mean))
            denom = math.sqrt(left_var * right_var)
            corr = 0.0 if denom <= 0 else covariance / denom
            pairwise[str(key)] = float(corr)
            pairwise_values.append(float(corr))

    return {
        "fused_accuracy": fused_accuracy,
        "best_single_name": best_single_name,
        "best_single_accuracy": best_single_accuracy,
        "mean_single_accuracy": mean_single_accuracy,
        "mean_member_logits_accuracy": mean_member_logits_accuracy,
        "gain_vs_best_single": fused_accuracy - best_single_accuracy,
        "gain_vs_mean_single": fused_accuracy - mean_single_accuracy,
        "member_accuracies": member_accuracies,
        "leave_one_out": leave_one_out,
        "pairwise_error_correlation_mean": float(sum(pairwise_values) / max(1, len(pairwise_values))),
        "pairwise_error_correlation_by_pair": pairwise,
    }


def _empty_calibration_totals(bins: int = 15) -> dict[str, object]:
    return {
        "bins": int(bins),
        "samples": 0.0,
        "brier_sum": 0.0,
        "nll_sum": 0.0,
        "confidence_sum": 0.0,
        "bin_confidence_sum": [0.0 for _ in range(bins)],
        "bin_correct_sum": [0.0 for _ in range(bins)],
        "bin_count": [0.0 for _ in range(bins)],
    }


def _accumulate_calibration_totals(
    totals: dict[str, object],
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if logits.numel() == 0 or labels.numel() == 0:
        return
    bins = int(totals.get("bins", 15))
    probs = torch.softmax(logits, dim=1)
    confidence, predictions = probs.max(dim=1)
    correctness = (predictions == labels).float()
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).to(dtype=probs.dtype)
    totals["samples"] = float(totals.get("samples", 0.0)) + float(labels.numel())
    totals["brier_sum"] = float(totals.get("brier_sum", 0.0)) + float(((probs - one_hot) ** 2).sum(dim=1).sum().item())
    totals["nll_sum"] = float(totals.get("nll_sum", 0.0)) + float(F.cross_entropy(logits, labels, reduction="sum").item())
    totals["confidence_sum"] = float(totals.get("confidence_sum", 0.0)) + float(confidence.sum().item())

    bin_confidence_sum = totals["bin_confidence_sum"]
    bin_correct_sum = totals["bin_correct_sum"]
    bin_count = totals["bin_count"]
    assert isinstance(bin_confidence_sum, list)
    assert isinstance(bin_correct_sum, list)
    assert isinstance(bin_count, list)
    bin_indices = torch.clamp((confidence * bins).to(torch.long), max=bins - 1)
    for index in range(bins):
        mask = bin_indices == index
        count = int(mask.sum().item())
        if count <= 0:
            continue
        bin_confidence_sum[index] += float(confidence[mask].sum().item())
        bin_correct_sum[index] += float(correctness[mask].sum().item())
        bin_count[index] += float(count)


def _reduce_calibration_totals(totals: dict[str, object], device: torch.device) -> dict[str, object]:
    if not _distributed_enabled():
        return totals
    bins = int(totals.get("bins", 15))
    scalar_keys = ["samples", "brier_sum", "nll_sum", "confidence_sum"]
    reduced_scalars = _reduce_scalar_sums([float(totals.get(key, 0.0)) for key in scalar_keys], device)
    for key, value in zip(scalar_keys, reduced_scalars):
        totals[key] = value
    for key in ("bin_confidence_sum", "bin_correct_sum", "bin_count"):
        values = totals.get(key, [0.0 for _ in range(bins)])
        if not isinstance(values, list):
            values = [0.0 for _ in range(bins)]
        totals[key] = _reduce_scalar_sums([float(value) for value in values], device)
    return totals


def _finalize_calibration_totals(totals: dict[str, object]) -> dict[str, float | list[dict[str, float]]]:
    samples = float(totals.get("samples", 0.0))
    if samples <= 0:
        return _calibration_metrics(torch.empty((0, 1)), torch.empty(0, dtype=torch.long))
    bins = int(totals.get("bins", 15))
    bin_confidence_sum = totals.get("bin_confidence_sum", [0.0 for _ in range(bins)])
    bin_correct_sum = totals.get("bin_correct_sum", [0.0 for _ in range(bins)])
    bin_count = totals.get("bin_count", [0.0 for _ in range(bins)])
    assert isinstance(bin_confidence_sum, list)
    assert isinstance(bin_correct_sum, list)
    assert isinstance(bin_count, list)
    ece = 0.0
    reliability_bins: list[dict[str, float]] = []
    for index in range(bins):
        count = float(bin_count[index])
        if count <= 0:
            confidence = 0.0
            accuracy = 0.0
        else:
            confidence = float(bin_confidence_sum[index]) / count
            accuracy = float(bin_correct_sum[index]) / count
            ece += (count / samples) * abs(accuracy - confidence)
        reliability_bins.append(
            {
                "bin_start": index / bins,
                "bin_end": (index + 1) / bins,
                "count": count,
                "accuracy": accuracy,
                "confidence": confidence,
            }
        )
    return {
        "ece": float(ece),
        "brier": float(totals.get("brier_sum", 0.0)) / samples,
        "nll": float(totals.get("nll_sum", 0.0)) / samples,
        "mean_confidence": float(totals.get("confidence_sum", 0.0)) / samples,
        "reliability_bins": reliability_bins,
    }


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
    recognizer_ensemble: ExternalRecognizerEnsemble | None = None,
    progress_desc: str | None = None,
) -> dict[str, float]:
    model.eval()
    if recognizer_ensemble is not None:
        recognizer_ensemble.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    total_samples = 0
    total_correct = 0
    member_totals = _empty_member_totals()
    recognizer_member_totals = _empty_member_totals()
    ensemble_totals = _empty_ensemble_totals()
    recognizer_ensemble_totals = _empty_ensemble_totals()
    calibration_totals = _empty_calibration_totals()
    recognizer_calibration_totals = _empty_calibration_totals()
    batches = 0
    eval_model = _unwrap_model(model)

    iterator = loader
    if progress_desc is not None:
        iterator = tqdm(loader, desc=progress_desc, unit="batch", leave=False, dynamic_ncols=True, smoothing=0.1)

    for images, labels, _, _ in iterator:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if hasattr(eval_model, "predict_eval_outputs"):
            eval_outputs = eval_model.predict_eval_outputs(images)
            embeddings = eval_outputs["embeddings"]
            logits = eval_outputs["predict_logits"]
            member_logits = eval_outputs.get("member_logits")
        else:
            embeddings = eval_model.forward_embeddings(images)
            logits = eval_model.predict_logits(images)
            member_logits = eval_model.predict_member_logits(images) if hasattr(eval_model, "predict_member_logits") else None
        _accumulate_calibration_totals(calibration_totals, logits, labels)
        if isinstance(member_logits, dict) and member_logits:
            _accumulate_member_metrics(member_totals, member_logits, labels)
            _accumulate_ensemble_totals(ensemble_totals, member_logits, logits, labels)
        if recognizer_ensemble is not None:
            eval_recognizer_ensemble = _unwrap_model(recognizer_ensemble)
            recognizer_eval_outputs = eval_recognizer_ensemble.predict_eval_outputs_from_embeddings(embeddings)
            recognizer_logits = recognizer_eval_outputs["predict_logits"]
            recognizer_member_logits = recognizer_eval_outputs.get("member_logits")
            _accumulate_calibration_totals(recognizer_calibration_totals, recognizer_logits, labels)
            if isinstance(recognizer_member_logits, dict) and recognizer_member_logits:
                _accumulate_member_metrics(recognizer_member_totals, recognizer_member_logits, labels)
                _accumulate_ensemble_totals(
                    recognizer_ensemble_totals,
                    recognizer_member_logits,
                    recognizer_logits,
                    labels,
                )
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
    recognizer_member_totals = _reduce_member_totals(recognizer_member_totals, device)
    accuracy = float(total_correct) / float(max(1.0, total_samples))
    ensemble_metrics = _finalize_ensemble_totals(_reduce_ensemble_totals(ensemble_totals, device))
    calibration = _finalize_calibration_totals(_reduce_calibration_totals(calibration_totals, device))
    recognizer_ensemble_metrics: dict[str, object] = {}
    recognizer_calibration: dict[str, object] = {}
    if recognizer_ensemble is not None:
        recognizer_ensemble_metrics = _finalize_ensemble_totals(
            _reduce_ensemble_totals(recognizer_ensemble_totals, device)
        )
        recognizer_calibration = _finalize_calibration_totals(
            _reduce_calibration_totals(recognizer_calibration_totals, device)
        )
    return {
        "loss": total_loss / max(1.0, total_batches),
        "accuracy": accuracy,
        "samples": float(total_samples),
        "correct": float(total_correct),
        "member_metrics": _finalize_member_metrics(member_totals),
        "calibration": calibration,
        "ensemble": ensemble_metrics,
        "recognizer_metrics": _finalize_member_metrics(recognizer_member_totals),
        "recognizer_calibration": recognizer_calibration,
        "recognizer_ensemble": recognizer_ensemble_metrics,
    }


def evaluate_robust(
    *,
    model: TrainableRecognizer,
    loader,
    device: torch.device,
    policy: AttackPolicy | None,
    surrogates: dict[str, SurrogateWrapper],
    image_size: int,
    recognizer_ensemble: ExternalRecognizerEnsemble | None = None,
    attack_chunk_size: int | None = None,
    progress_desc: str | None = None,
) -> dict[str, float]:
    if policy is None:
        return {"loss": 0.0, "accuracy": 0.0, "consistency": 0.0, "cached_hits": 0.0}

    model.eval()
    if recognizer_ensemble is not None:
        recognizer_ensemble.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_consistency = 0.0
    total_cached_hits = 0.0
    total_samples = 0
    total_correct = 0
    total_clean_correct = 0
    total_success_from_clean_correct = 0
    member_totals = _empty_member_totals()
    recognizer_member_totals = _empty_member_totals()
    ensemble_totals = _empty_ensemble_totals()
    recognizer_ensemble_totals = _empty_ensemble_totals()
    calibration_totals = _empty_calibration_totals()
    recognizer_calibration_totals = _empty_calibration_totals()
    batches = 0
    eval_model = _unwrap_model(model)

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
            recognizer_ensemble=recognizer_ensemble,
            device=device,
            attack_chunk_size=attack_chunk_size,
        )

        with torch.no_grad():
            if hasattr(eval_model, "predict_eval_outputs"):
                clean_outputs = eval_model.predict_eval_outputs(images)
                clean_embeddings = clean_outputs["embeddings"]
                clean_predict_logits = clean_outputs["predict_logits"]
            else:
                clean_embeddings = eval_model.forward_embeddings(images)
                clean_predict_logits = eval_model.predict_logits_from_embeddings(clean_embeddings)
            adv_outputs = model(attack_result.images, labels)
            adv_logits = adv_outputs["logits"]
            adv_loss_labels = adv_outputs.get("loss_labels", labels)
            adv_embeddings = adv_outputs["embeddings"]
            if hasattr(eval_model, "predict_eval_outputs"):
                adv_eval_outputs = eval_model.predict_eval_outputs(attack_result.images)
                adv_predict_logits = adv_eval_outputs["predict_logits"]
                member_logits = adv_eval_outputs.get("member_logits") or {}
            else:
                adv_predict_logits = adv_outputs["predict_logits"]
                member_logits = _member_logits_from_training_outputs(adv_outputs)
            member_loss_labels = _member_loss_labels_from_training_outputs(adv_outputs)
            if member_logits:
                member_label_source = labels if hasattr(eval_model, "predict_eval_outputs") else (member_loss_labels or labels)
                _accumulate_member_metrics(
                    member_totals,
                    member_logits,
                    member_label_source,
                )
                if isinstance(member_label_source, torch.Tensor):
                    _accumulate_ensemble_totals(ensemble_totals, member_logits, adv_predict_logits, member_label_source)
            if recognizer_ensemble is not None:
                eval_recognizer_ensemble = _unwrap_model(recognizer_ensemble)
                recognizer_eval_outputs = eval_recognizer_ensemble.predict_eval_outputs_from_embeddings(adv_embeddings)
                recognizer_logits = recognizer_eval_outputs["predict_logits"]
                recognizer_member_logits = recognizer_eval_outputs.get("member_logits")
                _accumulate_calibration_totals(recognizer_calibration_totals, recognizer_logits, labels)
                if isinstance(recognizer_member_logits, dict) and recognizer_member_logits:
                    _accumulate_member_metrics(recognizer_member_totals, recognizer_member_logits, labels)
                    _accumulate_ensemble_totals(
                        recognizer_ensemble_totals,
                        recognizer_member_logits,
                        recognizer_logits,
                        labels,
                    )

        loss = F.cross_entropy(adv_logits, adv_loss_labels)
        _accumulate_calibration_totals(calibration_totals, adv_predict_logits, labels)
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
    recognizer_member_totals = _reduce_member_totals(recognizer_member_totals, device)
    asr = float(total_success_from_clean_correct) / float(max(1.0, total_clean_correct))
    accuracy = float(total_correct) / float(max(1.0, total_samples))
    ensemble_metrics = _finalize_ensemble_totals(_reduce_ensemble_totals(ensemble_totals, device))
    calibration = _finalize_calibration_totals(_reduce_calibration_totals(calibration_totals, device))
    recognizer_ensemble_metrics: dict[str, object] = {}
    recognizer_calibration: dict[str, object] = {}
    if recognizer_ensemble is not None:
        recognizer_ensemble_metrics = _finalize_ensemble_totals(
            _reduce_ensemble_totals(recognizer_ensemble_totals, device)
        )
        recognizer_calibration = _finalize_calibration_totals(
            _reduce_calibration_totals(recognizer_calibration_totals, device)
        )
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
        "recognizer_metrics": _finalize_member_metrics(recognizer_member_totals),
        "recognizer_calibration": recognizer_calibration,
        "recognizer_ensemble": recognizer_ensemble_metrics,
    }


def evaluate_robust_all(
    *,
    model: TrainableRecognizer,
    loader,
    device: torch.device,
    policies: list[AttackPolicy],
    surrogates: dict[str, SurrogateWrapper],
    recognizer_ensemble: ExternalRecognizerEnsemble | None = None,
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
            recognizer_ensemble=recognizer_ensemble,
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


def _threshold_stats(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor | int]:
    score_tensor = scores.to(dtype=torch.float32)
    label_tensor = labels.to(device=score_tensor.device, dtype=torch.long)
    order = torch.argsort(score_tensor, descending=True)
    sorted_scores = score_tensor.index_select(0, order)
    sorted_labels = label_tensor.index_select(0, order)

    positive_flags = (sorted_labels == 1).to(torch.float32)
    negative_flags = (sorted_labels == 0).to(torch.float32)
    cumulative_tp = torch.cumsum(positive_flags, dim=0)
    cumulative_fp = torch.cumsum(negative_flags, dim=0)
    threshold_end_indices = torch.nonzero(
        torch.cat(
            [
                sorted_scores[:-1] != sorted_scores[1:],
                torch.ones(1, dtype=torch.bool, device=score_tensor.device),
            ]
        ),
        as_tuple=False,
    ).flatten()

    thresholds = sorted_scores.index_select(0, threshold_end_indices)
    tp = cumulative_tp.index_select(0, threshold_end_indices)
    fp = cumulative_fp.index_select(0, threshold_end_indices)
    pos_count = int((label_tensor == 1).sum().item())
    neg_count = int((label_tensor == 0).sum().item())
    return {
        "thresholds": thresholds,
        "tp": tp,
        "fp": fp,
        "pos_count": pos_count,
        "neg_count": neg_count,
    }


def _best_threshold_accuracy(
    scores: torch.Tensor,
    labels: torch.Tensor,
    stats: dict[str, torch.Tensor | int] | None = None,
) -> tuple[float, float]:
    if scores.numel() == 0:
        return 0.0, 0.0
    stats = stats or _threshold_stats(scores, labels)
    correct = stats["tp"] + (stats["neg_count"] - stats["fp"])
    accuracies = correct / float(max(1, scores.numel()))
    best_index = int(torch.argmax(accuracies).item())
    return float(accuracies[best_index].item()), float(stats["thresholds"][best_index].item())


def _eer(
    scores: torch.Tensor,
    labels: torch.Tensor,
    stats: dict[str, torch.Tensor | int] | None = None,
) -> float:
    if scores.numel() == 0:
        return 0.0
    stats = stats or _threshold_stats(scores, labels)
    far = stats["fp"] / float(max(1, stats["neg_count"]))
    frr = 1.0 - (stats["tp"] / float(max(1, stats["pos_count"])))
    gap = torch.abs(far - frr)
    best_index = int(torch.argmin(gap).item())
    return float((0.5 * (far[best_index] + frr[best_index])).item())


def _roc_curve(
    scores: torch.Tensor,
    labels: torch.Tensor,
    stats: dict[str, torch.Tensor | int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.numel() == 0:
        zeros = torch.zeros(1, dtype=torch.float32, device=scores.device)
        return zeros, zeros, zeros
    stats = stats or _threshold_stats(scores, labels)
    fpr = stats["fp"] / float(max(1, stats["neg_count"]))
    tpr = stats["tp"] / float(max(1, stats["pos_count"]))
    fpr_tensor = torch.cat([torch.zeros(1, dtype=torch.float32, device=scores.device), fpr, torch.ones(1, dtype=torch.float32, device=scores.device)])
    tpr_tensor = torch.cat([torch.zeros(1, dtype=torch.float32, device=scores.device), tpr, torch.ones(1, dtype=torch.float32, device=scores.device)])
    threshold_tensor = torch.cat(
        [
            torch.full((1,), float("inf"), dtype=torch.float32, device=scores.device),
            stats["thresholds"].to(dtype=torch.float32),
            torch.full((1,), float("-inf"), dtype=torch.float32, device=scores.device),
        ]
    )
    return fpr_tensor, tpr_tensor, threshold_tensor


def _roc_auc(
    scores: torch.Tensor,
    labels: torch.Tensor,
    stats: dict[str, torch.Tensor | int] | None = None,
) -> float:
    fpr, tpr, _ = _roc_curve(scores, labels, stats=stats)
    if fpr.numel() <= 1:
        return 0.0
    return float(torch.trapz(tpr, fpr).item())


def _tar_at_far(
    scores: torch.Tensor,
    labels: torch.Tensor,
    far_targets: tuple[float, ...] = _DEFAULT_FAR_TARGETS,
    stats: dict[str, torch.Tensor | int] | None = None,
) -> dict[str, float | bool | None]:
    if scores.numel() == 0:
        return {
            f"tar@far={target:g}": 0.0 for target in far_targets
        } | {
            f"threshold@far={target:g}": None for target in far_targets
        } | {
            f"far_supported@{target:g}": False for target in far_targets
        }

    stats = stats or _threshold_stats(scores, labels)
    neg_count = int(stats["neg_count"])
    far = stats["fp"] / float(max(1, stats["neg_count"]))
    tar = stats["tp"] / float(max(1, stats["pos_count"]))
    metrics: dict[str, float | bool | None] = {}
    for target in far_targets:
        supported = neg_count >= max(1, math.ceil(1.0 / target))
        eligible = torch.nonzero(far <= target, as_tuple=False).flatten()
        if eligible.numel() == 0:
            best_tar = 0.0
            best_threshold = None
        else:
            eligible_tar = tar.index_select(0, eligible)
            local_index = int(torch.argmax(eligible_tar).item())
            best_index = int(eligible[local_index].item())
            best_tar = float(tar[best_index].item())
            best_threshold = float(stats["thresholds"][best_index].item())
        metrics[f"tar@far={target:g}"] = best_tar
        metrics[f"threshold@far={target:g}"] = best_threshold
        metrics[f"far_supported@{target:g}"] = supported
    return metrics


def verification_metrics_from_scores(
    scores: torch.Tensor,
    labels: torch.Tensor,
    far_targets: tuple[float, ...] = _DEFAULT_FAR_TARGETS,
    progress=None,
) -> dict[str, float | bool | None]:
    progress = progress or _NullProgress()
    progress.set_description("Verification metrics: threshold")
    threshold_stats = _threshold_stats(scores, labels) if scores.numel() else None
    best_acc, best_threshold = _best_threshold_accuracy(scores, labels, stats=threshold_stats)
    progress.update(1)
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    progress.set_description("Verification metrics: eer")
    eer = _eer(scores, labels, stats=threshold_stats)
    progress.update(1)
    progress.set_description("Verification metrics: auc")
    roc_auc = _roc_auc(scores, labels, stats=threshold_stats)
    progress.update(1)
    metrics: dict[str, float | bool | None] = {
        "pairs": float(scores.numel()),
        "positive_pairs": float(positive_scores.numel()),
        "negative_pairs": float(negative_scores.numel()),
        "best_accuracy": best_acc,
        "best_threshold": best_threshold,
        "eer": eer,
        "roc_auc": roc_auc,
        "mean_positive_cosine": float(positive_scores.mean().item()) if positive_scores.numel() else 0.0,
        "mean_negative_cosine": float(negative_scores.mean().item()) if negative_scores.numel() else 0.0,
    }
    progress.set_description("Verification metrics: tar")
    metrics.update(_tar_at_far(scores, labels, far_targets, stats=threshold_stats))
    progress.update(1)
    return metrics


@torch.no_grad()
def evaluate_verification_pairs(
    *,
    model: TrainableRecognizer,
    pairs_path: Path,
    device: torch.device,
    progress_desc: str | None = None,
) -> dict[str, float]:
    pair_data = dict(**__import__("numpy").load(pairs_path, allow_pickle=True))
    img1_paths = [Path(item) for item in pair_data["img1_paths"].tolist()]
    img2_paths = [Path(item) for item in pair_data["img2_paths"].tolist()]
    labels = torch.as_tensor(pair_data["labels"].astype("int64"))
    unique_paths = sorted({path for path in img1_paths + img2_paths})
    model.eval()
    eval_model = _unwrap_model(model)
    if not unique_paths:
        return verification_metrics_from_scores(torch.empty(0), labels)

    batch_size = _VERIFICATION_BATCH_SIZE_CUDA if device.type == "cuda" else _VERIFICATION_BATCH_SIZE_CPU
    num_workers = _verification_num_workers(device)
    dataset = _VerificationImageDataset(unique_paths, image_size=_verification_image_size(eval_model))
    show_progress = progress_desc is not None and (not _distributed_enabled() or dist.get_rank() == 0)
    local_indices: list[int] = []
    local_embedding_rows: list[torch.Tensor] = []
    progress_world_size = dist.get_world_size() if _distributed_enabled() else 1
    while True:
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": bool(num_workers > 0),
        }
        if _distributed_enabled():
            loader_kwargs["sampler"] = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
                drop_last=False,
            )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = _VERIFICATION_PREFETCH_FACTOR
        loader = DataLoader(dataset, **loader_kwargs)
        embedding_progress = None
        local_indices = []
        local_embedding_rows = []
        try:
            if show_progress:
                embedding_progress = tqdm(
                    total=len(unique_paths),
                    desc=progress_desc,
                    unit="img",
                    leave=False,
                    dynamic_ncols=True,
                    smoothing=0.1,
                )
            for images, indices in loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                embeddings = eval_model.forward_embeddings(images).detach().cpu()
                for row, item_index in zip(embeddings, indices.tolist()):
                    local_indices.append(int(item_index))
                    local_embedding_rows.append(row)
                if embedding_progress is not None:
                    remaining = max(0, len(unique_paths) - int(embedding_progress.n))
                    embedding_progress.update(min(remaining, int(indices.numel()) * progress_world_size))
            if embedding_progress is not None:
                embedding_progress.close()
            break
        except RuntimeError as exc:
            if embedding_progress is not None:
                embedding_progress.close()
            oom = "out of memory" in str(exc).lower()
            if device.type != "cuda" or not oom or batch_size <= 64:
                raise
            batch_size = max(64, batch_size // 2)
            if show_progress:
                tqdm.write(f"[INFO] Verification OOM; retrying with batch_size={batch_size}")
            del loader
            _clear_verification_device_caches(device)

    embedding_rows: list[torch.Tensor | None] = [None] * len(unique_paths)
    if _distributed_enabled():
        gather_progress = None
        if show_progress:
            gather_progress = tqdm(
                total=3,
                desc=f"{progress_desc} gather",
                unit="step",
                leave=False,
                dynamic_ncols=True,
            )
        embedding_dim_tensor = torch.as_tensor(
            [int(local_embedding_rows[0].numel()) if local_embedding_rows else 0],
            dtype=torch.long,
            device=device,
        )
        dist.all_reduce(embedding_dim_tensor, op=dist.ReduceOp.MAX)
        embedding_dim = int(embedding_dim_tensor.item())
        if gather_progress is not None:
            gather_progress.update(1)
        local_count = len(local_indices)
        count_tensor = torch.as_tensor([local_count], dtype=torch.long, device=device)
        gathered_counts = [torch.zeros_like(count_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_counts, count_tensor)
        counts = [int(item.item()) for item in gathered_counts]
        max_count = max(counts) if counts else 0
        if gather_progress is not None:
            gather_progress.update(1)

        index_tensor = torch.full((max_count,), -1, dtype=torch.long, device=device)
        if local_count > 0:
            index_tensor[:local_count] = torch.as_tensor(local_indices, dtype=torch.long, device=device)

        embedding_tensor_local = torch.zeros((max_count, embedding_dim), dtype=torch.float32, device=device)
        if local_count > 0 and embedding_dim > 0:
            embedding_tensor_local[:local_count] = torch.stack(local_embedding_rows, dim=0).to(device, non_blocking=device.type == "cuda")

        gathered_indices = [torch.empty_like(index_tensor) for _ in range(dist.get_world_size())]
        gathered_embeddings = [torch.empty_like(embedding_tensor_local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_indices, index_tensor)
        dist.all_gather(gathered_embeddings, embedding_tensor_local)
        if gather_progress is not None:
            gather_progress.update(1)
            gather_progress.close()

        for rank_count, rank_indices, rank_embeddings in zip(counts, gathered_indices, gathered_embeddings):
            if rank_count <= 0:
                continue
            for source_index, row in zip(rank_indices[:rank_count].tolist(), rank_embeddings[:rank_count].cpu()):
                embedding_rows[source_index] = row
    else:
        for row, item_index in zip(local_embedding_rows, local_indices):
            embedding_rows[item_index] = row

    if any(item is None for item in embedding_rows):
        raise RuntimeError("Verification embedding extraction did not cover every unique path.")

    embedding_tensor = torch.stack([item for item in embedding_rows if item is not None], dim=0)
    normalized_embeddings = F.normalize(embedding_tensor, dim=1)
    path_to_index = {path: index for index, path in enumerate(unique_paths)}
    pair_index_1 = torch.as_tensor([path_to_index[path] for path in img1_paths], dtype=torch.long)
    pair_index_2 = torch.as_tensor([path_to_index[path] for path in img2_paths], dtype=torch.long)

    num_pairs = int(pair_index_1.numel())
    if _distributed_enabled():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        score_start = (num_pairs * rank) // world_size
        score_end = (num_pairs * (rank + 1)) // world_size
    else:
        world_size = 1
        rank = 0
        score_start = 0
        score_end = num_pairs

    score_chunks: list[torch.Tensor] = []
    score_ranges = range(score_start, score_end, _VERIFICATION_SCORE_CHUNK)
    if show_progress:
        score_ranges = tqdm(
            score_ranges,
            desc=f"{progress_desc} score",
            unit="chunk",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.1,
        )
    for start in score_ranges:
        end = min(score_end, start + _VERIFICATION_SCORE_CHUNK)
        emb_a = normalized_embeddings.index_select(0, pair_index_1[start:end]).to(device, non_blocking=device.type == "cuda")
        emb_b = normalized_embeddings.index_select(0, pair_index_2[start:end]).to(device, non_blocking=device.type == "cuda")
        score_chunks.append(F.cosine_similarity(emb_a, emb_b, dim=1).detach().cpu())
    local_score_tensor = torch.cat(score_chunks, dim=0) if score_chunks else torch.empty(0)
    if _distributed_enabled():
        local_count = torch.as_tensor([local_score_tensor.numel()], dtype=torch.long, device=device)
        gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(item.item()) for item in gathered_counts]
        max_count = max(counts) if counts else 0
        local_scores_device = torch.zeros((max_count,), dtype=torch.float32, device=device)
        if local_score_tensor.numel() > 0:
            local_scores_device[: local_score_tensor.numel()] = local_score_tensor.to(device, non_blocking=device.type == "cuda")
        gathered_scores = [torch.empty_like(local_scores_device) for _ in range(world_size)]
        dist.all_gather(gathered_scores, local_scores_device)
        score_tensor = torch.cat(
            [rank_scores[:rank_count].cpu() for rank_scores, rank_count in zip(gathered_scores, counts)],
            dim=0,
        )
    else:
        score_tensor = local_score_tensor
    metrics_progress = None
    if show_progress:
        metrics_progress = tqdm(
            total=4,
            desc=f"{progress_desc} metrics",
            unit="step",
            leave=False,
            dynamic_ncols=True,
        )
    result = verification_metrics_from_scores(score_tensor, labels, progress=metrics_progress)
    if metrics_progress is not None:
        metrics_progress.close()
    return result
