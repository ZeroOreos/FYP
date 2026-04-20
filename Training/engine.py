from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from tqdm.auto import tqdm

from Training.attacks import build_attack_policy_sampler, generate_attack_batch
from Training.config import EnsembleTrainingConfig, save_config_snapshot
from Training.dataset import build_dataloaders, build_eval_loader
from Training.evaluate import choose_eval_policy, evaluate_clean, evaluate_robust, evaluate_robust_all, evaluate_verification_pairs
from Training.losses import classification_accuracy, embedding_consistency_loss
from Training.pairing import HardPairMiningResult, mine_hard_pairs
from Training.recognizers import build_surrogates, build_target_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def _is_primary() -> bool:
    return _rank() == 0


def maybe_init_distributed(config: EnsembleTrainingConfig) -> bool:
    if not config.use_distributed or _world_size() <= 1:
        return False
    if dist.is_initialized():
        return True
    backend = config.distributed_backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"
    if backend == "nccl":
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("NCCL backend requested but no CUDA devices are visible to PyTorch.")
        local_rank = _local_rank()
        if local_rank < 0 or local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is out of range for {device_count} visible CUDA device(s)."
            )
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return True


def maybe_wrap_ddp(model: torch.nn.Module, device: torch.device, use_distributed: bool):
    if not use_distributed:
        return model
    if device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    return DistributedDataParallel(model)


def _unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _make_optimizer(config: EnsembleTrainingConfig, model: torch.nn.Module):
    if config.optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{config.optimizer_name}'.")


def _make_scheduler(config: EnsembleTrainingConfig, optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.lr_milestones,
        gamma=config.lr_gamma,
    )


def _autocast_context(config: EnsembleTrainingConfig, device: torch.device):
    if not config.use_mixed_precision:
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _save_checkpoint(output_dir: Path, epoch: int, model, optimizer, scheduler, scaler, history: list[dict]) -> Path:
    checkpoint_path = output_dir / "checkpoints" / f"epoch_{epoch:03d}.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    base_model = _unwrap_model(model)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if getattr(scaler, "is_enabled", lambda: False)() else None,
            "history": history,
        },
        checkpoint_path,
    )
    return checkpoint_path


def _write_history(output_dir: Path, history: list[dict]) -> None:
    history_path = output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")


def _capture_repro_state(output_dir: Path) -> None:
    payload: dict[str, Any] = {
        "python_version": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
    }
    try:
        import torchvision  # type: ignore

        payload["torchvision_version"] = torchvision.__version__
    except Exception as exc:
        payload["torchvision_version_error"] = str(exc)

    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    payload["cudnn_available"] = bool(torch.backends.cudnn.is_available())
    payload["cudnn_version"] = torch.backends.cudnn.version()
    payload["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    payload["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    payload["cuda_tf32_matmul"] = bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))
    payload["cudnn_tf32"] = bool(getattr(torch.backends.cudnn, "allow_tf32", False))
    if torch.cuda.is_available():
        payload["cuda_device_count"] = torch.cuda.device_count()
        payload["cuda_device_names"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        payload["cuda_device_capabilities"] = [
            list(torch.cuda.get_device_capability(index)) for index in range(torch.cuda.device_count())
        ]

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=output_dir.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        payload["git_commit"] = git_commit
    except Exception as exc:
        payload["git_commit_error"] = str(exc)

    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        freeze_path = output_dir / "pip_freeze.txt"
        freeze_path.write_text(pip_freeze, encoding="utf-8")
        payload["pip_freeze_path"] = str(freeze_path)
        payload["pip_freeze_sha256"] = hashlib.sha256(pip_freeze.encode("utf-8")).hexdigest()
    except Exception as exc:
        payload["pip_freeze_error"] = str(exc)

    with (output_dir / "repro_state.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


class RunLogger:
    def __init__(self, output_dir: Path) -> None:
        self.log_path = output_dir / "train.log"
        self.metrics_path = output_dir / "metrics.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(message)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")

    def metric(self, payload: dict[str, object]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def _curriculum_progress(config: EnsembleTrainingConfig, epoch: int) -> float:
    if not config.curriculum_enabled:
        return 0.0
    if config.curriculum_warmup_epochs <= 1:
        return 1.0
    clamped_epoch = min(max(epoch - 1, 0), config.curriculum_warmup_epochs - 1)
    return clamped_epoch / float(config.curriculum_warmup_epochs - 1)


def _resolve_epoch_schedule(config: EnsembleTrainingConfig, epoch: int) -> dict[str, float]:
    if config.training_depth_mode.strip().lower() == "three_stage":
        clean_warmup_epochs = max(0, int(config.clean_warmup_epochs))
        shallow_adv_epochs = max(0, int(config.shallow_adv_epochs))
        shallow_end_epoch = clean_warmup_epochs + shallow_adv_epochs
        if epoch <= clean_warmup_epochs:
            stage_name = "clean_warmup"
            attacks_enabled = False
            clean_fraction = 1.0
            hard_pair_fraction = 0.0
            hard_pair_weight = 0.0
        elif epoch <= shallow_end_epoch:
            stage_name = "shallow_adv"
            attacks_enabled = (not config.clean_only) and bool(config.enabled_attackers()) and epoch >= max(1, config.attack_start_epoch)
            clean_fraction = float(config.shallow_clean_fraction)
            hard_pair_fraction = float(config.hard_pair_fraction) * 0.5
            hard_pair_weight = float(config.hard_pair_weight) * 0.5
        else:
            stage_name = "full_adv"
            attacks_enabled = (not config.clean_only) and bool(config.enabled_attackers()) and epoch >= max(1, config.attack_start_epoch)
            clean_fraction = float(config.clean_fraction)
            hard_pair_fraction = float(config.hard_pair_fraction)
            hard_pair_weight = float(config.hard_pair_weight)
        return {
            "progress": 0.0,
            "stage_name": stage_name,
            "clean_fraction": clean_fraction,
            "hard_pair_fraction": hard_pair_fraction,
            "hard_pair_weight": hard_pair_weight,
            "attacks_enabled": float(attacks_enabled),
        }

    progress = _curriculum_progress(config, epoch)
    attacks_enabled = (not config.clean_only) and bool(config.enabled_attackers()) and epoch >= max(1, config.attack_start_epoch)
    if not config.curriculum_enabled:
        return {
            "progress": 0.0,
            "clean_fraction": float(config.clean_fraction),
            "hard_pair_fraction": float(config.hard_pair_fraction),
            "hard_pair_weight": float(config.hard_pair_weight),
            "attacks_enabled": float(attacks_enabled),
        }
    return {
        "progress": progress,
        "stage_name": "legacy_curriculum",
        "clean_fraction": _lerp(float(config.clean_fraction), float(config.curriculum_clean_fraction_end), progress),
        "hard_pair_fraction": _lerp(float(config.hard_pair_fraction), float(config.curriculum_hard_pair_fraction_end), progress),
        "hard_pair_weight": _lerp(float(config.hard_pair_weight), float(config.curriculum_hard_pair_weight_end), progress),
        "attacks_enabled": float(attacks_enabled),
    }


def _stage_attack_policies(
    config: EnsembleTrainingConfig,
    schedule: dict[str, float],
) -> list:
    if not bool(schedule["attacks_enabled"]):
        return []
    stage_name = str(schedule.get("stage_name", ""))
    policies = config.enabled_attackers()
    if stage_name != "shallow_adv":
        return policies

    allowed_names = {name.strip().lower() for name in config.shallow_attack_names if name.strip()}
    if allowed_names:
        stage_policies = [policy for policy in policies if policy.name.strip().lower() in allowed_names]
    else:
        stage_policies = policies

    scaled_policies = []
    for policy in stage_policies:
        scaled_policies.append(
            replace(
                policy,
                eps=float(policy.eps) * float(config.shallow_attack_eps_scale),
                alpha=float(policy.alpha) * float(config.shallow_attack_alpha_scale),
                steps=max(1, int(round(float(policy.steps) * float(config.shallow_attack_step_scale)))),
                restarts=min(max(1, int(config.shallow_attack_restart_cap)), max(1, int(policy.restarts))),
            )
        )
    return scaled_policies


def _apply_epoch_learning_rate(
    optimizer,
    scheduler,
    config: EnsembleTrainingConfig,
    epoch: int,
) -> None:
    scheduled_lrs = scheduler.get_last_lr()
    if config.lr_warmup_epochs <= 0:
        for group, lr in zip(optimizer.param_groups, scheduled_lrs):
            group["lr"] = lr
        return
    warmup_scale = min(1.0, float(epoch) / float(max(1, config.lr_warmup_epochs)))
    for group, lr in zip(optimizer.param_groups, scheduled_lrs):
        group["lr"] = lr * warmup_scale


def _mine_batch_hard_pairs(
    *,
    config: EnsembleTrainingConfig,
    images: torch.Tensor,
    labels: torch.Tensor,
    clean_embeddings: torch.Tensor,
    surrogates: dict[str, Any],
    schedule: dict[str, float],
) -> HardPairMiningResult:
    if config.pairing_strategy.strip().lower() in {"", "none", "off"}:
        return HardPairMiningResult(strategy="none")

    surrogate_embeddings: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, surrogate in surrogates.items():
            if name == "target" or name not in config.pairing_surrogate_weights:
                continue
            surrogate_embeddings[name] = surrogate.embed(images).detach()

    return mine_hard_pairs(
        target_embeddings=clean_embeddings.detach(),
        labels=labels.detach(),
        surrogate_embeddings=surrogate_embeddings,
        strategy=config.pairing_strategy,
        hard_pair_fraction=schedule["hard_pair_fraction"],
        min_pairs=config.hard_pair_min_pairs,
        max_pairs=config.hard_pair_max_pairs,
        surrogate_weights=config.pairing_surrogate_weights,
    )


def _joint_member_outputs(
    model: Any,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]] | None:
    if not hasattr(model, "forward_member_outputs"):
        return None
    raw_outputs = model.forward_member_outputs(images, labels)
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for name, item in raw_outputs.items():
        loss_per_sample = F.cross_entropy(item["logits"], labels, reduction="none")
        outputs[name] = {
            **item,
            "loss_per_sample": loss_per_sample,
            "loss": loss_per_sample.mean(),
        }
    return outputs


def _joint_member_weights(model: Any, member_outputs: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
    weights = getattr(model, "member_weights", None)
    if not weights:
        uniform = 1.0 / float(max(1, len(member_outputs)))
        return {name: uniform for name in member_outputs}
    total = sum(float(weights.get(name, 0.0)) for name in member_outputs)
    if total <= 0:
        uniform = 1.0 / float(max(1, len(member_outputs)))
        return {name: uniform for name in member_outputs}
    return {name: float(weights.get(name, 0.0)) / total for name in member_outputs}


def _weighted_member_loss_per_sample(
    model: Any,
    member_outputs: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    weights = _joint_member_weights(model, member_outputs)
    losses = [weights[name] * item["loss_per_sample"] for name, item in member_outputs.items()]
    return torch.stack(losses, dim=0).sum(dim=0)


def _average_member_consistency(
    model: Any,
    clean_member_outputs: dict[str, dict[str, torch.Tensor]],
    adv_member_outputs: dict[str, dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, float]]:
    per_member: dict[str, float] = {}
    consistency_terms = []
    weights = _joint_member_weights(model, clean_member_outputs)
    for name, clean_item in clean_member_outputs.items():
        adv_item = adv_member_outputs[name]
        value = embedding_consistency_loss(clean_item["embeddings"], adv_item["embeddings"])
        consistency_terms.append(weights[name] * value)
        per_member[name] = float(value.item())
    if not consistency_terms:
        zero = torch.zeros((), device=next(iter(clean_member_outputs.values()))["embeddings"].device)
        return zero, per_member
    return torch.stack(consistency_terms).sum(), per_member


def _accumulate_member_epoch_metric(
    totals: dict[str, dict[str, dict[str, float]]],
    *,
    name: str,
    key: str,
    value: float,
) -> None:
    member_bucket = totals.setdefault(name, {})
    metric_bucket = member_bucket.setdefault(key, {"sum": 0.0, "count": 0.0})
    metric_bucket["sum"] += float(value)
    metric_bucket["count"] += 1.0


def _finalize_member_epoch_metrics(
    totals: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for name, metrics in totals.items():
        finalized[name] = {}
        for key, bucket in metrics.items():
            count = max(1.0, bucket["count"])
            finalized[name][key] = bucket["sum"] / count
    return finalized


def _progress_bar(
    iterable,
    *,
    total: int | None = None,
    desc: str,
    enabled: bool,
    leave: bool = True,
    unit: str = "batch",
):
    if not enabled:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        smoothing=0.1,
        miniters=1,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )


def _manual_progress_bar(
    *,
    total: int,
    desc: str,
    enabled: bool,
    leave: bool = True,
    unit: str = "img",
):
    if not enabled:
        return None
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        smoothing=0.1,
        miniters=1,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )


def _policy_names(policies: list[Any]) -> str:
    if not policies:
        return "none"
    return ",".join(policy.name for policy in policies)


def _estimate_epoch_attack_work(
    schedule: dict[str, float],
    policies: list[Any],
) -> int:
    if not bool(schedule["attacks_enabled"]) or not policies:
        return 0
    max_steps = max(int(getattr(policy, "steps", 1)) * max(1, int(getattr(policy, "restarts", 1))) for policy in policies)
    return max_steps


def _should_run_eval(epoch: int, total_epochs: int, every: int) -> bool:
    if every <= 0:
        return epoch == total_epochs
    return epoch == total_epochs or (epoch % every == 0)


def _clear_device_caches(device: torch.device) -> None:
    if device.type == "mps":
        gc.collect()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def _cuda_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated_mb": float(torch.cuda.memory_allocated(index)) / (1024.0 * 1024.0),
        "reserved_mb": float(torch.cuda.memory_reserved(index)) / (1024.0 * 1024.0),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated(index)) / (1024.0 * 1024.0),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved(index)) / (1024.0 * 1024.0),
    }


def _format_cuda_memory_stats(device: torch.device) -> str:
    stats = _cuda_memory_stats(device)
    if not stats:
        return "cuda_memory=unavailable"
    return (
        f"alloc={stats['allocated_mb']:.1f}MB "
        f"reserved={stats['reserved_mb']:.1f}MB "
        f"max_alloc={stats['max_allocated_mb']:.1f}MB "
        f"max_reserved={stats['max_reserved_mb']:.1f}MB"
    )


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(index)


def _is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _is_device_mismatch(exc: RuntimeError) -> bool:
    message = str(exc)
    return "Expected all tensors to be on the same device" in message or "Expected all tensors to be on the same" in message


def _log_cuda_runtime_diagnostics(
    logger: RunLogger,
    *,
    config: EnsembleTrainingConfig,
    device: torch.device,
) -> None:
    logger.info(
        "[CUDA]"
        f" torch={torch.__version__}"
        f" cuda={torch.version.cuda}"
        f" cudnn={torch.backends.cudnn.version()}"
        f" available={torch.cuda.is_available()}"
    )
    if device.type != "cuda":
        return
    index = device.index if device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    logger.info(
        "[CUDA]"
        f" device_index={index}"
        f" device_name={torch.cuda.get_device_name(index)}"
        f" capability={capability[0]}.{capability[1]}"
        f" visible_devices={torch.cuda.device_count()}"
        f" local_rank={_local_rank()}"
    )
    logger.info(
        "[CUDA]"
        f" cudnn_available={torch.backends.cudnn.is_available()}"
        f" benchmark={torch.backends.cudnn.benchmark}"
        f" deterministic={torch.backends.cudnn.deterministic}"
        f" tf32_matmul={getattr(torch.backends.cuda.matmul, 'allow_tf32', False)}"
        f" tf32_cudnn={getattr(torch.backends.cudnn, 'allow_tf32', False)}"
    )
    logger.info(
        "[CUDA]"
        f" batch_size={config.batch_size}"
        f" image_size={config.image_size}"
        f" mixed_precision={config.use_mixed_precision}"
        f" gradient_checkpointing={config.use_gradient_checkpointing}"
    )


def _raise_batch_runtime_error(
    *,
    exc: RuntimeError,
    config: EnsembleTrainingConfig,
    device: torch.device,
    epoch: int,
    batch_index: int,
) -> None:
    if _is_cuda_oom(exc):
        raise RuntimeError(
            "CUDA OOM during training batch "
            f"(epoch={epoch}, batch={batch_index}, batch_size={config.batch_size}, "
            f"image_size={config.image_size}, mixed_precision={config.use_mixed_precision}). "
            f"{_format_cuda_memory_stats(device)}"
        ) from exc
    if _is_device_mismatch(exc):
        raise RuntimeError(
            "Device mismatch during training batch "
            f"(epoch={epoch}, batch={batch_index}, model_device={device}). "
            "Check tensor creation and transfer paths for mixed CPU/CUDA tensors."
        ) from exc
    raise exc


def run_training(config: EnsembleTrainingConfig) -> dict[str, object]:
    set_seed(config.seed + _rank())
    distributed = maybe_init_distributed(config)

    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir)
    if _is_primary():
        save_config_snapshot(config, output_dir / "config.snapshot.json")
        _capture_repro_state(output_dir)

    train_ds, val_ds, train_loader, val_loader, class_to_idx = build_dataloaders(
        train_dir=config.resolved_train_dir(),
        val_dir=config.resolved_val_dir(),
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        distributed=distributed,
        dataset_fraction=config.dataset_fraction,
        dataset_subset_seed=config.dataset_subset_seed,
        dataset_min_images_per_identity=config.dataset_min_images_per_identity,
    )
    test_loader = None
    if config.resolved_test_dir() is not None:
        _, test_loader = build_eval_loader(
            data_dir=config.resolved_test_dir(),
            class_to_idx=class_to_idx,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

    base_model, device = build_target_model(
        model_name=config.target_model,
        num_classes=len(class_to_idx),
        embedding_dim=config.embedding_dim,
        device_name=config.device,
        backbone_name=config.target_backbone,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        arcface_scale=config.arcface_scale,
        arcface_margin=config.arcface_margin,
        use_partial_fc=config.use_partial_fc,
        partial_fc_negative_sample_rate=config.partial_fc_negative_sample_rate,
        sub_center_count=config.sub_center_count,
        dropout_p=config.dropout_p,
        member_weights=config.joint_pool_member_weights,
    )
    if distributed and config.use_sync_batchnorm and device.type == "cuda":
        base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
    surrogates = build_surrogates(config.surrogate_models, base_model, device)
    model = maybe_wrap_ddp(base_model, device, distributed)

    optimizer = _make_optimizer(config, model)
    scheduler = _make_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_mixed_precision and device.type == "cuda")

    rng = random.Random(config.seed)
    eval_policy = choose_eval_policy(config.all_attackers(), config.eval_attack_name)
    history: list[dict] = []

    if _is_primary():
        logger.info(f"[INFO] Train samples: {len(train_ds)}")
        logger.info(f"[INFO] Val samples:   {len(val_ds)}")
        logger.info(f"[INFO] Classes:       {len(class_to_idx)}")
        logger.info(f"[INFO] Device:        {device}")
        logger.info(f"[INFO] Num workers:   {config.num_workers}")
        if device.type == "mps":
            logger.info(f"[INFO] MPS limiter:   {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'unset')}")
        _log_cuda_runtime_diagnostics(logger, config=config, device=device)
        logger.info(f"[INFO] Backbone:      {config.target_backbone}")
        logger.info(f"[INFO] Embedding dim:  {config.embedding_dim} (placeholder under the 512-D ceiling)")
        logger.info(f"[INFO] Dataset:       {config.dataset_name}")
        logger.info(
            f"[INFO] Data slice:    fraction={config.dataset_fraction} "
            f"min_images_per_id={config.dataset_min_images_per_identity} seed={config.dataset_subset_seed}"
        )
        logger.info(f"[INFO] Aligner:       {config.alignment_detector} / {config.alignment_landmarks}-point / {config.normalized_crop_size}x{config.normalized_crop_size}")
        logger.info(f"[INFO] Surrogates:    {sorted(surrogates)}")
        logger.info(f"[INFO] Optimizer:     {config.optimizer_name} lr={config.learning_rate} momentum={config.momentum}")
        logger.info(f"[INFO] LR warmup:     {config.lr_warmup_epochs}")
        logger.info(f"[INFO] Partial FC:    {config.use_partial_fc} rate={config.partial_fc_negative_sample_rate}")
        logger.info(f"[INFO] Sub-centers:   {config.sub_center_count}")
        logger.info(f"[INFO] Pairing:       {config.pairing_strategy} frac={config.hard_pair_fraction} weight={config.hard_pair_weight}")
        logger.info(f"[INFO] Curriculum:    enabled={config.curriculum_enabled} warmup={config.curriculum_warmup_epochs}")
        logger.info(
            f"[INFO] Eval cadence:  clean={config.clean_eval_every_epochs} "
            f"robust={config.robust_eval_every_epochs} full_robust={config.full_robust_eval_every_epochs} "
            f"verify={config.verification_eval_every_epochs}"
        )
        logger.info(f"[INFO] AMP:           {config.use_mixed_precision}")
        logger.info(f"[INFO] Checkpointing: {config.use_gradient_checkpointing}")
        logger.info(f"[INFO] Distributed:   requested={config.use_distributed} world_size={_world_size()}")
        logger.info(f"[INFO] Primary atk:   {[policy.name for policy in config.enabled_primary_attackers()]}")
        logger.info(f"[INFO] Surrogate atk: {[policy.name for policy in config.enabled_surrogate_attackers()]}")
        if config.enabled_attackers():
            logger.info(f"[INFO] Attack mix:    {config.attack_sampling_strategy}")
            logger.info(
                f"[INFO] Depth mode:    {config.training_depth_mode} "
                f"warmup={config.clean_warmup_epochs} shallow={config.shallow_adv_epochs}"
            )
        if eval_policy is not None:
            logger.info(f"[INFO] Eval attack:   {eval_policy.name}")
        elif config.clean_only or not config.enabled_attackers():
            logger.info("[INFO] Ensemble:      disabled; running clean placeholder training.")

    for epoch in range(1, config.epochs + 1):
        _clear_device_caches(device)
        _reset_cuda_peak_memory(device)
        model.train()
        schedule = _resolve_epoch_schedule(config, epoch)
        epoch_attack_policies = _stage_attack_policies(config, schedule)
        attack_sampler = build_attack_policy_sampler(
            policies=epoch_attack_policies,
            strategy=config.attack_sampling_strategy,
            rng=rng,
        ) if epoch_attack_policies else None
        _apply_epoch_learning_rate(optimizer, scheduler, config, epoch)
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = 0.0
        train_acc = 0.0
        train_consistency = 0.0
        train_cached_hits = 0.0
        train_hard_pair_count = 0.0
        train_hard_sample_count = 0.0
        train_hard_pair_hardness = 0.0
        train_hard_loss = 0.0
        train_member_metrics_totals: dict[str, dict[str, dict[str, float]]] = {}
        batches = 0

        total_batches = max(1, len(train_loader))
        total_train_samples = len(train_ds)
        attack_work = _estimate_epoch_attack_work(schedule, epoch_attack_policies)
        if _is_primary():
            logger.info(
                "[EPOCH-START]"
                f" {epoch}/{config.epochs}"
                f" stage={schedule.get('stage_name', 'unknown')}"
                f" lr={optimizer.param_groups[0]['lr']:.5f}"
                f" train_batches={total_batches}"
                f" train_samples={total_train_samples}"
                f" clean_fraction={float(schedule['clean_fraction']):.2f}"
                f" hard_pair_fraction={float(schedule['hard_pair_fraction']):.2f}"
                f" hard_pair_weight={float(schedule['hard_pair_weight']):.2f}"
                f" approx_attack_iters={attack_work}"
                f" attackers={_policy_names(epoch_attack_policies)}"
            )
            if device.type == "cuda":
                logger.info(f"[CUDA] epoch={epoch}/{config.epochs} start {_format_cuda_memory_stats(device)}")
        samples_seen = 0
        train_progress = _manual_progress_bar(
            total=total_train_samples,
            desc=f"Train {epoch}/{config.epochs} [{schedule.get('stage_name', 'na')}]",
            enabled=_is_primary(),
            leave=True,
            unit="img",
        )
        if train_progress is not None:
            train_progress.set_postfix(
                {
                    "lr": f"{optimizer.param_groups[0]['lr']:.4g}",
                    "cf": f"{float(schedule['clean_fraction']):.2f}",
                    "atk": "on" if bool(schedule["attacks_enabled"]) else "off",
                    "step": f"0/{total_batches}",
                },
                refresh=False,
            )
        for batch_index, (images, labels, _, rel_paths) in enumerate(train_loader, start=1):
            try:
                images = images.to(device, non_blocking=device.type == "cuda")
                labels = labels.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                train_model = _unwrap_model(model)

                with _autocast_context(config, device):
                    clean_logits, clean_embeddings = train_model.forward_logits(images, labels)
                    clean_predict_logits = train_model.predict_logits_from_embeddings(clean_embeddings)
                    clean_member_outputs = _joint_member_outputs(train_model, images, labels)
                    if clean_member_outputs is None:
                        clean_loss_per_sample = F.cross_entropy(clean_logits, labels, reduction="none")
                        clean_loss = clean_loss_per_sample.mean()
                    else:
                        clean_loss_per_sample = _weighted_member_loss_per_sample(train_model, clean_member_outputs)
                        clean_loss = clean_loss_per_sample.mean()
                        for name, item in clean_member_outputs.items():
                            _accumulate_member_epoch_metric(
                                train_member_metrics_totals,
                                name=name,
                                key="clean_loss",
                                value=float(item["loss"].item()),
                            )
                            _accumulate_member_epoch_metric(
                                train_member_metrics_totals,
                                name=name,
                                key="clean_accuracy",
                                value=classification_accuracy(item["predict_logits"].detach(), labels),
                            )
                    hard_pairs = _mine_batch_hard_pairs(
                        config=config,
                        images=images,
                        labels=labels,
                        clean_embeddings=clean_embeddings,
                        surrogates=surrogates,
                        schedule=schedule,
                    )

                    if (not bool(schedule["attacks_enabled"])) or config.clean_only or not config.enabled_attackers():
                        attack_result = None
                        adv_loss = torch.zeros((), device=device)
                        consistency = torch.zeros((), device=device)
                        adv_loss_per_sample = torch.empty(0, device=device)
                        adv_member_outputs = None
                    else:
                        requested_adv = int(round(images.size(0) * (1.0 - schedule["clean_fraction"])))
                        adv_count = min(images.size(0), max(2, requested_adv))
                        adv_images = images[:adv_count]
                        adv_labels = labels[:adv_count]
                        adv_rel_paths = list(rel_paths[:adv_count])

                        if attack_sampler is None:
                            raise RuntimeError("Attack sampler missing while attacks are enabled.")
                        policy = attack_sampler.choose()
                        attack_result = generate_attack_batch(
                            policy=policy,
                            images=adv_images,
                            labels=adv_labels,
                            rel_paths=adv_rel_paths,
                            image_size=config.image_size,
                            target_model=train_model,
                            surrogates=surrogates,
                            device=device,
                        )
                        adv_logits, adv_embeddings = train_model.forward_logits(attack_result.images, adv_labels)
                        adv_member_outputs = _joint_member_outputs(train_model, attack_result.images, adv_labels)
                        if adv_member_outputs is None:
                            adv_loss_per_sample = F.cross_entropy(adv_logits, adv_labels, reduction="none")
                            adv_loss = adv_loss_per_sample.mean()
                            consistency = embedding_consistency_loss(clean_embeddings[:adv_count], adv_embeddings)
                        else:
                            adv_loss_per_sample = _weighted_member_loss_per_sample(train_model, adv_member_outputs)
                            adv_loss = adv_loss_per_sample.mean()
                            consistency, per_member_consistency = _average_member_consistency(
                                train_model,
                                {
                                    name: {
                                        **item,
                                        "embeddings": item["embeddings"][:adv_count],
                                    }
                                    for name, item in clean_member_outputs.items()
                                },
                                adv_member_outputs,
                            )
                            for name, item in adv_member_outputs.items():
                                _accumulate_member_epoch_metric(
                                    train_member_metrics_totals,
                                    name=name,
                                    key="adv_loss",
                                    value=float(item["loss"].item()),
                                )
                                _accumulate_member_epoch_metric(
                                    train_member_metrics_totals,
                                    name=name,
                                    key="adv_accuracy",
                                    value=classification_accuracy(item["predict_logits"].detach(), adv_labels),
                                )
                                _accumulate_member_epoch_metric(
                                    train_member_metrics_totals,
                                    name=name,
                                    key="consistency",
                                    value=per_member_consistency.get(name, 0.0),
                                )

                    hard_loss = torch.zeros((), device=device)
                    if hard_pairs.sample_indices.numel() > 0:
                        hard_loss = hard_loss + clean_loss_per_sample[hard_pairs.sample_indices].mean()
                        if adv_loss_per_sample.numel() > 0:
                            adv_hard_indices = hard_pairs.sample_indices[hard_pairs.sample_indices < adv_loss_per_sample.size(0)]
                            if adv_hard_indices.numel() > 0:
                                hard_loss = hard_loss + adv_loss_per_sample[adv_hard_indices].mean()

                    loss = (
                        config.clean_weight * clean_loss
                        + config.adv_weight * adv_loss
                        + config.consistency_weight * consistency
                        + schedule["hard_pair_weight"] * hard_loss
                    )

                if not torch.isfinite(loss.detach()).all():
                    raise RuntimeError(
                        "Non-finite loss encountered during training "
                        f"(epoch={epoch}, batch={batch_index}, batch_size={config.batch_size}, "
                        f"mixed_precision={config.use_mixed_precision})."
                    )

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            except RuntimeError as exc:
                _raise_batch_runtime_error(
                    exc=exc,
                    config=config,
                    device=device,
                    epoch=epoch,
                    batch_index=batch_index,
                )

            train_loss += float(loss.item())
            train_acc += classification_accuracy(clean_predict_logits.detach(), labels)
            train_consistency += float(consistency.item())
            train_cached_hits += float(0 if attack_result is None else attack_result.cached_hits)
            train_hard_pair_count += float(hard_pairs.pair_count)
            train_hard_sample_count += float(hard_pairs.hard_sample_count)
            train_hard_pair_hardness += float(hard_pairs.mean_hardness)
            train_hard_loss += float(hard_loss.item())
            batches += 1
            samples_seen += int(labels.numel())
            if train_progress is not None:
                postfix = {
                    "loss": f"{train_loss / batches:.4f}",
                    "acc": f"{train_acc / batches:.4f}",
                    "stage": str(schedule.get("stage_name", "na")),
                    "atk": "on" if bool(schedule["attacks_enabled"]) else "off",
                    "step": f"{batch_index}/{total_batches}",
                }
                if attack_result is not None:
                    postfix["policy"] = attack_result.policy_name
                train_progress.set_postfix(postfix, refresh=False)
                train_progress.update(int(labels.numel()))
            if _is_primary() and config.log_every_batches > 0 and (
                batch_index % config.log_every_batches == 0 or batch_index == total_batches
            ):
                logger.info(
                    "[BATCH]"
                    f" epoch={epoch}/{config.epochs}"
                    f" step={batch_index}/{total_batches}"
                    f" seen={samples_seen}/{total_train_samples}"
                    f" loss={train_loss / batches:.4f}"
                    f" acc={train_acc / batches:.4f}"
                    f" stage={schedule.get('stage_name', 'na')}"
                    f" attacks={'on' if bool(schedule['attacks_enabled']) else 'off'}"
                )
        if train_progress is not None:
            train_progress.close()
        _clear_device_caches(device)
        if _is_primary() and device.type == "cuda":
            logger.info(f"[CUDA] epoch={epoch}/{config.epochs} post-train {_format_cuda_memory_stats(device)}")

        if batches > 0:
            scheduler.step()

        run_clean_eval = _should_run_eval(epoch, config.epochs, config.clean_eval_every_epochs)
        run_robust_eval = (
            (not config.clean_only)
            and bool(config.enabled_attackers())
            and _should_run_eval(epoch, config.epochs, config.robust_eval_every_epochs)
        )
        run_full_robust_eval = (
            run_robust_eval
            and config.evaluate_all_attacks
            and _should_run_eval(epoch, config.epochs, config.full_robust_eval_every_epochs)
        )
        run_verification_eval = (
            config.val_pairs_path is not None
            and _should_run_eval(epoch, config.epochs, config.verification_eval_every_epochs)
        )

        clean_metrics = {"skipped": True}
        if run_clean_eval:
            clean_metrics = evaluate_clean(
                _unwrap_model(model),
                val_loader,
                device,
                progress_desc=f"Val clean {epoch}/{config.epochs} [{schedule.get('stage_name', 'na')}]" if _is_primary() else None,
            )

        robust_metrics = {"skipped": True}
        robust_by_attack = None
        if run_full_robust_eval:
            robust_eval = evaluate_robust_all(
                model=_unwrap_model(model),
                loader=val_loader,
                device=device,
                policies=config.all_attackers(),
                surrogates=surrogates,
                image_size=config.image_size,
                progress_prefix=f"Val robust {epoch}/{config.epochs} [{schedule.get('stage_name', 'na')}]" if _is_primary() else None,
            )
            robust_metrics = robust_eval["average"]
            robust_by_attack = robust_eval["by_attack"]
        elif run_robust_eval:
            robust_metrics = evaluate_robust(
                model=_unwrap_model(model),
                loader=val_loader,
                device=device,
                policy=eval_policy,
                surrogates=surrogates,
                image_size=config.image_size,
                progress_desc=f"Val robust {epoch}/{config.epochs} [{schedule.get('stage_name', 'na')}]" if _is_primary() else None,
            )

        verification_metrics = None
        if run_verification_eval:
            verification_metrics = evaluate_verification_pairs(
                model=_unwrap_model(model),
                pairs_path=Path(config.val_pairs_path).resolve(),
                device=device,
            )
        test_clean_metrics = None
        test_verification_metrics = None
        if test_loader is not None and run_clean_eval:
            test_clean_metrics = evaluate_clean(
                _unwrap_model(model),
                test_loader,
                device,
                progress_desc=f"Test clean {epoch}/{config.epochs} [{schedule.get('stage_name', 'na')}]" if _is_primary() else None,
            )
        if config.test_pairs_path is not None:
            test_verification_metrics = evaluate_verification_pairs(
                model=_unwrap_model(model),
                pairs_path=Path(config.test_pairs_path).resolve(),
                device=device,
            )
        _clear_device_caches(device)
        if _is_primary() and device.type == "cuda":
            logger.info(f"[CUDA] epoch={epoch}/{config.epochs} post-eval {_format_cuda_memory_stats(device)}")

        epoch_record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss / max(1, batches),
            "train_accuracy": train_acc / max(1, batches),
            "train_consistency": train_consistency / max(1, batches),
            "train_cached_hits": train_cached_hits / max(1, batches),
            "train_hard_pair_count": train_hard_pair_count / max(1, batches),
            "train_hard_sample_count": train_hard_sample_count / max(1, batches),
            "train_hard_pair_hardness": train_hard_pair_hardness / max(1, batches),
            "train_hard_loss": train_hard_loss / max(1, batches),
            "curriculum_progress": schedule["progress"],
            "training_stage": schedule.get("stage_name", "unknown"),
            "epoch_clean_fraction": schedule["clean_fraction"],
            "epoch_hard_pair_fraction": schedule["hard_pair_fraction"],
            "epoch_hard_pair_weight": schedule["hard_pair_weight"],
            "train_member_metrics": _finalize_member_epoch_metrics(train_member_metrics_totals),
            "val_clean": clean_metrics,
            "val_robust": robust_metrics,
            "val_robust_by_attack": robust_by_attack,
            "val_verification": verification_metrics,
            "test_clean": test_clean_metrics,
            "test_verification": test_verification_metrics,
        }
        history.append(epoch_record)
        if _is_primary():
            _write_history(output_dir, history)
            logger.metric(epoch_record)
            logger.info(
                "[EPOCH]"
                f" {epoch}/{config.epochs}"
                f" stage={epoch_record['training_stage']}"
                f" lr={epoch_record['lr']:.5f}"
                f" train_loss={epoch_record['train_loss']:.4f}"
                f" train_acc={epoch_record['train_accuracy']:.4f}"
                f" hard_pairs={epoch_record['train_hard_pair_count']:.2f}"
                f" val_clean_acc={clean_metrics.get('accuracy', float('nan')):.4f}"
                f" val_robust_acc={robust_metrics.get('accuracy', float('nan')):.4f}"
            )
            if epoch_record["train_member_metrics"]:
                compact = " ".join(
                    (
                        f"{name}:"
                        f"clean={metrics.get('clean_loss', 0.0):.3f}/"
                        f"{metrics.get('clean_accuracy', 0.0):.3f}"
                        f",adv={metrics.get('adv_loss', 0.0):.3f}/"
                        f"{metrics.get('adv_accuracy', 0.0):.3f}"
                    )
                    for name, metrics in epoch_record["train_member_metrics"].items()
                )
                logger.info(f"[MEMBERS] epoch={epoch}/{config.epochs} {compact}")

            if epoch % max(1, config.checkpoint_every) == 0:
                checkpoint_path = _save_checkpoint(output_dir, epoch, model, optimizer, scheduler, scaler, history)
                logger.info(f"[INFO] Saved checkpoint: {checkpoint_path}")

    final_summary = {
        "output_dir": str(output_dir),
        "epochs": config.epochs,
        "history": history,
    }
    if _is_primary():
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(final_summary, handle, indent=2)
            handle.write("\n")
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return final_summary
