from __future__ import annotations

import contextlib
import gc
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, replace
from datetime import timedelta
from importlib.util import find_spec
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
from Training.dataset import build_dataloaders, build_eval_loader, build_train_loader
from Training.evaluate import choose_eval_policy, evaluate_clean, evaluate_robust, evaluate_robust_all, evaluate_verification_pairs
from Training.losses import classification_accuracy, classification_accuracy_tensor, embedding_consistency_loss
from Training.pairing import HardPairMiningResult, mine_hard_pairs
from Training.recognizers import build_recognizer_ensemble, build_surrogates, build_target_model


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


def _effective_global_batch_size_from_batch(config: EnsembleTrainingConfig, batch_size: int) -> int:
    return max(1, int(batch_size)) * max(1, int(config.gradient_accumulation_steps)) * max(1, _world_size())


def _effective_global_batch_size(config: EnsembleTrainingConfig) -> int:
    return _effective_global_batch_size_from_batch(config, int(config.batch_size))


def _reference_global_batch_size(config: EnsembleTrainingConfig) -> int:
    reference_batch_size = config.reference_batch_size
    if reference_batch_size is None:
        reference_batch_size = int(config.batch_size)
    return max(1, int(reference_batch_size)) * max(1, int(config.gradient_accumulation_steps)) * max(1, _world_size())


def _phase_batch_size(config: EnsembleTrainingConfig, stage_name: str) -> int:
    normalized_stage = stage_name.strip().lower()
    if normalized_stage == "clean_warmup" and config.clean_warmup_batch_size is not None:
        return max(1, int(config.clean_warmup_batch_size))
    if normalized_stage == "shallow_adv" and config.shallow_adv_batch_size is not None:
        return max(1, int(config.shallow_adv_batch_size))
    if normalized_stage == "full_adv" and config.full_adv_batch_size is not None:
        return max(1, int(config.full_adv_batch_size))
    return max(1, int(config.batch_size))


def _learning_rate_scale(config: EnsembleTrainingConfig, *, batch_size: int | None = None) -> float:
    if not config.auto_scale_learning_rate:
        return 1.0
    resolved_batch_size = int(config.batch_size if batch_size is None else batch_size)
    current_batch = float(_effective_global_batch_size_from_batch(config, resolved_batch_size))
    reference_batch = float(_reference_global_batch_size(config))
    ratio = max(1e-12, current_batch / max(1.0, reference_batch))
    mode = config.lr_scale_mode.strip().lower()
    if mode == "linear":
        return ratio
    if mode == "sqrt":
        return ratio ** 0.5
    raise ValueError(f"Unsupported lr_scale_mode '{config.lr_scale_mode}'.")


def _resolved_learning_rate(config: EnsembleTrainingConfig, *, batch_size: int | None = None) -> float:
    return float(config.learning_rate) * _learning_rate_scale(config, batch_size=batch_size)


def _apply_runtime_profile(config: EnsembleTrainingConfig) -> list[str]:
    notes: list[str] = []
    profile = config.normalized_runtime_profile()
    if profile == "benchmark":
        if config.use_torch_compile:
            config.use_torch_compile = False
            notes.append("Runtime profile 'benchmark' keeps torch.compile disabled for clean throughput baselines.")
    elif profile == "paper_full":
        if config.use_torch_compile:
            config.use_torch_compile = False
            notes.append("Runtime profile 'paper_full' keeps torch.compile disabled on the mainline defended run.")
        if config.use_distributed and max(1, int(config.gradient_accumulation_steps)) > 1 and config.ddp_no_sync_accumulation:
            config.ddp_no_sync_accumulation = False
            notes.append("Runtime profile 'paper_full' disables DDP no_sync accumulation for defended full-run stability.")
    return notes


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=max(60, int(config.distributed_timeout_seconds))),
    )
    return True


def maybe_wrap_ddp(
    model: torch.nn.Module,
    device: torch.device,
    config: EnsembleTrainingConfig,
    use_distributed: bool,
):
    if not use_distributed:
        return model
    ddp_kwargs = {
        "broadcast_buffers": bool(config.ddp_broadcast_buffers),
        "gradient_as_bucket_view": bool(config.ddp_gradient_as_bucket_view),
    }
    if config.ddp_bucket_cap_mb is not None:
        ddp_kwargs["bucket_cap_mb"] = int(config.ddp_bucket_cap_mb)
    if "static_graph" in inspect.signature(DistributedDataParallel).parameters:
        ddp_kwargs["static_graph"] = bool(config.ddp_static_graph)
    if device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            **ddp_kwargs,
    )
    return DistributedDataParallel(model, **ddp_kwargs)


def _unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _ddp_logging_data(model: torch.nn.Module) -> dict[str, Any]:
    if not isinstance(model, DistributedDataParallel):
        return {}
    reducer = getattr(model, "_get_ddp_logging_data", None)
    if reducer is None:
        return {}
    try:
        data = reducer()
    except Exception:
        return {}
    return dict(data) if isinstance(data, dict) else {}


@contextlib.contextmanager
def _ddp_no_sync(*modules: torch.nn.Module | None):
    with contextlib.ExitStack() as stack:
        for module in modules:
            if isinstance(module, DistributedDataParallel):
                stack.enter_context(module.no_sync())
        yield


def _trainable_parameters(*modules: torch.nn.Module | None) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    if not params:
        raise RuntimeError("No trainable parameters were provided to the optimizer.")
    return params


def _parameter_count(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _make_optimizer(config: EnsembleTrainingConfig, *modules: torch.nn.Module | None):
    optimizer_kwargs: dict[str, object] = {}
    if config.optimizer_foreach is not None:
        optimizer_kwargs["foreach"] = bool(config.optimizer_foreach)
    if config.optimizer_fused is not None:
        optimizer_kwargs["fused"] = bool(config.optimizer_fused)
    if optimizer_kwargs.get("foreach") and optimizer_kwargs.get("fused"):
        warnings.warn(
            "`optimizer_foreach` and `optimizer_fused` were both enabled; disabling fused to keep the optimizer valid.",
            stacklevel=2,
        )
        optimizer_kwargs["fused"] = False

    params = _trainable_parameters(*modules)
    if config.optimizer_name.lower() == "sgd":
        signature = inspect.signature(torch.optim.SGD)
        filtered_kwargs = {key: value for key, value in optimizer_kwargs.items() if key in signature.parameters}
        return torch.optim.SGD(
            params,
            lr=_resolved_learning_rate(config),
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            **filtered_kwargs,
        )
    if config.optimizer_name.lower() == "adamw":
        signature = inspect.signature(torch.optim.AdamW)
        filtered_kwargs = {key: value for key, value in optimizer_kwargs.items() if key in signature.parameters}
        return torch.optim.AdamW(
            params,
            lr=_resolved_learning_rate(config),
            weight_decay=config.weight_decay,
            **filtered_kwargs,
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
        dtype = _resolve_mixed_precision_dtype(config, device)
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _make_grad_scaler(config: EnsembleTrainingConfig, device: torch.device, amp_dtype: torch.dtype | None):
    enabled = config.use_mixed_precision and device.type == "cuda" and amp_dtype == torch.float16
    amp_module = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_module, "GradScaler", None)
    if grad_scaler_cls is not None:
        return grad_scaler_cls("cuda", enabled=enabled)
    cuda_amp_module = getattr(torch.cuda, "amp", None)
    cuda_grad_scaler_cls = getattr(cuda_amp_module, "GradScaler", None)
    if cuda_grad_scaler_cls is not None:
        return cuda_grad_scaler_cls(enabled=enabled)
    raise RuntimeError("Mixed precision requested, but no GradScaler implementation is available in this PyTorch build.")


def _resolve_mixed_precision_dtype(
    config: EnsembleTrainingConfig,
    device: torch.device,
) -> torch.dtype | None:
    if not config.use_mixed_precision:
        return None
    choice = config.mixed_precision_dtype.strip().lower()
    if device.type == "cuda":
        if choice == "fp16":
            return torch.float16
        if choice == "bf16":
            return torch.bfloat16
        if choice == "auto":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        raise ValueError(f"Unsupported mixed_precision_dtype '{config.mixed_precision_dtype}'.")
    if device.type == "cpu":
        return torch.bfloat16
    return None


_RESUME_SIGNATURE_IGNORED_KEYS = {
    "output_dir",
    "resume_from",
    "checkpoint_every",
    "log_every_batches",
}


def _training_signature(config: EnsembleTrainingConfig, *, class_count: int) -> dict[str, object]:
    payload = asdict(config)
    for key in _RESUME_SIGNATURE_IGNORED_KEYS:
        payload.pop(key, None)
    payload["class_count"] = int(class_count)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "version": 1,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "class_count": int(class_count),
        "payload": payload,
    }


def _check_resume_signature(
    checkpoint: dict[str, Any],
    *,
    expected_signature: dict[str, object],
) -> None:
    stored_signature = checkpoint.get("training_signature")
    if stored_signature is None:
        warnings.warn(
            "Checkpoint has no training signature; resume cannot be treated as paper-clean. "
            "Use a fresh run for final experiments.",
            stacklevel=2,
        )
        return
    stored_hash = str(stored_signature.get("sha256"))
    expected_hash = str(expected_signature.get("sha256"))
    if stored_hash != expected_hash:
        if _truthy(os.environ.get("FYP_ALLOW_SIGNATURE_MISMATCH", "0")):
            warnings.warn(
                "Training signature mismatch ignored because FYP_ALLOW_SIGNATURE_MISMATCH=1.",
                stacklevel=2,
            )
            return
        raise RuntimeError(
            "Checkpoint training signature does not match the current config/class universe. "
            f"checkpoint_sha={stored_hash} current_sha={expected_hash}. "
            "Start a fresh run or set FYP_ALLOW_SIGNATURE_MISMATCH=1 only for debugging."
        )


def _save_checkpoint(
    output_dir: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    scaler,
    history: list[dict],
    *,
    recognizer_ensemble=None,
    training_signature: dict[str, object] | None = None,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "latest.pth"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in checkpoint_dir.glob("*.pth"):
        if stale_path != checkpoint_path:
            stale_path.unlink(missing_ok=True)
    base_model = _unwrap_model(model)
    payload = {
        "epoch": epoch,
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if getattr(scaler, "is_enabled", lambda: False)() else None,
        "history": history,
        "training_signature": training_signature,
    }
    if recognizer_ensemble is not None:
        payload["recognizer_ensemble_state_dict"] = _unwrap_model(recognizer_ensemble).state_dict()
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    model,
    recognizer_ensemble=None,
    optimizer,
    scheduler,
    scaler,
    expected_signature: dict[str, object] | None = None,
) -> tuple[int, list[dict]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if expected_signature is not None:
        _check_resume_signature(checkpoint, expected_signature=expected_signature)
    _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    recognizer_state = checkpoint.get("recognizer_ensemble_state_dict")
    if recognizer_ensemble is not None:
        if recognizer_state is None:
            raise RuntimeError(
                "Checkpoint has no recognizer_ensemble_state_dict. "
                "A trainable recognizer ensemble cannot be resumed from this legacy checkpoint for paper runs."
            )
        _unwrap_model(recognizer_ensemble).load_state_dict(recognizer_state)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state is not None and getattr(scaler, "is_enabled", lambda: False)():
        scaler.load_state_dict(scaler_state)
    history = checkpoint.get("history", [])
    return int(checkpoint["epoch"]), list(history)


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

    if os.environ.get("FYP_CAPTURE_PIP_FREEZE", "").strip().lower() in {"1", "true", "yes", "on"}:
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
    else:
        payload["pip_freeze_skipped"] = True

    with (output_dir / "repro_state.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


class RunLogger:
    def __init__(self, output_dir: Path) -> None:
        self.log_path = output_dir / "train.log"
        self.metrics_path = output_dir / "metrics.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str, *, echo: bool = True) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if echo:
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
    *,
    batch_size: int | None = None,
) -> None:
    scheduled_lrs = scheduler.get_last_lr()
    base_resolved_lr = _resolved_learning_rate(config)
    current_resolved_lr = _resolved_learning_rate(config, batch_size=batch_size)
    if base_resolved_lr > 0:
        scheduled_lrs = [
            float(current_resolved_lr) * (float(lr) / float(base_resolved_lr))
            for lr in scheduled_lrs
        ]
    else:
        scheduled_lrs = [float(current_resolved_lr) for _ in scheduled_lrs]
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


def _member_weights_from_source(weight_source: Any, member_outputs: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
    weight_source = _unwrap_model(weight_source)
    weights = getattr(weight_source, "member_weights", None)
    if not weights:
        uniform = 1.0 / float(max(1, len(member_outputs)))
        base_weights = {name: uniform for name in member_outputs}
    else:
        total = sum(float(weights.get(name, 0.0)) for name in member_outputs)
        if total <= 0:
            uniform = 1.0 / float(max(1, len(member_outputs)))
            base_weights = {name: uniform for name in member_outputs}
        else:
            base_weights = {name: float(weights.get(name, 0.0)) / total for name in member_outputs}
    strategy = str(getattr(weight_source, "weight_strategy", "static")).strip().lower()
    if strategy != "loss_proportional":
        return base_weights
    adaptive_scores = {
        name: max(1e-8, float(item["loss"].detach().item())) * max(1e-8, base_weights.get(name, 0.0))
        for name, item in member_outputs.items()
    }
    adaptive_total = sum(adaptive_scores.values())
    if adaptive_total <= 0:
        return base_weights
    return {name: adaptive_scores[name] / adaptive_total for name in member_outputs}


def _weighted_member_loss_per_sample(
    weight_source: Any,
    member_outputs: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    weights = _member_weights_from_source(weight_source, member_outputs)
    losses = [weights[name] * item["loss_per_sample"] for name, item in member_outputs.items()]
    return torch.stack(losses, dim=0).sum(dim=0)


def _average_member_consistency(
    weight_source: Any,
    clean_member_outputs: dict[str, dict[str, torch.Tensor]],
    adv_member_outputs: dict[str, dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_member: dict[str, torch.Tensor] = {}
    consistency_terms = []
    weights = _member_weights_from_source(weight_source, clean_member_outputs)
    for name, clean_item in clean_member_outputs.items():
        adv_item = adv_member_outputs[name]
        value = embedding_consistency_loss(clean_item["embeddings"], adv_item["embeddings"])
        consistency_terms.append(weights[name] * value)
        per_member[name] = value.detach()
    if not consistency_terms:
        zero = torch.zeros((), device=next(iter(clean_member_outputs.values()))["embeddings"].device)
        return zero, per_member
    return torch.stack(consistency_terms).sum(), per_member


def _accumulate_member_epoch_metric(
    totals: dict[str, dict[str, dict[str, torch.Tensor | float]]],
    *,
    name: str,
    key: str,
    value: torch.Tensor | float,
) -> None:
    member_bucket = totals.setdefault(name, {})
    if isinstance(value, torch.Tensor):
        detached_value = value.detach()
        metric_bucket = member_bucket.setdefault(
            key,
            {
                "sum": torch.zeros((), device=detached_value.device, dtype=detached_value.dtype),
                "count": torch.zeros((), device=detached_value.device, dtype=detached_value.dtype),
            },
        )
        metric_bucket["sum"] = metric_bucket["sum"] + detached_value
        metric_bucket["count"] = metric_bucket["count"] + torch.ones(
            (),
            device=detached_value.device,
            dtype=detached_value.dtype,
        )
        return
    metric_bucket = member_bucket.setdefault(key, {"sum": 0.0, "count": 0.0})
    metric_bucket["sum"] += float(value)
    metric_bucket["count"] += 1.0


def _finalize_member_epoch_metrics(
    totals: dict[str, dict[str, dict[str, torch.Tensor | float]]],
) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for name, metrics in totals.items():
        finalized[name] = {}
        for key, bucket in metrics.items():
            bucket_sum = bucket["sum"]
            bucket_count = bucket["count"]
            if isinstance(bucket_sum, torch.Tensor):
                count = float(max(1.0, float(bucket_count.item())))
                finalized[name][key] = float(bucket_sum.item()) / count
            else:
                count = max(1.0, bucket_count)
                finalized[name][key] = bucket_sum / count
    return finalized


def _merge_metric_groups(
    *groups: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for group in groups:
        for name, metrics in group.items():
            merged[name] = dict(metrics)
    return merged


def _accumulate_weight_profile(
    totals: dict[str, dict[str, dict[str, torch.Tensor | float]]],
    *,
    prefix: str,
    weights: dict[str, float],
) -> None:
    for name, value in weights.items():
        _accumulate_member_epoch_metric(
            totals,
            name=f"{prefix}::{name}",
            key="weight",
            value=float(value),
        )


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


def _should_run_eval(epoch: int, total_epochs: int, every: int, *, offset: int = 0, include_final: bool = True) -> bool:
    if every <= 0:
        return include_final and epoch == total_epochs
    if include_final and epoch == total_epochs:
        return True
    return ((epoch - int(offset)) % every) == 0


def _clear_device_caches(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        return
    if device.type == "mps":
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


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _device_summary(device: torch.device) -> dict[str, object]:
    summary: dict[str, object] = {
        "type": device.type,
        "index": device.index,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        summary.update(
            {
                "name": props.name,
                "total_memory_mb": float(props.total_memory) / (1024.0 * 1024.0),
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return summary


def _best_history_value(history: list[dict[str, object]], getter) -> float | None:
    best: float | None = None
    for item in history:
        value = getter(item)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if best is None or numeric > best:
            best = numeric
    return best


def _lowest_history_value(history: list[dict[str, object]], getter) -> float | None:
    best: float | None = None
    for item in history:
        value = getter(item)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if best is None or numeric < best:
            best = numeric
    return best


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(index)


def _configure_cuda_runtime(config: EnsembleTrainingConfig, device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(config.enable_tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.enable_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(config.float32_matmul_precision)
    torch.backends.cudnn.benchmark = True


def _maybe_enable_channels_last(
    model: torch.nn.Module,
    config: EnsembleTrainingConfig,
    device: torch.device,
) -> torch.nn.Module:
    if device.type != "cuda" or not config.use_channels_last:
        return model
    return model.to(memory_format=torch.channels_last)


def _maybe_channels_last_tensor(
    tensor: torch.Tensor,
    config: EnsembleTrainingConfig,
    device: torch.device,
) -> torch.Tensor:
    if device.type != "cuda" or not config.use_channels_last or tensor.ndim != 4:
        return tensor
    return tensor.contiguous(memory_format=torch.channels_last)


def _maybe_compile_model(
    model: torch.nn.Module,
    config: EnsembleTrainingConfig,
) -> torch.nn.Module:
    if not config.use_torch_compile:
        return model
    if not hasattr(torch, "compile"):
        warnings.warn(
            "torch.compile requested, but this PyTorch build does not expose torch.compile; falling back to eager.",
            stacklevel=2,
        )
        config.use_torch_compile = False
        return model
    if config.torch_compile_backend == "inductor" and find_spec("triton") is None:
        warnings.warn(
            "torch.compile with inductor requested, but Triton is not installed; falling back to eager.",
            stacklevel=2,
        )
        config.use_torch_compile = False
        return model
    return torch.compile(
        model,
        backend=config.torch_compile_backend,
        mode=config.torch_compile_mode,
    )


def _reduce_mean_in_place(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    return tensor


def _reduce_member_metric_totals(
    totals: dict[str, dict[str, dict[str, torch.Tensor | float]]],
    enabled: bool,
) -> None:
    if not enabled or not dist.is_initialized():
        return
    for metrics in totals.values():
        for bucket in metrics.values():
            for key in ("sum", "count"):
                value = bucket[key]
                if isinstance(value, torch.Tensor):
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)


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
    batch_size: int | None = None,
) -> None:
    resolved_batch_size = int(config.batch_size if batch_size is None else batch_size)
    if _is_cuda_oom(exc):
        raise RuntimeError(
            "CUDA OOM during training batch "
            f"(epoch={epoch}, batch={batch_index}, batch_size={resolved_batch_size}, "
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


def run_training(config: EnsembleTrainingConfig, resume_from: Path | None = None) -> dict[str, object]:
    set_seed(config.seed + _rank())
    runtime_profile_notes = _apply_runtime_profile(config)
    distributed = maybe_init_distributed(config)
    device_name = config.device
    if device_name == "auto" and distributed and torch.cuda.is_available():
        device_name = "cuda"

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
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
    )
    active_train_batch_size = int(config.batch_size)
    test_loader = None
    if config.resolved_test_dir() is not None:
        _, test_loader = build_eval_loader(
            data_dir=config.resolved_test_dir(),
            class_to_idx=class_to_idx,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            distributed=distributed,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            pin_memory=config.pin_memory,
        )

    base_model, device = build_target_model(
        model_name=config.target_model,
        num_classes=len(class_to_idx),
        embedding_dim=config.embedding_dim,
        device_name=device_name,
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
    _configure_cuda_runtime(config, device)
    if distributed and config.use_sync_batchnorm and device.type == "cuda":
        base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
    base_model = _maybe_enable_channels_last(base_model, config, device)
    base_model = _maybe_compile_model(base_model, config)
    surrogates = build_surrogates(config.surrogate_models, base_model, device)
    recognizer_ensemble = build_recognizer_ensemble(
        recognizer_specs=config.enabled_recognizers(),
        num_classes=len(class_to_idx),
        embedding_dim=config.embedding_dim,
        device=device,
        arcface_scale=config.arcface_scale,
        arcface_margin=config.arcface_margin,
        use_partial_fc=config.use_partial_fc,
        partial_fc_negative_sample_rate=(
            config.partial_fc_negative_sample_rate
            if config.recognizer_partial_fc_negative_sample_rate is None
            else float(config.recognizer_partial_fc_negative_sample_rate)
        ),
        sub_center_count=config.sub_center_count,
        weight_strategy=config.recognizer_weight_strategy,
    )
    if recognizer_ensemble is not None:
        recognizer_ensemble = maybe_wrap_ddp(recognizer_ensemble, device, config, distributed)
    model = maybe_wrap_ddp(base_model, device, config, distributed)
    ddp_logging_data = _ddp_logging_data(model)
    training_signature = _training_signature(config, class_count=len(class_to_idx))

    optimizer = _make_optimizer(config, model, recognizer_ensemble)
    scheduler = _make_scheduler(config, optimizer)
    amp_dtype = _resolve_mixed_precision_dtype(config, device)
    scaler = _make_grad_scaler(config, device, amp_dtype)

    rng = random.Random(config.seed)
    eval_policy = choose_eval_policy(config.all_eval_attackers(), config.eval_attack_name)
    history: list[dict] = []
    start_epoch = 1

    if resume_from is not None:
        resumed_epoch, history = _load_checkpoint(
            resume_from,
            model=model,
            recognizer_ensemble=recognizer_ensemble,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_signature=training_signature,
        )
        start_epoch = resumed_epoch + 1

    if _is_primary():
        lr_scale = _learning_rate_scale(config)
        resolved_learning_rate = _resolved_learning_rate(config)
        logger.info(f"[INFO] Runtime:       profile={config.runtime_profile}")
        for note in runtime_profile_notes:
            logger.info(f"[INFO] Safety:       {note}")
        logger.info(f"[INFO] Train samples: {len(train_ds)}")
        logger.info(f"[INFO] Val samples:   {len(val_ds)}")
        logger.info(f"[INFO] Classes:       {len(class_to_idx)}")
        logger.info(f"[INFO] Device:        {device}")
        logger.info(f"[INFO] Num workers:   {config.num_workers}")
        logger.info(f"[INFO] Recognizers:   mode={config.recognizers_mode} target={config.target_model}")
        if config.uses_joint_train():
            logger.info("[INFO] Joint train:   ablation/ceiling path enabled")
        logger.info(f"[INFO] Eff batch:     {_effective_global_batch_size(config)}")
        logger.info(
            f"[INFO] Phase batch:    clean={config.clean_warmup_batch_size or config.batch_size} "
            f"shallow={config.shallow_adv_batch_size or config.batch_size} "
            f"full={config.full_adv_batch_size or config.batch_size}"
        )
        logger.info(
            f"[INFO] LR policy:     auto_scale={config.auto_scale_learning_rate} "
            f"mode={config.lr_scale_mode} ref_batch={config.reference_batch_size or config.batch_size} "
            f"scale={lr_scale:.4f}"
        )
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
        logger.info(f"[INFO] Rec ensemble:  {[policy.name for policy in config.enabled_recognizers()]}")
        logger.info(
            f"[INFO] Rec weights:    clean={config.recognizer_clean_weight} adv={config.recognizer_adv_weight} "
            f"strategy={config.recognizer_weight_strategy}"
        )
        logger.info(
            f"[INFO] Trainable params: target={_parameter_count(model):,} "
            f"recognizers={_parameter_count(recognizer_ensemble):,}"
        )
        logger.info(
            f"[INFO] Optimizer:     {config.optimizer_name} "
            f"base_lr={config.learning_rate} resolved_lr={resolved_learning_rate} momentum={config.momentum}"
        )
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
        logger.info(f"[INFO] AMP dtype:     {config.mixed_precision_dtype} -> {amp_dtype}")
        logger.info(f"[INFO] Checkpointing: {config.use_gradient_checkpointing}")
        logger.info(f"[INFO] Channels last: {config.use_channels_last}")
        logger.info(f"[INFO] TF32:          {config.enable_tf32}")
        logger.info(f"[INFO] Compile:       {config.use_torch_compile}")
        logger.info(f"[INFO] Distributed:   requested={config.use_distributed} world_size={_world_size()}")
        logger.info(f"[INFO] DDP timeout:   {config.distributed_timeout_seconds}s")
        if ddp_logging_data:
            logger.info(
                f"[INFO] DDP static ok: {_truthy(ddp_logging_data.get('can_set_static_graph', False))} "
                f"static={config.ddp_static_graph} bucket_view={config.ddp_gradient_as_bucket_view} "
                f"broadcast_buffers={config.ddp_broadcast_buffers} bucket_cap_mb={config.ddp_bucket_cap_mb} "
                f"no_sync_accum={config.ddp_no_sync_accumulation}"
            )
        logger.info(
            f"[INFO] Loader:        pin_memory={config.pin_memory} persistent_workers={config.persistent_workers} "
            f"prefetch_factor={config.prefetch_factor}"
        )
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
        if resume_from is not None:
            logger.info(f"[INFO] Resume:        {resume_from}")
            logger.info(f"[INFO] Resume epoch:  {start_epoch}/{config.epochs}")

    for epoch in range(start_epoch, config.epochs + 1):
        _clear_device_caches(device)
        _reset_cuda_peak_memory(device)
        model.train()
        schedule = _resolve_epoch_schedule(config, epoch)
        epoch_batch_size = _phase_batch_size(config, schedule.get("stage_name", ""))
        recognizer_start_epoch = max(1, int(config.recognizer_start_epoch))
        active_recognizer_ensemble = (
            recognizer_ensemble
            if recognizer_ensemble is not None and epoch >= recognizer_start_epoch
            else None
        )
        if epoch_batch_size != active_train_batch_size:
            train_loader = build_train_loader(
                train_ds,
                batch_size=epoch_batch_size,
                num_workers=config.num_workers,
                distributed=distributed,
                dataset_subset_seed=config.dataset_subset_seed,
                persistent_workers=config.persistent_workers,
                prefetch_factor=config.prefetch_factor,
                pin_memory=config.pin_memory,
            )
            active_train_batch_size = epoch_batch_size
        epoch_attack_policies = _stage_attack_policies(config, schedule)
        attack_sampler = build_attack_policy_sampler(
            policies=epoch_attack_policies,
            strategy=config.attack_sampling_strategy,
            rng=rng,
        ) if epoch_attack_policies else None
        _apply_epoch_learning_rate(optimizer, scheduler, config, epoch, batch_size=epoch_batch_size)
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = torch.zeros((), device=device)
        train_acc = torch.zeros((), device=device)
        train_clean_loss = torch.zeros((), device=device)
        train_adv_loss = torch.zeros((), device=device)
        train_recognizer_clean_loss = torch.zeros((), device=device)
        train_recognizer_adv_loss = torch.zeros((), device=device)
        train_consistency = torch.zeros((), device=device)
        train_cached_hits = torch.zeros((), device=device)
        train_hard_pair_count = torch.zeros((), device=device)
        train_hard_sample_count = torch.zeros((), device=device)
        train_hard_pair_hardness = torch.zeros((), device=device)
        train_hard_loss = torch.zeros((), device=device)
        train_joint_target_member_metrics_totals: dict[str, dict[str, dict[str, torch.Tensor | float]]] = {}
        train_recognizer_metrics_totals: dict[str, dict[str, dict[str, torch.Tensor | float]]] = {}
        train_recognizer_weight_profile_totals: dict[str, dict[str, dict[str, torch.Tensor | float]]] = {}
        batches = 0
        accumulation_steps = max(1, int(config.gradient_accumulation_steps))
        attacked_batches = 0
        attacked_samples = 0
        attack_policy_counts: dict[str, int] = {}

        clean_forward_s_total = 0.0
        hard_pair_s_total = 0.0
        attack_generation_s_total = 0.0
        adv_forward_s_total = 0.0
        backward_s_total = 0.0
        optimizer_s_total = 0.0

        total_batches = max(1, len(train_loader))
        total_train_samples = len(train_ds)
        attack_work = _estimate_epoch_attack_work(schedule, epoch_attack_policies)
        if _is_primary():
            logger.info(
                "[EPOCH-START]"
                f" {epoch}/{config.epochs}"
                f" stage={schedule.get('stage_name', 'unknown')}"
                f" lr={optimizer.param_groups[0]['lr']:.5f}"
                f" batch={epoch_batch_size}"
                f" train_batches={total_batches}"
                f" train_samples={total_train_samples}"
                f" clean_fraction={float(schedule['clean_fraction']):.2f}"
                f" hard_pair_fraction={float(schedule['hard_pair_fraction']):.2f}"
                f" hard_pair_weight={float(schedule['hard_pair_weight']):.2f}"
                f" approx_attack_iters={attack_work}"
                f" attackers={_policy_names(epoch_attack_policies)}"
                f" recognizers={'on' if active_recognizer_ensemble is not None else 'off'}"
            )
            if device.type == "cuda":
                logger.info(f"[CUDA] epoch={epoch}/{config.epochs} start {_format_cuda_memory_stats(device)}")
        samples_seen = 0
        optimizer_steps = 0
        epoch_start_time = time.perf_counter()
        epoch_data_wait_s = 0.0
        epoch_compute_s = 0.0
        log_window_start_time = epoch_start_time
        log_window_samples_seen = 0
        log_window_optimizer_steps = 0
        log_window_data_wait_s = 0.0
        log_window_compute_s = 0.0
        last_batch_end_time = epoch_start_time
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
                    "accum": f"1/{accumulation_steps}",
                    "step": f"0/{total_batches}",
                },
                refresh=False,
            )
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, labels, _, rel_paths) in enumerate(train_loader, start=1):
            batch_start_time = time.perf_counter()
            data_wait_s = max(0.0, batch_start_time - last_batch_end_time)
            try:
                images = images.to(device, non_blocking=device.type == "cuda")
                labels = labels.to(device, non_blocking=device.type == "cuda")
                images = _maybe_channels_last_tensor(images, config, device)
                train_model = _unwrap_model(model)
                should_step = (batch_index % accumulation_steps == 0) or (batch_index == total_batches)
                sync_context = contextlib.nullcontext()
                if (
                    config.ddp_no_sync_accumulation
                    and accumulation_steps > 1
                    and not should_step
                ):
                    sync_context = _ddp_no_sync(model, active_recognizer_ensemble)

                with sync_context:
                    with _autocast_context(config, device):
                        clean_forward_start = time.perf_counter()
                        clean_outputs = model(images, labels)
                        clean_forward_s_total += max(0.0, time.perf_counter() - clean_forward_start)
                        clean_logits = clean_outputs["logits"]
                        clean_loss_labels = clean_outputs.get("loss_labels", labels)
                        clean_embeddings = clean_outputs["embeddings"]
                        clean_predict_logits = clean_outputs["predict_logits"]
                        clean_member_outputs = clean_outputs["member_outputs"]
                        if active_recognizer_ensemble is not None:
                            clean_recognizer_outputs = active_recognizer_ensemble(clean_embeddings, labels)
                            _accumulate_weight_profile(
                                train_recognizer_weight_profile_totals,
                                prefix="clean",
                                weights=_member_weights_from_source(active_recognizer_ensemble, clean_recognizer_outputs),
                            )
                            clean_recognizer_loss_per_sample = _weighted_member_loss_per_sample(
                                active_recognizer_ensemble,
                                clean_recognizer_outputs,
                            )
                            clean_recognizer_loss = clean_recognizer_loss_per_sample.mean()
                        else:
                            clean_recognizer_outputs = None
                            clean_recognizer_loss_per_sample = torch.empty(0, device=device)
                            clean_recognizer_loss = torch.zeros((), device=device)
                        adv_recognizer_loss_per_sample = torch.empty(0, device=device)
                        if clean_member_outputs is None:
                            clean_loss_per_sample = F.cross_entropy(clean_logits, clean_loss_labels, reduction="none")
                            clean_loss = clean_loss_per_sample.mean()
                        else:
                            clean_loss_per_sample = _weighted_member_loss_per_sample(train_model, clean_member_outputs)
                            clean_loss = clean_loss_per_sample.mean()
                            for name, item in clean_member_outputs.items():
                                member_loss_labels = item.get("loss_labels", labels)
                                _accumulate_member_epoch_metric(
                                    train_joint_target_member_metrics_totals,
                                    name=name,
                                    key="clean_loss",
                                    value=item["loss"],
                                )
                                _accumulate_member_epoch_metric(
                                    train_joint_target_member_metrics_totals,
                                    name=name,
                                    key="clean_accuracy",
                                    value=classification_accuracy_tensor(
                                        item["predict_logits"].detach(),
                                        member_loss_labels,
                                    ),
                                )
                        hard_pair_start = time.perf_counter()
                        hard_pairs = _mine_batch_hard_pairs(
                            config=config,
                            images=images,
                            labels=labels,
                            clean_embeddings=clean_embeddings,
                            surrogates=surrogates,
                            schedule=schedule,
                        )
                        hard_pair_s_total += max(0.0, time.perf_counter() - hard_pair_start)

                        if (not bool(schedule["attacks_enabled"])) or config.clean_only or not config.enabled_attackers():
                            attack_result = None
                            adv_loss = torch.zeros((), device=device)
                            adv_recognizer_loss = torch.zeros((), device=device)
                            consistency = torch.zeros((), device=device)
                            adv_loss_per_sample = torch.empty(0, device=device)
                            adv_member_outputs = None
                            adv_recognizer_outputs = None
                        else:
                            requested_adv = int(round(images.size(0) * (1.0 - schedule["clean_fraction"])))
                            adv_count = min(images.size(0), max(2, requested_adv))
                            adv_images = images[:adv_count]
                            adv_labels = labels[:adv_count]
                            adv_rel_paths = list(rel_paths[:adv_count])

                            if attack_sampler is None:
                                raise RuntimeError("Attack sampler missing while attacks are enabled.")
                            policy = attack_sampler.choose()
                            attack_generation_start = time.perf_counter()
                            attack_result = generate_attack_batch(
                                policy=policy,
                                images=adv_images,
                                labels=adv_labels,
                                rel_paths=adv_rel_paths,
                                image_size=config.image_size,
                                target_model=train_model,
                                surrogates=surrogates,
                                recognizer_ensemble=active_recognizer_ensemble,
                                device=device,
                                attack_chunk_size=config.attack_chunk_size,
                            )
                            attack_generation_s_total += max(0.0, time.perf_counter() - attack_generation_start)
                            attacked_batches += 1
                            attacked_samples += int(adv_count)
                            attack_policy_counts[policy.name] = attack_policy_counts.get(policy.name, 0) + 1
                            attack_images = _maybe_channels_last_tensor(attack_result.images, config, device)
                            adv_forward_start = time.perf_counter()
                            adv_outputs = model(attack_images, adv_labels)
                            adv_forward_s_total += max(0.0, time.perf_counter() - adv_forward_start)
                            adv_logits = adv_outputs["logits"]
                            adv_loss_labels = adv_outputs.get("loss_labels", adv_labels)
                            adv_embeddings = adv_outputs["embeddings"]
                            adv_member_outputs = adv_outputs["member_outputs"]
                            if active_recognizer_ensemble is not None:
                                adv_recognizer_outputs = active_recognizer_ensemble(adv_embeddings, adv_labels)
                                _accumulate_weight_profile(
                                    train_recognizer_weight_profile_totals,
                                    prefix="adv",
                                    weights=_member_weights_from_source(active_recognizer_ensemble, adv_recognizer_outputs),
                                )
                                adv_recognizer_loss_per_sample = _weighted_member_loss_per_sample(
                                    active_recognizer_ensemble,
                                    adv_recognizer_outputs,
                                )
                                adv_recognizer_loss = adv_recognizer_loss_per_sample.mean()
                            else:
                                adv_recognizer_outputs = None
                                adv_recognizer_loss_per_sample = torch.empty(0, device=device)
                                adv_recognizer_loss = torch.zeros((), device=device)
                            if adv_member_outputs is None:
                                adv_loss_per_sample = F.cross_entropy(adv_logits, adv_loss_labels, reduction="none")
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
                                    member_loss_labels = item.get("loss_labels", adv_labels)
                                    _accumulate_member_epoch_metric(
                                        train_joint_target_member_metrics_totals,
                                        name=name,
                                        key="adv_loss",
                                        value=item["loss"],
                                    )
                                    _accumulate_member_epoch_metric(
                                        train_joint_target_member_metrics_totals,
                                        name=name,
                                        key="adv_accuracy",
                                        value=classification_accuracy_tensor(
                                            item["predict_logits"].detach(),
                                            member_loss_labels,
                                        ),
                                    )
                                    _accumulate_member_epoch_metric(
                                        train_joint_target_member_metrics_totals,
                                        name=name,
                                        key="consistency",
                                        value=per_member_consistency.get(name, torch.zeros((), device=device)),
                                    )
                        if clean_recognizer_outputs is not None:
                            for name, item in clean_recognizer_outputs.items():
                                _accumulate_member_epoch_metric(
                                    train_recognizer_metrics_totals,
                                    name=name,
                                    key="clean_loss",
                                    value=item["loss"],
                                )
                                _accumulate_member_epoch_metric(
                                    train_recognizer_metrics_totals,
                                    name=name,
                                    key="clean_accuracy",
                                    value=classification_accuracy_tensor(
                                        item["predict_logits"].detach(),
                                        item.get("loss_labels", labels),
                                    ),
                                )
                        if adv_recognizer_outputs is not None:
                            for name, item in adv_recognizer_outputs.items():
                                _accumulate_member_epoch_metric(
                                    train_recognizer_metrics_totals,
                                    name=name,
                                    key="adv_loss",
                                    value=item["loss"],
                                )
                                _accumulate_member_epoch_metric(
                                    train_recognizer_metrics_totals,
                                    name=name,
                                    key="adv_accuracy",
                                    value=classification_accuracy_tensor(
                                        item["predict_logits"].detach(),
                                        item.get("loss_labels", adv_labels),
                                    ),
                                )

                        hard_loss = torch.zeros((), device=device)
                        if hard_pairs.sample_indices.numel() > 0:
                            hard_loss = hard_loss + clean_loss_per_sample[hard_pairs.sample_indices].mean()
                            if adv_loss_per_sample.numel() > 0:
                                adv_hard_indices = hard_pairs.sample_indices[hard_pairs.sample_indices < adv_loss_per_sample.size(0)]
                                if adv_hard_indices.numel() > 0:
                                    hard_loss = hard_loss + adv_loss_per_sample[adv_hard_indices].mean()
                            if clean_recognizer_loss_per_sample.numel() > 0:
                                hard_loss = hard_loss + clean_recognizer_loss_per_sample[hard_pairs.sample_indices].mean()
                            if adv_recognizer_loss_per_sample.numel() > 0:
                                adv_rec_indices = hard_pairs.sample_indices[hard_pairs.sample_indices < adv_recognizer_loss_per_sample.size(0)]
                                if adv_rec_indices.numel() > 0:
                                    hard_loss = hard_loss + adv_recognizer_loss_per_sample[adv_rec_indices].mean()

                        loss = (
                            config.clean_weight * clean_loss
                            + config.adv_weight * adv_loss
                            + config.recognizer_clean_weight * clean_recognizer_loss
                            + config.recognizer_adv_weight * adv_recognizer_loss
                            + config.consistency_weight * consistency
                            + schedule["hard_pair_weight"] * hard_loss
                        )

                    if not torch.isfinite(loss.detach()).all():
                        raise RuntimeError(
                            "Non-finite loss encountered during training "
                            f"(epoch={epoch}, batch={batch_index}, batch_size={config.batch_size}, "
                            f"mixed_precision={config.use_mixed_precision})."
                        )

                    scaled_loss = loss / accumulation_steps
                    if scaler.is_enabled():
                        backward_start = time.perf_counter()
                        scaler.scale(scaled_loss).backward()
                        backward_s_total += max(0.0, time.perf_counter() - backward_start)
                        if should_step:
                            optimizer_start = time.perf_counter()
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            optimizer_s_total += max(0.0, time.perf_counter() - optimizer_start)
                            optimizer_steps += 1
                    else:
                        backward_start = time.perf_counter()
                        scaled_loss.backward()
                        backward_s_total += max(0.0, time.perf_counter() - backward_start)
                        if should_step:
                            optimizer_start = time.perf_counter()
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            optimizer_s_total += max(0.0, time.perf_counter() - optimizer_start)
                            optimizer_steps += 1
            except RuntimeError as exc:
                _raise_batch_runtime_error(
                    exc=exc,
                    config=config,
                    device=device,
                    epoch=epoch,
                    batch_index=batch_index,
                    batch_size=epoch_batch_size,
                )

            train_loss += loss.detach()
            train_acc += classification_accuracy_tensor(clean_predict_logits.detach(), clean_loss_labels)
            train_clean_loss += clean_loss.detach()
            train_adv_loss += adv_loss.detach()
            train_recognizer_clean_loss += clean_recognizer_loss.detach()
            train_recognizer_adv_loss += adv_recognizer_loss.detach()
            train_consistency += consistency.detach()
            train_cached_hits += torch.tensor(
                float(0 if attack_result is None else attack_result.cached_hits),
                device=device,
            )
            train_hard_pair_count += torch.tensor(float(hard_pairs.pair_count), device=device)
            train_hard_sample_count += torch.tensor(float(hard_pairs.hard_sample_count), device=device)
            train_hard_pair_hardness += torch.tensor(float(hard_pairs.mean_hardness), device=device)
            train_hard_loss += hard_loss.detach()
            batches += 1
            samples_seen += int(labels.numel())
            global_samples_seen = min(total_train_samples, samples_seen * max(1, _world_size()))
            batch_end_time = time.perf_counter()
            compute_s = max(0.0, batch_end_time - batch_start_time)
            epoch_data_wait_s += data_wait_s
            epoch_compute_s += compute_s
            log_window_data_wait_s += data_wait_s
            log_window_compute_s += compute_s
            last_batch_end_time = batch_end_time
            if train_progress is not None:
                progress_delta = min(total_train_samples - train_progress.n, global_samples_seen - train_progress.n)
                if progress_delta > 0:
                    train_progress.update(int(progress_delta))
                should_refresh_progress = (
                    config.log_every_batches > 0
                    and (batch_index % config.log_every_batches == 0 or batch_index == total_batches)
                )
                if should_refresh_progress:
                    now = time.perf_counter()
                    elapsed = max(1e-6, now - log_window_start_time)
                    log_window_samples = samples_seen - log_window_samples_seen
                    log_window_steps = optimizer_steps - log_window_optimizer_steps
                    global_img_s = (log_window_samples * max(1, _world_size())) / elapsed
                    optimizer_steps_s = log_window_steps / elapsed
                    logged_batches = max(1, batch_index if batch_index == total_batches else config.log_every_batches)
                    avg_wait_ms = (log_window_data_wait_s * 1000.0) / max(
                        1,
                        min(logged_batches, batch_index),
                    )
                    avg_compute_ms = (log_window_compute_s * 1000.0) / max(
                        1,
                        min(logged_batches, batch_index),
                    )
                    postfix = {
                        "L": f"{float((train_loss / max(1, batches)).item()):.2f}",
                        "Lc": f"{float((train_clean_loss / max(1, batches)).item()):.2f}",
                        "La": f"{float((train_adv_loss / max(1, batches)).item()):.2f}",
                        "Lh": f"{float((train_hard_loss / max(1, batches)).item()):.2f}",
                        "cons": f"{float((train_consistency / max(1, batches)).item()):.3f}",
                        "acc": f"{float((train_acc / max(1, batches)).item()):.4f}",
                        "stage": str(schedule.get("stage_name", "na")),
                        "atk": "on" if bool(schedule["attacks_enabled"]) else "off",
                        "img/s": f"{global_img_s:.1f}",
                        "step_ms": f"{avg_compute_ms:.1f}",
                        "step": f"{batch_index}/{total_batches}",
                    }
                    if attack_result is not None:
                        postfix["policy"] = attack_result.policy_name
                    train_progress.set_postfix(postfix, refresh=False)
            if _is_primary() and config.log_every_batches > 0 and (
                batch_index % config.log_every_batches == 0 or batch_index == total_batches
            ):
                now = time.perf_counter()
                elapsed = max(1e-6, now - log_window_start_time)
                log_window_samples = samples_seen - log_window_samples_seen
                log_window_steps = optimizer_steps - log_window_optimizer_steps
                global_img_s = (log_window_samples * max(1, _world_size())) / elapsed
                optimizer_steps_s = log_window_steps / elapsed
                logged_batches = max(1, batch_index if batch_index == total_batches else config.log_every_batches)
                avg_wait_ms = (log_window_data_wait_s * 1000.0) / max(1, min(logged_batches, batch_index))
                avg_compute_ms = (log_window_compute_s * 1000.0) / max(1, min(logged_batches, batch_index))
                logger.info(
                    "[BATCH]"
                    f" epoch={epoch}/{config.epochs}"
                    f" step={batch_index}/{total_batches}"
                    f" seen={global_samples_seen}/{total_train_samples}"
                    f" loss={float((train_loss / max(1, batches)).item()):.4f}"
                    f" acc={float((train_acc / max(1, batches)).item()):.4f}"
                    f" stage={schedule.get('stage_name', 'na')}"
                    f" attacks={'on' if bool(schedule['attacks_enabled']) else 'off'}"
                    f" accum={((batch_index - 1) % accumulation_steps) + 1}/{accumulation_steps}"
                    f" global_img_s={global_img_s:.1f}"
                    f" opt_steps_s={optimizer_steps_s:.2f}"
                    f" avg_wait_ms={avg_wait_ms:.1f}"
                    f" avg_step_ms={avg_compute_ms:.1f}",
                    echo=False,
                )
                log_window_start_time = now
                log_window_samples_seen = samples_seen
                log_window_optimizer_steps = optimizer_steps
                log_window_data_wait_s = 0.0
                log_window_compute_s = 0.0
        if train_progress is not None:
            train_progress.close()
        _clear_device_caches(device)
        post_train_cuda_stats = _cuda_memory_stats(device)
        if _is_primary() and device.type == "cuda":
            logger.info(f"[CUDA] epoch={epoch}/{config.epochs} post-train {_format_cuda_memory_stats(device)}")

        if batches > 0:
            scheduler.step()

        _reduce_mean_in_place(train_loss, distributed)
        _reduce_mean_in_place(train_acc, distributed)
        _reduce_mean_in_place(train_clean_loss, distributed)
        _reduce_mean_in_place(train_adv_loss, distributed)
        _reduce_mean_in_place(train_recognizer_clean_loss, distributed)
        _reduce_mean_in_place(train_recognizer_adv_loss, distributed)
        _reduce_mean_in_place(train_consistency, distributed)
        _reduce_mean_in_place(train_cached_hits, distributed)
        _reduce_mean_in_place(train_hard_pair_count, distributed)
        _reduce_mean_in_place(train_hard_sample_count, distributed)
        _reduce_mean_in_place(train_hard_pair_hardness, distributed)
        _reduce_mean_in_place(train_hard_loss, distributed)
        _reduce_member_metric_totals(train_joint_target_member_metrics_totals, distributed)
        _reduce_member_metric_totals(train_recognizer_metrics_totals, distributed)
        _reduce_member_metric_totals(train_recognizer_weight_profile_totals, distributed)

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
            and (not run_robust_eval)
            and _should_run_eval(
                epoch,
                config.epochs,
                config.verification_eval_every_epochs,
                offset=int(config.verification_eval_offset),
                include_final=False,
            )
        )

        clean_metrics = {"skipped": True}
        robust_metrics = {"skipped": True}
        robust_by_attack = None
        verification_metrics = None
        test_clean_metrics = None
        test_verification_metrics = None
        if run_clean_eval:
            clean_metrics = evaluate_clean(
                _unwrap_model(model),
                val_loader,
                device,
                recognizer_ensemble=active_recognizer_ensemble,
                progress_desc=(
                    f"Val clean {epoch}/{config.epochs} [{schedule.get('stage_name', 'na')}]"
                    if _is_primary()
                    else None
                ),
            )

        if run_full_robust_eval:
            robust_eval = evaluate_robust_all(
                model=_unwrap_model(model),
                loader=val_loader,
                device=device,
                policies=config.all_eval_attackers(),
                surrogates=surrogates,
                recognizer_ensemble=active_recognizer_ensemble,
                image_size=config.image_size,
                attack_chunk_size=config.attack_chunk_size,
                progress_prefix=(
                    f"Val robust {epoch}/{config.epochs} [{schedule.get('stage_name', 'na')}]"
                    if _is_primary()
                    else None
                ),
                distributed=distributed,
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
                recognizer_ensemble=active_recognizer_ensemble,
                image_size=config.image_size,
                progress_desc=(
                    f"Val robust {epoch}/{config.epochs} [{schedule.get('stage_name', 'na')}]"
                    if _is_primary()
                    else None
                ),
            )

        if run_verification_eval:
            val_pairs_path = Path(config.val_pairs_path).resolve()
            if val_pairs_path.exists():
                rank_verification_metrics = evaluate_verification_pairs(
                    model=_unwrap_model(model),
                    pairs_path=val_pairs_path,
                    device=device,
                    progress_desc=(
                        f"Verification {epoch}/{config.epochs}"
                        if _is_primary()
                        else None
                    ),
                )
                if _is_primary():
                    verification_metrics = rank_verification_metrics
            else:
                if _is_primary():
                    verification_metrics = {
                        "skipped": True,
                        "reason": "missing_pairs_file",
                        "pairs_path": str(val_pairs_path),
                    }
                    logger.info(
                        f"[INFO] Verification skipped: missing pair bundle at {val_pairs_path}"
                    )
        if test_loader is not None and run_clean_eval:
            test_clean_metrics = evaluate_clean(
                _unwrap_model(model),
                test_loader,
                device,
                recognizer_ensemble=active_recognizer_ensemble,
                progress_desc=(
                    f"Test clean {epoch}/{config.epochs} [{schedule.get('stage_name', 'na')}]"
                    if _is_primary()
                    else None
                ),
            )
        if config.test_pairs_path is not None:
            test_pairs_path = Path(config.test_pairs_path).resolve()
            if test_pairs_path.exists():
                rank_test_verification_metrics = evaluate_verification_pairs(
                    model=_unwrap_model(model),
                    pairs_path=test_pairs_path,
                    device=device,
                    progress_desc="Test verification" if _is_primary() else None,
                )
                if _is_primary():
                    test_verification_metrics = rank_test_verification_metrics
            else:
                if _is_primary():
                    test_verification_metrics = {
                        "skipped": True,
                        "reason": "missing_pairs_file",
                        "pairs_path": str(test_pairs_path),
                    }
                    logger.info(
                        f"[INFO] Test verification skipped: missing pair bundle at {test_pairs_path}"
                    )
        _clear_device_caches(device)
        post_eval_cuda_stats = _cuda_memory_stats(device)
        if _is_primary() and device.type == "cuda":
            logger.info(f"[CUDA] epoch={epoch}/{config.epochs} post-eval {_format_cuda_memory_stats(device)}")
        if distributed and dist.is_initialized():
            dist.barrier()

        epoch_wall_s = max(1e-6, time.perf_counter() - epoch_start_time)
        performance = {
            "epoch_wall_s": epoch_wall_s,
            "global_img_s": _safe_divide(float(samples_seen * max(1, _world_size())), epoch_wall_s),
            "optimizer_steps_s": _safe_divide(float(optimizer_steps), epoch_wall_s),
            "samples_seen": float(samples_seen),
            "optimizer_steps": float(optimizer_steps),
            "avg_wait_ms": _safe_divide(epoch_data_wait_s * 1000.0, float(max(1, batches))),
            "avg_step_ms": _safe_divide(epoch_compute_s * 1000.0, float(max(1, batches))),
            "avg_clean_forward_ms": _safe_divide(clean_forward_s_total * 1000.0, float(max(1, batches))),
            "avg_hard_pair_ms": _safe_divide(hard_pair_s_total * 1000.0, float(max(1, batches))),
            "avg_attack_generation_ms": _safe_divide(attack_generation_s_total * 1000.0, float(max(1, attacked_batches))),
            "avg_attack_generation_ms_per_attacked_sample": _safe_divide(
                attack_generation_s_total * 1000.0,
                float(max(1, attacked_samples)),
            ),
            "avg_adv_forward_ms": _safe_divide(adv_forward_s_total * 1000.0, float(max(1, attacked_batches))),
            "avg_backward_ms": _safe_divide(backward_s_total * 1000.0, float(max(1, batches))),
            "avg_optimizer_ms": _safe_divide(optimizer_s_total * 1000.0, float(max(1, optimizer_steps))),
            "train_compute_s": float(epoch_compute_s),
            "train_data_wait_s": float(epoch_data_wait_s),
            "attack_generation_s": float(attack_generation_s_total),
            "clean_forward_s": float(clean_forward_s_total),
            "hard_pair_s": float(hard_pair_s_total),
            "adv_forward_s": float(adv_forward_s_total),
            "backward_s": float(backward_s_total),
            "optimizer_s": float(optimizer_s_total),
            "data_wait_wall_share": _safe_divide(epoch_data_wait_s, epoch_wall_s),
            "attack_generation_compute_share": _safe_divide(attack_generation_s_total, epoch_compute_s),
            "clean_forward_compute_share": _safe_divide(clean_forward_s_total, epoch_compute_s),
            "hard_pair_compute_share": _safe_divide(hard_pair_s_total, epoch_compute_s),
            "adv_forward_compute_share": _safe_divide(adv_forward_s_total, epoch_compute_s),
            "backward_compute_share": _safe_divide(backward_s_total, epoch_compute_s),
            "optimizer_compute_share": _safe_divide(optimizer_s_total, epoch_compute_s),
            "attacked_batches": float(attacked_batches),
            "attacked_samples": float(attacked_samples),
            "attack_share_of_batches": _safe_divide(float(attacked_batches), float(max(1, batches))),
            "attack_policy_counts": {name: int(count) for name, count in sorted(attack_policy_counts.items())},
            "attack_policy_fractions": {
                name: float(count) / float(max(1, attacked_batches))
                for name, count in sorted(attack_policy_counts.items())
            },
        }
        memory = {
            "post_train_cuda": post_train_cuda_stats,
            "post_eval_cuda": post_eval_cuda_stats,
            "cuda_peak_mb": post_eval_cuda_stats.get("max_allocated_mb", post_train_cuda_stats.get("max_allocated_mb", 0.0)),
            "cuda_reserved_peak_mb": post_eval_cuda_stats.get("max_reserved_mb", post_train_cuda_stats.get("max_reserved_mb", 0.0)),
        }

        train_joint_target_member_metrics = _finalize_member_epoch_metrics(train_joint_target_member_metrics_totals)
        train_recognizer_metrics = _finalize_member_epoch_metrics(train_recognizer_metrics_totals)
        train_recognizer_weight_profile = _finalize_member_epoch_metrics(train_recognizer_weight_profile_totals)
        avg_train_loss = float((train_loss / max(1, batches)).item())
        avg_clean_loss = float((train_clean_loss / max(1, batches)).item())
        avg_adv_loss = float((train_adv_loss / max(1, batches)).item())
        avg_recognizer_clean_loss = float((train_recognizer_clean_loss / max(1, batches)).item())
        avg_recognizer_adv_loss = float((train_recognizer_adv_loss / max(1, batches)).item())
        avg_consistency = float((train_consistency / max(1, batches)).item())
        avg_hard_loss = float((train_hard_loss / max(1, batches)).item())
        weighted_loss_components = {
            "clean": float(config.clean_weight) * avg_clean_loss,
            "adv": float(config.adv_weight) * avg_adv_loss,
            "recognizer_clean": float(config.recognizer_clean_weight) * avg_recognizer_clean_loss,
            "recognizer_adv": float(config.recognizer_adv_weight) * avg_recognizer_adv_loss,
            "consistency": float(config.consistency_weight) * avg_consistency,
            "hard_pair": float(schedule["hard_pair_weight"]) * avg_hard_loss,
        }
        weighted_loss_components["sum"] = float(sum(weighted_loss_components.values()))
        weighted_loss_components["residual"] = avg_train_loss - weighted_loss_components["sum"]
        epoch_record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "base_learning_rate": float(config.learning_rate),
            "resolved_learning_rate": float(_resolved_learning_rate(config, batch_size=epoch_batch_size)),
            "learning_rate_scale": float(_learning_rate_scale(config, batch_size=epoch_batch_size)),
            "reference_batch_size": int(config.reference_batch_size or config.batch_size),
            "epoch_batch_size": int(epoch_batch_size),
            "effective_global_batch_size": int(_effective_global_batch_size_from_batch(config, epoch_batch_size)),
            "train_loss": avg_train_loss,
            "train_accuracy": float((train_acc / max(1, batches)).item()),
            "train_clean_loss": avg_clean_loss,
            "train_adv_loss": avg_adv_loss,
            "train_recognizer_clean_loss": avg_recognizer_clean_loss,
            "train_recognizer_adv_loss": avg_recognizer_adv_loss,
            "train_consistency": avg_consistency,
            "train_cached_hits": float((train_cached_hits / max(1, batches)).item()),
            "train_hard_pair_count": float((train_hard_pair_count / max(1, batches)).item()),
            "train_hard_sample_count": float((train_hard_sample_count / max(1, batches)).item()),
            "train_hard_pair_hardness": float((train_hard_pair_hardness / max(1, batches)).item()),
            "train_hard_loss": avg_hard_loss,
            "train_weighted_loss_components": weighted_loss_components,
            "curriculum_progress": schedule["progress"],
            "training_stage": schedule.get("stage_name", "unknown"),
            "epoch_clean_fraction": schedule["clean_fraction"],
            "epoch_hard_pair_fraction": schedule["hard_pair_fraction"],
            "epoch_hard_pair_weight": schedule["hard_pair_weight"],
            "recognizer_ensemble_trainable": bool(_parameter_count(recognizer_ensemble) > 0),
            "recognizer_ensemble_param_count": int(_parameter_count(recognizer_ensemble)),
            "train_joint_target_member_metrics": train_joint_target_member_metrics,
            "train_recognizer_metrics": train_recognizer_metrics,
            "train_recognizer_weight_profile": train_recognizer_weight_profile,
            "train_member_metrics": _merge_metric_groups(
                train_joint_target_member_metrics,
                {f"rec::{name}": metrics for name, metrics in train_recognizer_metrics.items()},
                {f"recw::{name}": metrics for name, metrics in train_recognizer_weight_profile.items()},
            ),
            "val_clean": clean_metrics,
            "val_robust": robust_metrics,
            "val_robust_by_attack": robust_by_attack,
            "val_verification": verification_metrics,
            "test_clean": test_clean_metrics,
            "test_verification": test_verification_metrics,
            "performance": performance,
            "memory": memory,
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
                f" L={epoch_record['train_loss']:.3f}"
                f" Lc={epoch_record['train_clean_loss']:.3f}"
                f" La={epoch_record['train_adv_loss']:.3f}"
                f" Lrc={epoch_record['train_recognizer_clean_loss']:.3f}"
                f" Lra={epoch_record['train_recognizer_adv_loss']:.3f}"
                f" Lh={epoch_record['train_hard_loss']:.3f}"
                f" cons={epoch_record['train_consistency']:.4f}"
                f" acc={epoch_record['train_accuracy']:.4f}"
                f" vclean={clean_metrics.get('loss', float('nan')):.3f}/{clean_metrics.get('accuracy', float('nan')):.4f}"
                f" vrob={robust_metrics.get('loss', float('nan')):.3f}/{robust_metrics.get('accuracy', float('nan')):.4f}"
                f" img_s={performance.get('global_img_s', 0.0):.0f}"
                f" atk_ms={performance.get('avg_attack_generation_ms', 0.0):.1f}"
                f" mem={memory.get('cuda_peak_mb', 0.0):.0f}MB"
            )
            weighted_components = epoch_record["train_weighted_loss_components"]
            logger.info(
                "[LOSS-COMP]"
                f" epoch={epoch}/{config.epochs}"
                f" clean={weighted_components.get('clean', 0.0):.3f}"
                f" adv={weighted_components.get('adv', 0.0):.3f}"
                f" rec_clean={weighted_components.get('recognizer_clean', 0.0):.3f}"
                f" rec_adv={weighted_components.get('recognizer_adv', 0.0):.3f}"
                f" cons={weighted_components.get('consistency', 0.0):.3f}"
                f" hard={weighted_components.get('hard_pair', 0.0):.3f}"
                f" sum={weighted_components.get('sum', 0.0):.3f}"
                f" residual={weighted_components.get('residual', 0.0):.5f}"
            )
            if epoch_record["train_joint_target_member_metrics"]:
                compact = " ".join(
                    (
                        f"{name}:"
                        f"clean={metrics.get('clean_loss', 0.0):.3f}/"
                        f"{metrics.get('clean_accuracy', 0.0):.3f}"
                        f",adv={metrics.get('adv_loss', 0.0):.3f}/"
                        f"{metrics.get('adv_accuracy', 0.0):.3f}"
                    )
                    for name, metrics in epoch_record["train_joint_target_member_metrics"].items()
                )
                logger.info(f"[TARGET-MEMBERS] epoch={epoch}/{config.epochs} {compact}")
            if epoch_record["train_recognizer_metrics"]:
                compact = " ".join(
                    (
                        f"{name}:"
                        f"clean={metrics.get('clean_loss', 0.0):.3f}/"
                        f"{metrics.get('clean_accuracy', 0.0):.3f}"
                        f",adv={metrics.get('adv_loss', 0.0):.3f}/"
                        f"{metrics.get('adv_accuracy', 0.0):.3f}"
                    )
                    for name, metrics in epoch_record["train_recognizer_metrics"].items()
                )
                logger.info(f"[RECOGNIZERS] epoch={epoch}/{config.epochs} {compact}")
            if epoch_record["train_recognizer_weight_profile"]:
                compact = " ".join(
                    f"{name}:{metrics.get('weight', 0.0):.3f}"
                    for name, metrics in epoch_record["train_recognizer_weight_profile"].items()
                )
                logger.info(f"[REC-WEIGHTS] epoch={epoch}/{config.epochs} {compact}")

            if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
                checkpoint_path = _save_checkpoint(
                    output_dir,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    history,
                    recognizer_ensemble=recognizer_ensemble,
                    training_signature=training_signature,
                )
                logger.info(f"[INFO] Saved checkpoint: {checkpoint_path}")

    if history and config.val_pairs_path is not None:
        latest_record = history[-1]
        latest_verification = latest_record.get("val_verification")
        needs_final_verification = not (
            isinstance(latest_verification, dict)
            and not latest_verification.get("skipped")
        )
        if needs_final_verification:
            val_pairs_path = Path(config.val_pairs_path).resolve()
            final_epoch = int(latest_record.get("epoch", config.epochs))
            if val_pairs_path.exists():
                if _is_primary():
                    logger.info(
                        "[INFO] Running final verification backfill "
                        f"for epoch {final_epoch}/{config.epochs}: {val_pairs_path}"
                    )
                rank_final_verification_metrics = evaluate_verification_pairs(
                    model=_unwrap_model(model),
                    pairs_path=val_pairs_path,
                    device=device,
                    progress_desc=(
                        f"Final verification {final_epoch}/{config.epochs}"
                        if _is_primary()
                        else None
                    ),
                )
                if _is_primary():
                    latest_record["val_verification"] = rank_final_verification_metrics
                    latest_record["val_verification_backfilled"] = True
                    latest_record["val_verification_backfill_reason"] = (
                        "final_epoch_overlapped_robust_eval_or_cadence_skip"
                    )
                    _write_history(output_dir, history)
                    with (output_dir / "posthoc-verification.json").open("w", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "epoch": final_epoch,
                                "reason": latest_record["val_verification_backfill_reason"],
                                "result": rank_final_verification_metrics,
                                "stages": {
                                    "verification": {
                                        "ok": True,
                                        "backfilled": True,
                                        "result": rank_final_verification_metrics,
                                    }
                                },
                            },
                            handle,
                            indent=2,
                            sort_keys=True,
                        )
                        handle.write("\n")
                    checkpoint_path = _save_checkpoint(
                        output_dir,
                        final_epoch,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        history,
                        recognizer_ensemble=recognizer_ensemble,
                        training_signature=training_signature,
                    )
                    logger.info(
                        "[INFO] Final verification backfill complete; "
                        f"updated history, posthoc-verification.json, and {checkpoint_path}"
                    )
            elif _is_primary():
                latest_record["val_verification"] = {
                    "skipped": True,
                    "reason": "missing_pairs_file",
                    "pairs_path": str(val_pairs_path),
                }
                latest_record["val_verification_backfilled"] = False
                _write_history(output_dir, history)
                logger.info(
                    f"[INFO] Final verification backfill skipped: missing pair bundle at {val_pairs_path}"
                )
            if distributed and dist.is_initialized():
                dist.barrier()

    if history:
        latest_record = history[-1]
        final_eval_attackers = config.enabled_eval_attackers()
        expected_robust_attack_names = {policy.name for policy in final_eval_attackers if policy.enabled}
        latest_by_attack = latest_record.get("val_robust_by_attack")
        completed_robust_attack_names = (
            set(latest_by_attack)
            if isinstance(latest_by_attack, dict)
            else set()
        )
        latest_robust = latest_record.get("val_robust")
        needs_final_robust = not (
            isinstance(latest_robust, dict)
            and not latest_robust.get("skipped")
        )
        if expected_robust_attack_names and not expected_robust_attack_names.issubset(completed_robust_attack_names):
            needs_final_robust = True
        if needs_final_robust and final_eval_attackers:
            final_epoch = int(latest_record.get("epoch", config.epochs))
            if _is_primary():
                missing_robust = sorted(expected_robust_attack_names - completed_robust_attack_names)
                logger.info(
                    "[INFO] Running final robust backfill "
                    f"for epoch {final_epoch}/{config.epochs}: {_policy_names(final_eval_attackers)}"
                    + (f" missing={missing_robust}" if missing_robust else "")
                )
            if config.evaluate_all_attacks or len(final_eval_attackers) > 1:
                final_robust_eval = evaluate_robust_all(
                    model=_unwrap_model(model),
                    loader=val_loader,
                    device=device,
                    policies=final_eval_attackers,
                    surrogates=surrogates,
                    recognizer_ensemble=active_recognizer_ensemble,
                    image_size=config.image_size,
                    attack_chunk_size=config.attack_chunk_size,
                    progress_prefix=(
                        f"Final robust {final_epoch}/{config.epochs}"
                        if _is_primary()
                        else None
                    ),
                    distributed=distributed,
                )
                final_robust_metrics = final_robust_eval["average"]
                final_robust_by_attack = final_robust_eval["by_attack"]
            else:
                final_policy = eval_policy or final_eval_attackers[0]
                final_robust_metrics = evaluate_robust(
                    model=_unwrap_model(model),
                    loader=val_loader,
                    device=device,
                    policy=final_policy,
                    surrogates=surrogates,
                    recognizer_ensemble=active_recognizer_ensemble,
                    image_size=config.image_size,
                    attack_chunk_size=config.attack_chunk_size,
                    progress_desc=(
                        f"Final robust {final_epoch}/{config.epochs}"
                        if _is_primary()
                        else None
                    ),
                )
                final_robust_by_attack = {final_policy.name: final_robust_metrics}
            if _is_primary():
                latest_record["val_robust"] = final_robust_metrics
                latest_record["val_robust_by_attack"] = final_robust_by_attack
                latest_record["val_robust_backfilled"] = True
                latest_record["val_robust_backfill_reason"] = (
                    "final_epoch_eval_attackers_absent_or_incomplete"
                )
                _write_history(output_dir, history)
                with (output_dir / "posthoc-robust.json").open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "epoch": final_epoch,
                            "reason": latest_record["val_robust_backfill_reason"],
                            "result": final_robust_metrics,
                            "by_attack": final_robust_by_attack,
                            "stages": {
                                "robust": {
                                    "ok": True,
                                    "backfilled": True,
                                    "result": {
                                        "average": final_robust_metrics,
                                        "by_attack": final_robust_by_attack,
                                    },
                                }
                            },
                        },
                        handle,
                        indent=2,
                        sort_keys=True,
                    )
                    handle.write("\n")
                checkpoint_path = _save_checkpoint(
                    output_dir,
                    final_epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    history,
                    recognizer_ensemble=recognizer_ensemble,
                    training_signature=training_signature,
                )
                logger.info(
                    "[INFO] Final robust backfill complete; "
                    f"updated history, posthoc-robust.json, and {checkpoint_path}"
                )
            if distributed and dist.is_initialized():
                dist.barrier()

    final_summary = {
        "output_dir": str(output_dir),
        "epochs": config.epochs,
        "history": history,
        "metrics_schema_version": 4,
        "runtime_profile": config.normalized_runtime_profile(),
        "device": _device_summary(device),
        "training_signature": training_signature,
        "recognizer_ensemble_trainable": bool(_parameter_count(recognizer_ensemble) > 0),
        "recognizer_ensemble_param_count": int(_parameter_count(recognizer_ensemble)),
        "distributed": {
            "enabled": bool(distributed),
            "world_size": int(_world_size()),
            "backend": config.distributed_backend,
            "ddp_no_sync_accumulation": bool(config.ddp_no_sync_accumulation),
            "ddp_static_graph": bool(config.ddp_static_graph),
        },
        "runtime_flags": {
            "batch_size": int(config.batch_size),
            "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
            "effective_global_batch_size": int(_effective_global_batch_size(config)),
            "mixed_precision": bool(config.use_mixed_precision),
            "mixed_precision_dtype": config.mixed_precision_dtype,
            "channels_last": bool(config.use_channels_last),
            "sync_batchnorm": bool(config.use_sync_batchnorm),
            "gradient_checkpointing": bool(config.use_gradient_checkpointing),
            "torch_compile": bool(config.use_torch_compile),
            "num_workers": int(config.num_workers),
        },
        "latest_epoch": history[-1] if history else None,
        "best": {
            "val_clean_accuracy": _best_history_value(history, lambda item: (item.get("val_clean") or {}).get("accuracy")),
            "val_robust_accuracy": _best_history_value(history, lambda item: (item.get("val_robust") or {}).get("accuracy")),
            "val_verification_auc": _best_history_value(history, lambda item: (item.get("val_verification") or {}).get("roc_auc")),
            "val_verification_eer": _lowest_history_value(history, lambda item: (item.get("val_verification") or {}).get("eer")),
            "val_verification_tar_far_1e_4": _best_history_value(history, lambda item: (item.get("val_verification") or {}).get("tar@far=0.0001")),
            "val_verification_tar_far_1e_5": _best_history_value(history, lambda item: (item.get("val_verification") or {}).get("tar@far=1e-05")),
            "val_robust_attack_success_rate": _lowest_history_value(history, lambda item: (item.get("val_robust") or {}).get("attack_success_rate")),
            "val_clean_ensemble_gain_vs_best_single": _best_history_value(history, lambda item: ((item.get("val_clean") or {}).get("ensemble") or {}).get("gain_vs_best_single")),
            "val_robust_ensemble_gain_vs_best_single": _best_history_value(history, lambda item: ((item.get("val_robust") or {}).get("ensemble") or {}).get("gain_vs_best_single")),
            "val_clean_recognizer_ensemble_gain_vs_best_single": _best_history_value(history, lambda item: ((item.get("val_clean") or {}).get("recognizer_ensemble") or {}).get("gain_vs_best_single")),
            "val_robust_recognizer_ensemble_gain_vs_best_single": _best_history_value(history, lambda item: ((item.get("val_robust") or {}).get("recognizer_ensemble") or {}).get("gain_vs_best_single")),
            "throughput_global_img_s": _best_history_value(history, lambda item: (item.get("performance") or {}).get("global_img_s")),
        },
    }
    if _is_primary():
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(final_summary, handle, indent=2)
            handle.write("\n")
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return final_summary
