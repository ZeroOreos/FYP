#!/usr/bin/env python3
# Example: torchrun --standalone --nproc_per_node=1 scripts/probe_verification.py --help
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Training.config import load_config
from Training.dataset import build_configured_class_to_idx
from Training.engine import maybe_init_distributed, maybe_wrap_ddp
from Training.evaluate import evaluate_verification_pairs
from Training.recognizers import build_target_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe verification-only throughput for a run/config.")
    parser.add_argument("--config", type=Path, required=True, help="Training config to build the model.")
    parser.add_argument("--pairs", type=Path, default=None, help="Pair bundle. Defaults to config.val_pairs_path.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to load. Defaults to config.resume_from or output_dir/checkpoints/latest.pth.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path for the primary rank timing report.")
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Load only checkpoint tensors whose shapes match the current model. Useful for embedding-only verification probes after class-count drift.",
    )
    return parser.parse_args()


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_primary() -> bool:
    return _rank() == 0


def _distributed_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def _infer_checkpoint(config, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    resume_from = config.resolved_resume_from()
    if resume_from is not None:
        return resume_from.expanduser().resolve()
    candidate = config.resolved_output_dir() / "checkpoints" / "latest.pth"
    return candidate if candidate.exists() else None


def _pair_shape(pairs_path: Path) -> dict[str, object]:
    pair_data = dict(np.load(pairs_path, allow_pickle=True))
    img1_paths = [str(item) for item in pair_data["img1_paths"].tolist()]
    img2_paths = [str(item) for item in pair_data["img2_paths"].tolist()]
    unique_refs = set(img1_paths + img2_paths)
    shard_refs = sum(1 for item in unique_refs if "::" in item)
    image_refs = len(unique_refs) - shard_refs
    return {
        "pairs": len(img1_paths),
        "unique_refs": len(unique_refs),
        "image_refs": image_refs,
        "shard_refs": shard_refs,
        "reference_mode": "mixed" if image_refs and shard_refs else ("image_path" if image_refs else "shard"),
    }


def _load_checkpoint_for_probe(model, checkpoint_path: Path, *, allow_partial: bool) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    target_model = getattr(model, "module", model)
    state_dict = checkpoint["model_state_dict"]
    if not allow_partial:
        target_model.load_state_dict(state_dict)
        return {
            "loaded_tensors": len(state_dict),
            "skipped_tensors": 0,
            "partial_load": False,
        }

    current_state = target_model.state_dict()
    compatible = {
        name: tensor
        for name, tensor in state_dict.items()
        if name in current_state and tuple(current_state[name].shape) == tuple(tensor.shape)
    }
    skipped = sorted(name for name in state_dict if name not in compatible)
    current_state.update(compatible)
    target_model.load_state_dict(current_state)
    return {
        "loaded_tensors": len(compatible),
        "skipped_tensors": len(skipped),
        "partial_load": True,
        "skipped_tensor_names": skipped[:20],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config.expanduser().resolve())
    distributed = maybe_init_distributed(config)
    device_name = config.device
    if device_name == "auto" and distributed and torch.cuda.is_available():
        device_name = "cuda"

    pairs_path = (args.pairs.expanduser().resolve() if args.pairs is not None else Path(config.val_pairs_path or "")).resolve()
    if not pairs_path.exists():
        raise SystemExit(f"Pair bundle not found: {pairs_path}")

    class_to_idx = build_configured_class_to_idx(
        config.resolved_train_dir(),
        config.resolved_val_dir(),
        dataset_fraction=config.dataset_fraction,
        dataset_subset_seed=config.dataset_subset_seed,
        dataset_min_images_per_identity=config.dataset_min_images_per_identity,
    )
    model, device = build_target_model(
        model_name=config.target_model,
        num_classes=len(class_to_idx),
        embedding_dim=config.embedding_dim,
        device_name=device_name,
        backbone_name=config.target_backbone,
        use_gradient_checkpointing=False,
        arcface_scale=config.arcface_scale,
        arcface_margin=config.arcface_margin,
        use_partial_fc=config.use_partial_fc,
        partial_fc_negative_sample_rate=config.partial_fc_negative_sample_rate,
        sub_center_count=config.sub_center_count,
        dropout_p=config.dropout_p,
        member_weights=config.joint_pool_member_weights,
    )
    model = maybe_wrap_ddp(model, device, config, distributed)

    checkpoint_path = _infer_checkpoint(config, args.checkpoint)
    checkpoint_load_info: dict[str, object] | None = None
    if checkpoint_path is not None:
        checkpoint_load_info = _load_checkpoint_for_probe(
            model,
            checkpoint_path,
            allow_partial=bool(args.allow_partial_checkpoint),
        )
        if _is_primary():
            print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
            print(f"[INFO] Checkpoint load: {checkpoint_load_info}")
    elif _is_primary():
        print("[WARN] No checkpoint found; probing randomly initialized model throughput only.")

    pair_info = _pair_shape(pairs_path)
    if _is_primary():
        print(f"[INFO] Pair bundle: {pairs_path}")
        print(f"[INFO] Pair info: {pair_info}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    metrics = evaluate_verification_pairs(
        model=model,
        pairs_path=pairs_path,
        device=device,
        progress_desc="Verify probe",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - start

    elapsed_tensor = torch.as_tensor([elapsed_s], dtype=torch.float64, device=device)
    if _distributed_enabled():
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    wall_s = float(elapsed_tensor.item())
    report = {
        "pairs_path": str(pairs_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_load": checkpoint_load_info,
        "distributed": bool(distributed),
        "world_size": int(dist.get_world_size()) if _distributed_enabled() else 1,
        "device": str(device),
        "elapsed_s": wall_s,
        "pairs_per_s": float(pair_info["pairs"]) / max(1e-9, wall_s),
        "unique_refs_per_s": float(pair_info["unique_refs"]) / max(1e-9, wall_s),
        **pair_info,
        "metrics": metrics,
    }
    if _is_primary():
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.output_json is not None:
            args.output_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            args.output_json.expanduser().resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
