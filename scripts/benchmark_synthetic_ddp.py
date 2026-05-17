#!/usr/bin/env python3
# Example: torchrun --standalone --nproc_per_node=1 scripts/benchmark_synthetic_ddp.py --help
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from Training.config import EnsembleTrainingConfig, load_config
from Training.engine import _make_optimizer, _make_scheduler, _resolve_mixed_precision_dtype, maybe_init_distributed, maybe_wrap_ddp
from Training.recognizers import build_target_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic DDP benchmark for the training stack.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    return parser.parse_args()


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_primary() -> bool:
    return _rank() == 0


def _autocast_context(config: EnsembleTrainingConfig, device: torch.device):
    if not config.use_mixed_precision:
        return torch.autocast(device_type="cuda", enabled=False)
    if device.type == "cuda":
        dtype = _resolve_mixed_precision_dtype(config, device)
        if dtype is None:
            return torch.autocast(device_type="cuda", enabled=False)
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    distributed = maybe_init_distributed(config)

    num_classes = 205_849
    model, device = build_target_model(
        model_name=config.target_model,
        num_classes=num_classes,
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
    )
    model = maybe_wrap_ddp(model, device, config, distributed)
    optimizer = _make_optimizer(config, model)
    scheduler = _make_scheduler(config, optimizer)
    del scheduler

    batch_size = int(config.batch_size)
    image_size = int(config.image_size)
    accumulation_steps = max(1, int(config.gradient_accumulation_steps))

    def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.rand(batch_size, 3, image_size, image_size, dtype=torch.float32)
        labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        return images, labels

    optimizer.zero_grad(set_to_none=True)
    timings: list[dict[str, float]] = []
    total_steps = args.warmup_steps + args.steps
    for step in range(total_steps):
        images_cpu, labels_cpu = make_batch()

        t0 = time.perf_counter()
        images = images_cpu.to(device, non_blocking=device.type == "cuda")
        labels = labels_cpu.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        should_step = ((step + 1) % accumulation_steps == 0)
        sync_context = (
            model.no_sync()
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) and not should_step
            else contextlib.nullcontext()
        )
        with sync_context:
            with _autocast_context(config, device):
                out = model(images, labels)
                loss_labels = out.get("loss_labels", labels)
                loss = torch.nn.functional.cross_entropy(out["logits"], loss_labels)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        (loss / accumulation_steps).backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        t4 = time.perf_counter()

        if step >= args.warmup_steps:
            timings.append(
                {
                    "h2d_s": t1 - t0,
                    "forward_s": t2 - t1,
                    "backward_s": t3 - t2,
                    "optimizer_s": t4 - t3,
                }
            )

    summary = {
        "config": str(args.config.resolve()),
        "batch_size": batch_size,
        "world_size": _world_size(),
        "global_batch": batch_size * _world_size(),
        "steps": args.steps,
        "avg_h2d_ms": 1000.0 * sum(item["h2d_s"] for item in timings) / len(timings),
        "avg_forward_ms": 1000.0 * sum(item["forward_s"] for item in timings) / len(timings),
        "avg_backward_ms": 1000.0 * sum(item["backward_s"] for item in timings) / len(timings),
        "avg_optimizer_ms": 1000.0 * sum(item["optimizer_s"] for item in timings) / len(timings),
    }
    total_step_s = sum(sum(item.values()) for item in timings) / len(timings)
    summary["global_img_s"] = (batch_size * _world_size()) / total_step_s
    if _is_primary():
        print(json.dumps(summary, indent=2, sort_keys=True))

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
