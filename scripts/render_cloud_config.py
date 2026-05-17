#!/usr/bin/env python3
# Example: python scripts/render_cloud_config.py --help
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a cloud/server training config from a base JSON config.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="Defaults to $FYP_WEBFACE4M_ROOT.")
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to $FYP_OUTPUT_ROOT or <repo>/TrainingRuns.")
    parser.add_argument("--run-name", default=None, help="Defaults to <base-name>-<UTC timestamp>.")
    parser.add_argument("--cached-attack-root", action="append", default=None)
    parser.add_argument("--enable-distributed", action="store_true")
    parser.add_argument("--disable-distributed", action="store_true")
    parser.add_argument("--enable-sync-batchnorm", action="store_true")
    parser.add_argument("--disable-sync-batchnorm", action="store_true")
    parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument("--disable-torch-compile", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dataset-fraction", type=float, default=None)
    parser.add_argument("--enable-gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def _default_run_name(base_config: Path) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{base_config.stem}-{stamp}"


def main() -> None:
    args = parse_args()
    with args.base_config.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    repo_root = Path(__file__).resolve().parents[1]
    data_root = (args.data_root or Path(os.environ.get("FYP_WEBFACE4M_ROOT", repo_root / "Dataset" / "WebFace4M"))).resolve()
    output_root = (args.output_root or Path(os.environ.get("FYP_OUTPUT_ROOT", repo_root / "TrainingRuns"))).resolve()
    run_name = args.run_name or _default_run_name(args.base_config)

    payload["train_dir"] = str((data_root / "manifests" / "train.jsonl").resolve())
    payload["val_dir"] = str((data_root / "manifests" / "val.jsonl").resolve())
    payload["output_dir"] = str((output_root / run_name).resolve())

    if args.enable_distributed and args.disable_distributed:
        raise SystemExit("Choose only one of --enable-distributed or --disable-distributed.")
    if args.enable_sync_batchnorm and args.disable_sync_batchnorm:
        raise SystemExit("Choose only one of --enable-sync-batchnorm or --disable-sync-batchnorm.")
    if args.enable_torch_compile and args.disable_torch_compile:
        raise SystemExit("Choose only one of --enable-torch-compile or --disable-torch-compile.")
    if args.enable_gradient_checkpointing and args.disable_gradient_checkpointing:
        raise SystemExit("Choose only one of --enable-gradient-checkpointing or --disable-gradient-checkpointing.")
    if args.enable_distributed:
        payload["use_distributed"] = True
    if args.disable_distributed:
        payload["use_distributed"] = False
    if args.enable_sync_batchnorm:
        payload["use_sync_batchnorm"] = True
    if args.disable_sync_batchnorm:
        payload["use_sync_batchnorm"] = False
    if args.enable_torch_compile:
        payload["use_torch_compile"] = True
    if args.disable_torch_compile:
        payload["use_torch_compile"] = False
    if args.num_workers is not None:
        payload["num_workers"] = int(args.num_workers)
    if args.batch_size is not None:
        payload["batch_size"] = int(args.batch_size)
    if args.gradient_accumulation_steps is not None:
        payload["gradient_accumulation_steps"] = max(1, int(args.gradient_accumulation_steps))
    if args.epochs is not None:
        payload["epochs"] = int(args.epochs)
    if args.dataset_fraction is not None:
        payload["dataset_fraction"] = float(args.dataset_fraction)
    if args.enable_gradient_checkpointing:
        payload["use_gradient_checkpointing"] = True
    if args.disable_gradient_checkpointing:
        payload["use_gradient_checkpointing"] = False
    if args.cached_attack_root:
        roots = [str(Path(item).resolve()) for item in args.cached_attack_root]
        for policy in payload.get("surrogate_attackers", []):
            if str(policy.get("kind", "")).strip().lower() == "cached":
                policy["cache_roots"] = roots

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({"output_config": str(args.output_config.resolve()), "run_dir": payload["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
