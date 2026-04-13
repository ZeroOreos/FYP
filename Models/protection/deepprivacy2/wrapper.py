#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(BOOTSTRAP_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_PROJECT_ROOT))

from Models.protection.shared import PROTECTION_ASSET_ROOT, PROTECTION_SOURCE_ROOT, PROTECTION_WORKDIR_ROOT
from Models.protection.shared import collect_images, ensure_dir, resolve_device, run_command, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepPrivacy2 face anonymization over a dataset tree.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-dir", type=Path, default=PROTECTION_SOURCE_ROOT / "DeepPrivacy2_upstream")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--work-dir", type=Path, default=PROTECTION_WORKDIR_ROOT / "deepprivacy2")
    parser.add_argument("--asset-dir", type=Path, default=PROTECTION_ASSET_ROOT / "deepprivacy2")
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--max-res", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = collect_images(args.dataset_dir)
    resolved_device = resolve_device(args.device)
    effective_device = resolved_device
    python_bin = os.path.abspath(args.python_bin)
    output_dir = ensure_dir(args.output_dir)
    work_dir = ensure_dir(args.work_dir)
    asset_dir = ensure_dir(args.asset_dir)
    config_path = args.config_path or (args.repo_dir / "configs" / "anonymizers" / "face.py")

    cmd = [
        python_bin,
        str((args.repo_dir / "anonymize.py").resolve()),
        str(config_path.resolve()),
        "--input_path",
        str(args.dataset_dir.resolve()),
        "--output_path",
        str(output_dir.resolve()),
        "--seed",
        str(args.seed),
    ]
    if args.max_res is not None:
        cmd.extend(["--max-res", str(args.max_res)])

    env_updates = {
        "PYTHONPATH": str(args.repo_dir),
        "BASE_OUTPUT_DIR": str(work_dir / "outputs"),
        "PRETRAINED_CHECKPOINTS_PATH": str(asset_dir / "pretrained_checkpoints"),
        "TORCH_HOME": str(asset_dir / "torch_home"),
        "MPLCONFIGDIR": str(work_dir / "matplotlib"),
        "WANDB_MODE": "offline",
        "WANDB_SILENT": "true",
    }
    if effective_device == "cuda":
        env_updates["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        env_updates["CUDA_VISIBLE_DEVICES"] = ""

    run_command(cmd, cwd=args.repo_dir, env_updates=env_updates)

    records: list[dict[str, str]] = []
    for _, rel_path in items:
        output_path = output_dir / rel_path
        status = "ok" if output_path.exists() else "missing"
        record = {"input": str(rel_path), "status": status}
        if output_path.exists():
            record["output"] = str(output_path)
        records.append(record)

    write_json(
        output_dir / "_deepprivacy2_summary.json",
        {
            "requested_device": args.device,
            "resolved_device": resolved_device,
            "effective_device": effective_device,
            "config_path": str(config_path),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
