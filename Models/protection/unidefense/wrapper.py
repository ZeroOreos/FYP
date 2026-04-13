#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(BOOTSTRAP_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_PROJECT_ROOT))

from Models.protection.shared import PROTECTION_SOURCE_ROOT, PROTECTION_WORKDIR_ROOT, resolve_device, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin wrapper around UniDefense test-time entrypoints.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engine", choices=("FE", "OCIM", "UE"), default="FE")
    parser.add_argument("--ds-config", type=Path, default=None)
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--repo-dir", type=Path, default=PROTECTION_SOURCE_ROOT / "UniDefense_upstream")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--work-dir", type=Path, default=PROTECTION_WORKDIR_ROOT / "unidefense")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    python_bin = os.path.abspath(args.python_bin)
    cmd = [
        python_bin,
        str((args.repo_dir / "main.py").resolve()),
        "--config",
        str(args.config.resolve()),
        "--engine",
        args.engine,
        "--test",
        "--offline",
    ]
    if args.ds_config is not None:
        cmd.extend(["--ds_config", str(args.ds_config.resolve())])
    if args.exp_id is not None:
        cmd.extend(["--exp_id", args.exp_id])
    if device == "cuda":
        cmd.extend(["-r", "0"])
    else:
        cmd.extend(["-r", "0"])
    cmd.extend(args.extra_args)
    run_command(
        cmd,
        cwd=args.repo_dir,
        env_updates={
            "PYTHONPATH": str(args.repo_dir),
            "MPLCONFIGDIR": str((args.work_dir / "matplotlib").resolve()),
            "WANDB_MODE": "dryrun",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "FYP_TORCH_DEVICE": device,
            "FYP_ALLOW_RANDOM_INIT": "1" if args.allow_random_init else "0",
            "CUDA_VISIBLE_DEVICES": "0" if device == "cuda" else "",
        },
    )


if __name__ == "__main__":
    main()
