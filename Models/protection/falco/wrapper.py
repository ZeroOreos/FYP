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


SCRIPTS = {
    "extract-features": "extract_features.py",
    "create-fake-dataset": "create_fake_dataset.py",
    "pair-nn": "pair_nn.py",
    "invert": "invert.py",
    "anonymize": "anonymize.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin wrapper around FALCO's stage-based scripts.")
    parser.add_argument("stage", choices=sorted(SCRIPTS))
    parser.add_argument("stage_args", nargs=argparse.REMAINDER)
    parser.add_argument("--repo-dir", type=Path, default=PROTECTION_SOURCE_ROOT / "FALCO_upstream")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--work-dir", type=Path, default=PROTECTION_WORKDIR_ROOT / "falco")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    python_bin = os.path.abspath(args.python_bin)
    script = args.repo_dir / SCRIPTS[args.stage]
    stage_args = list(args.stage_args)
    if device == "cpu" and "--cuda" not in stage_args and "--no-cuda" not in stage_args:
        stage_args.insert(0, "--no-cuda")
    if device == "cuda" and "--cuda" not in stage_args and "--no-cuda" not in stage_args:
        stage_args.insert(0, "--cuda")
    run_command(
        [python_bin, str(script.resolve()), *stage_args],
        cwd=args.repo_dir,
        env_updates={
            "PYTHONPATH": str(args.repo_dir),
            "MPLCONFIGDIR": str((args.work_dir / "matplotlib").resolve()),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "FYP_TORCH_DEVICE": device,
        },
    )


if __name__ == "__main__":
    main()
