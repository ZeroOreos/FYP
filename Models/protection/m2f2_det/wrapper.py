#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(BOOTSTRAP_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_PROJECT_ROOT))

from Models.protection.shared import PROTECTION_ASSET_ROOT, PROTECTION_SOURCE_ROOT, PROTECTION_WORKDIR_ROOT, resolve_device, run_command


SCRIPTS = {
    "stage1-det": ("python", "stage_1_detection_inference.py"),
    "stage3-det": ("shell", "stage_3_inference_det.sh"),
    "stage3-exp": ("shell", "stage_3_inference_exp.sh"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin wrapper around M2F2-Det inference stages.")
    parser.add_argument("stage", choices=sorted(SCRIPTS))
    parser.add_argument("stage_args", nargs=argparse.REMAINDER)
    parser.add_argument("--repo-dir", type=Path, default=PROTECTION_SOURCE_ROOT / "M2F2_Det_upstream")
    parser.add_argument("--asset-dir", type=Path, default=PROTECTION_ASSET_ROOT / "m2f2_det")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--work-dir", type=Path, default=PROTECTION_WORKDIR_ROOT / "m2f2_det")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--test-split", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Smoke-only override for smaller local inference batches.")
    parser.add_argument("--num-frames", type=int, default=None, help="Smoke-only frame cap per video key; use small real fixtures instead of repeated synthetic copies.")
    parser.add_argument("--allow-random-init", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_bin = os.path.abspath(args.python_bin)
    device = resolve_device(args.device)
    work_dir = args.work_dir.resolve()
    kind, rel_script = SCRIPTS[args.stage]
    script = (args.repo_dir / rel_script).resolve()
    if kind == "python":
        cmd = [python_bin, str(script), *args.stage_args]
    else:
        cmd = ["sh", str(script), *args.stage_args]
    run_command(
        cmd,
        cwd=args.repo_dir,
        env_updates={
            "PYTHONPATH": str(args.repo_dir),
            "MPLCONFIGDIR": str((args.work_dir / "matplotlib").resolve()),
            "TORCH_HOME": str((args.asset_dir / "torch_home").resolve()),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "FYP_TORCH_DEVICE": device,
            "FYP_ALLOW_RANDOM_INIT": "1" if args.allow_random_init else "0",
            "FYP_M2F2_ASSET_DIR": str(args.asset_dir.resolve()),
            "FYP_M2F2_CLIP_DIR": str((args.asset_dir / "huggingface" / "openai--clip-vit-large-patch14-336").resolve()),
            "FYP_M2F2_DATA_ROOT": str(args.data_root.resolve()) if args.data_root is not None else "",
            "FYP_M2F2_TEST_SPLIT": str(args.test_split.resolve()) if args.test_split is not None else "",
            "FYP_M2F2_BATCH_SIZE": str(int(args.batch_size)) if args.batch_size is not None else "",
            "FYP_M2F2_NUM_FRAMES": str(int(args.num_frames)) if args.num_frames is not None else "",
            "FYP_M2F2_RUN_DIR": str((work_dir / args.stage).resolve()),
            "CUDA_VISIBLE_DEVICES": "0" if device == "cuda" else "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )


if __name__ == "__main__":
    main()
