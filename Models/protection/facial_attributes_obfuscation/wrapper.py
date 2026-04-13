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


GENERATOR_PATH_FLAGS = {
    "--celeba_image_dir",
    "--attr_path",
    "--rafd_image_dir",
    "--log_dir",
    "--model_save_dir",
    "--sample_dir",
    "--result_dir",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin wrapper around Facial Attributes Obfuscation stages.")
    parser.add_argument("stage", choices=("stage1", "generator-train", "generator-test"))
    parser.add_argument("stage_args", nargs=argparse.REMAINDER)
    parser.add_argument("--repo-dir", type=Path, default=PROTECTION_SOURCE_ROOT / "Facial_Attributes_Obfuscation_upstream")
    parser.add_argument("--asset-dir", type=Path, default=PROTECTION_ASSET_ROOT / "facial_attributes_obfuscation")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--work-dir", type=Path, default=PROTECTION_WORKDIR_ROOT / "facial_attributes_obfuscation")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--allow-random-init", action="store_true")
    return parser.parse_args()


def resolve_generator_stage_args(stage_args: list[str]) -> list[str]:
    resolved = []
    i = 0
    while i < len(stage_args):
        arg = stage_args[i]
        resolved.append(arg)
        if arg in GENERATOR_PATH_FLAGS and i + 1 < len(stage_args):
            resolved.append(str(Path(stage_args[i + 1]).resolve()))
            i += 2
            continue
        i += 1
    return resolved


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.device)
    python_bin = os.path.abspath(args.python_bin)
    asset_dir = args.asset_dir.resolve()
    repo_dir = args.repo_dir.resolve()
    work_dir = args.work_dir.resolve()
    env_updates = {
        "PYTHONPATH": str(repo_dir),
        "MPLCONFIGDIR": str((work_dir / "matplotlib").resolve()),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "FYP_TORCH_DEVICE": resolved_device,
        "CUDA_VISIBLE_DEVICES": "0" if resolved_device == "cuda" else "",
    }
    if args.stage == "stage1":
        cls_device = "cuda:0" if resolved_device == "cuda" else resolved_device
        seg_device = "cuda:0" if resolved_device == "cuda" else resolved_device
        attr_device = "cuda:0" if resolved_device == "cuda" else resolved_device
        cmd = [
            python_bin,
            "-m",
            "utils.score",
            "--mode",
            "arcface",
            "--dataset-root",
            str((repo_dir / "images").resolve()),
            "--data-list",
            str((repo_dir / "images" / "label.txt").resolve()),
            "--num-classes",
            "8631",
            "--pre-trained",
            str((asset_dir / "pretrained" / "ArcFace-8631.pth").resolve()),
            "--seg-pre-trained",
            str((asset_dir / "pretrained" / "FaceParser.ckpt").resolve()),
            "--attr-pre-trained",
            str((asset_dir / "pretrained" / "Face-Attributes2.pth").resolve()),
            "--cls-device",
            cls_device,
            "--seg-device",
            seg_device,
            "--attr-device",
            attr_device,
            "--save-dir",
            str((work_dir / "results" / "arc_mode").resolve()),
            *args.stage_args,
        ]
        run_command(cmd, cwd=repo_dir, env_updates=env_updates)
        return

    mode = "train" if args.stage == "generator-train" else "test"
    stage_args = resolve_generator_stage_args(list(args.stage_args))
    cmd = [
        python_bin,
        str((args.repo_dir / "Generator" / "main.py").resolve()),
        "--mode",
        mode,
        "--device",
        resolved_device,
        "--senet-weights",
        str((asset_dir / "pretrained" / "resnet50_scratch_weight.pkl").resolve()),
        "--lpips-weights",
        str((repo_dir / "Generator" / "metrics" / "lpips_weights.ckpt").resolve()),
        *stage_args,
    ]
    if args.allow_random_init:
        cmd.append("--allow-random-init")
    run_command(cmd, cwd=repo_dir / "Generator", env_updates=env_updates)


if __name__ == "__main__":
    main()
