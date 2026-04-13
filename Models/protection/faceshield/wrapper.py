#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(BOOTSTRAP_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_PROJECT_ROOT))

from Models.protection.shared import PROTECTION_ASSET_ROOT, PROTECTION_SOURCE_ROOT, PROTECTION_WORKDIR_ROOT
from Models.protection.shared import resolve_device, run_command


def parse_args() -> argparse.Namespace:
    asset_root = PROTECTION_ASSET_ROOT / "faceshield"
    parser = argparse.ArgumentParser(description="Run FaceShield's official attack.py wrapper.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-dir", type=Path, default=PROTECTION_SOURCE_ROOT / "FaceShield_upstream")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--work-dir", type=Path, default=PROTECTION_WORKDIR_ROOT / "faceshield")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--resize-shape", type=int, default=512)
    parser.add_argument("--proj-func", default="l1")
    parser.add_argument("--attn-func", default="l2")
    parser.add_argument("--attn-threshold", type=float, default=0.2)
    parser.add_argument("--arc-func", default="cosine")
    parser.add_argument("--total-iter", type=int, default=30)
    parser.add_argument("--noise-clamp", type=int, default=12)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--model-path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--vae-model-path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--unet-config", type=Path, default=PROTECTION_SOURCE_ROOT / "FaceShield_upstream" / "utils" / "unet" / "unet_config15.json")
    parser.add_argument("--pretrained-ip-adapter-path", type=Path, default=PROTECTION_SOURCE_ROOT / "FaceShield_upstream" / "utils" / "unet" / "ip_adapter" / "ip-adapter_sd15.bin")
    parser.add_argument("--image-encoder-path", type=str, default="h94/IP-Adapter")
    parser.add_argument("--pretrained-arcface50-path", type=Path, default=asset_root / "models" / "arcface50_checkpoint.tar")
    parser.add_argument("--pretrained-arcface100-path", type=Path, default=asset_root / "models" / "arcface100_checkpoint.tar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device != "cuda":
        raise RuntimeError("FaceShield's upstream inference path is CUDA-only right now; auto resolved to a non-CUDA device on this machine.")

    cmd = [
        str(Path(args.python_bin).resolve()),
        str((args.repo_dir / "attack.py").resolve()),
        "--model_path",
        args.model_path,
        "--vae_model_path",
        args.vae_model_path,
        "--unet_config",
        str(args.unet_config.resolve()),
        "--pretrained_ip_adapter_path",
        str(args.pretrained_ip_adapter_path.resolve()),
        "--image_encoder_path",
        args.image_encoder_path,
        "--pretrained_arcface50_path",
        str(args.pretrained_arcface50_path.resolve()),
        "--pretrained_arcface100_path",
        str(args.pretrained_arcface100_path.resolve()),
        "--save_path",
        str(args.output_dir.resolve()),
        "--resize_shape",
        str(int(args.resize_shape)),
        "--proj_func",
        args.proj_func,
        "--attn_func",
        args.attn_func,
        "--attn_threshold",
        str(float(args.attn_threshold)),
        "--arc_func",
        args.arc_func,
        "--total_iter",
        str(int(args.total_iter)),
        "--noise_clamp",
        str(int(args.noise_clamp)),
        "--step_size",
        str(float(args.step_size)),
        "--image_path",
        str(args.image_path.resolve()),
    ]
    run_command(
        cmd,
        cwd=args.repo_dir,
        env_updates={
            "PYTHONPATH": str(args.repo_dir),
            "MPLCONFIGDIR": str((args.work_dir / "matplotlib").resolve()),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )


if __name__ == "__main__":
    main()
