#!/usr/bin/env python3
# python3 main_attack_slowed.py <gallery_dir> [pause options] -> slowed attack pipeline

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import psutil  # noqa: F401
except ImportError:
    print(
        "[ERROR] psutil is required for slowed/main_attack_slowed.py.\n"
        "Install it with:\n"
        "    python3 -m pip install psutil",
        file=sys.stderr,
    )
    sys.exit(1)

from Shared.slowed import add_pause_args
from Shared.slowed import resolve_main_script, run_paused_subprocess, validate_input_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run main_attack.py with thermal-aware process-level pausing.")
    parser.add_argument("gallery_dir", type=str, help="Path to the clean gallery dataset directory.")
    parser.add_argument("--main-script", type=str, default=None, help="Path to main_attack.py.")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter to use.")

    probe_group = parser.add_mutually_exclusive_group(required=True)
    probe_group.add_argument("--probe-dir", type=str, help="Existing attack probe dataset root.")
    probe_group.add_argument("--attack-method", type=str, help="Attack method token used to derive the probe root.")

    parser.add_argument("--attack-generator-script", type=str, default=None, help="Optional generator backend script.")
    parser.add_argument("--pair-model", type=str, default="InsightFace", help="Model used to generate attack pairs.")
    parser.add_argument("--top-k", type=int, default=5, help="Nearest non-match identities kept per victim.")
    parser.add_argument("--samples-per-identity-pair", type=int, default=3, help="Pairs kept per identity pair.")
    parser.add_argument("--pairing-mode", choices=("hard", "semi_hard"), default="hard", help="Attack pair selection mode.")
    parser.add_argument("--min-identity-sim", type=float, default=None, help="Optional identity similarity floor.")
    parser.add_argument("--min-image-sim", type=float, default=None, help="Optional image similarity floor.")
    add_pause_args(parser)
    return parser.parse_args()


def build_main_attack_cmd(main_script: Path, gallery_dir: Path, args: argparse.Namespace) -> list[str]:
    cmd = [args.python_bin, str(main_script), str(gallery_dir)]
    if args.probe_dir:
        cmd.extend(["--probe-dir", str(Path(args.probe_dir).expanduser().resolve())])
    if args.attack_method:
        cmd.extend(["--attack-method", args.attack_method])
    if args.attack_generator_script:
        cmd.extend(["--attack-generator-script", str(Path(args.attack_generator_script).expanduser().resolve())])
    cmd.extend(["--pair-model", args.pair_model])
    cmd.extend(["--top-k", str(args.top_k)])
    cmd.extend(["--samples-per-identity-pair", str(args.samples_per_identity_pair)])
    cmd.extend(["--pairing-mode", args.pairing_mode])
    if args.min_identity_sim is not None:
        cmd.extend(["--min-identity-sim", str(args.min_identity_sim)])
    if args.min_image_sim is not None:
        cmd.extend(["--min-image-sim", str(args.min_image_sim)])
    return cmd


def main() -> int:
    args = parse_args()
    try:
        gallery_dir = validate_input_dir(args.gallery_dir)
        main_script = resolve_main_script(args.main_script, "main_attack.py")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("[INFO] === slowed/main_attack_slowed.py ===")
    print(f"[INFO] Gallery directory: {gallery_dir}")
    print(f"[INFO] Main script:       {main_script}")
    print(f"[INFO] Python binary:     {args.python_bin}")

    cmd = build_main_attack_cmd(main_script, gallery_dir, args)
    return run_paused_subprocess(cmd, args, main_label="main_attack.py")


if __name__ == "__main__":
    raise SystemExit(main())
