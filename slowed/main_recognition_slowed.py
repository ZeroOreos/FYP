#!/usr/bin/env python3
# python3 main_recognition_slowed.py <dataset_dir> [pause options] -> slowed recognition pipeline

from __future__ import annotations

import argparse
import sys

try:
    import psutil  # noqa: F401
except ImportError:
    print(
        "[ERROR] psutil is required for slowed/main_recognition_slowed.py.\n"
        "Install it with:\n"
        "    python3 -m pip install psutil",
        file=sys.stderr,
    )
    sys.exit(1)

from Utility.slowed import add_pause_args
from Utility.slowed import resolve_main_script, run_paused_subprocess, validate_input_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run main_recognition.py with thermal-aware process-level pausing.")
    parser.add_argument("input_dir", type=str, help="Path to the input dataset directory.")
    parser.add_argument("--main-script", type=str, default=None, help="Path to main_recognition.py.")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter to use.")
    add_pause_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        input_dir = validate_input_dir(args.input_dir)
        main_script = resolve_main_script(args.main_script, "main_recognition.py")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("[INFO] === slowed/main_recognition_slowed.py ===")
    print(f"[INFO] Input directory:  {input_dir}")
    print(f"[INFO] Main script:      {main_script}")
    print(f"[INFO] Python binary:    {args.python_bin}")

    cmd = [args.python_bin, str(main_script), str(input_dir)]
    return run_paused_subprocess(cmd, args, main_label="main_recognition.py")


if __name__ == "__main__":
    raise SystemExit(main())
