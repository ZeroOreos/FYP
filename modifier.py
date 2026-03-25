#!/usr/bin/env python3
# python3 modifier.py <dataset_dir> --step <spec> [--step ...] -> wrapper around Recognition/materialize.py and Attack/materialize.py

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from Attack.materialize import ATTACK_METHODS
from Recognition.materialize import REGISTRY as MODIFY_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parent
MODIFY_SCRIPT = PROJECT_ROOT / "Recognition" / "materialize.py"
ATTACK_SCRIPT = PROJECT_ROOT / "Attack" / "materialize.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset materialization wrapper for non-attack and attack variants.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--step", action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pair-input", type=Path, default=None)
    parser.add_argument("--generator-script", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    if args.pair_input is not None:
        args.pair_input = args.pair_input.resolve()
    if args.generator_script is not None:
        args.generator_script = args.generator_script.resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
    return args


def is_recognition_materialize_mode(steps: list[str], generator_script: str | None) -> bool:
    if generator_script:
        return False
    for raw in steps:
        name = raw.split(":", 1)[0].strip()
        if name not in MODIFY_REGISTRY:
            return False
    return True


def validate_attack_steps(steps: list[str]) -> None:
    for raw in steps:
        name = raw.split(":", 1)[0].strip()
        if name not in ATTACK_METHODS:
            raise ValueError(f"unknown attack method '{name}'")


def build_subcommand(args: argparse.Namespace) -> list[str]:
    script = MODIFY_SCRIPT if is_recognition_materialize_mode(args.step, args.generator_script) else ATTACK_SCRIPT
    cmd = [sys.executable, str(script), str(args.dataset_dir)]
    for step in args.step:
        cmd.extend(["--step", step])
    cmd.extend(["--seed", str(args.seed)])

    if script == MODIFY_SCRIPT:
        if args.num_workers is not None:
            cmd.extend(["--num-workers", str(args.num_workers)])
        if args.pair_input:
            cmd.extend(["--pair-input", str(args.pair_input)])
        if args.overwrite:
            cmd.append("--overwrite")
        if args.force:
            cmd.append("--force")
    else:
        validate_attack_steps(args.step)
        if not args.pair_input:
            raise ValueError("attack materialization requires --pair-input")
        if not args.generator_script:
            raise ValueError("attack materialization currently requires --generator-script")
        cmd.extend(["--pair-input", str(args.pair_input)])
        cmd.extend(["--generator-script", str(args.generator_script)])
        if args.output_dir:
            cmd.extend(["--output-dir", str(args.output_dir)])
        if args.force:
            cmd.append("--force")
    return cmd


def main() -> None:
    args = parse_args()
    cmd = build_subcommand(args)
    print("[INFO] modifier.py dispatch")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
