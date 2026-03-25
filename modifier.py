#!/usr/bin/env python3
# python3 modifier.py <dataset_dir> --step <spec> [--step ...] -> wrapper around Utility/modify.py and Attack/materialize.py

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from Attack.materialize import ATTACK_METHODS
from Utility.modify import REGISTRY as MODIFY_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parent
MODIFY_SCRIPT = PROJECT_ROOT / "Utility" / "modify.py"
ATTACK_SCRIPT = PROJECT_ROOT / "Attack" / "materialize.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset materialization wrapper for non-attack and attack variants.")
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--step", action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pair-input", type=str, default=None)
    parser.add_argument("--generator-script", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def is_modify_mode(steps: list[str], generator_script: str | None) -> bool:
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
    script = MODIFY_SCRIPT if is_modify_mode(args.step, args.generator_script) else ATTACK_SCRIPT
    cmd = [sys.executable, str(script), str(Path(args.dataset_dir).resolve())]
    for step in args.step:
        cmd.extend(["--step", step])
    cmd.extend(["--seed", str(args.seed)])

    if script == MODIFY_SCRIPT:
        if args.num_workers is not None:
            cmd.extend(["--num-workers", str(args.num_workers)])
        if args.pair_input:
            cmd.extend(["--pair-input", str(Path(args.pair_input).resolve())])
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
        cmd.extend(["--pair-input", str(Path(args.pair_input).resolve())])
        cmd.extend(["--generator-script", str(Path(args.generator_script).resolve())])
        if args.output_dir:
            cmd.extend(["--output-dir", str(Path(args.output_dir).resolve())])
        if args.force:
            cmd.append("--force")
    return cmd


def main() -> None:
    args = parse_args()
    cmd = build_subcommand(args)
    print("[INFO] modifier.py wrapper dispatch")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
