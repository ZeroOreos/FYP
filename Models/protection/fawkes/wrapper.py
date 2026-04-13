#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(BOOTSTRAP_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_PROJECT_ROOT))

from Models.protection.shared import PROTECTION_SOURCE_ROOT, PROTECTION_WORKDIR_ROOT, collect_images
from Models.protection.shared import copy_stage_outputs, ensure_dir, group_images_by_parent, resolve_device
from Models.protection.shared import run_command, stage_group_copy, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fawkes protection over a dataset tree.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-dir", type=Path, default=PROTECTION_SOURCE_ROOT / "Fawkes_upstream")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--work-dir", type=Path, default=PROTECTION_WORKDIR_ROOT / "fawkes")
    parser.add_argument("--mode", choices=("low", "mid", "high"), default="low")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--format", choices=("png", "jpg"), default="png")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--no-align", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = collect_images(args.dataset_dir)
    device = resolve_device(args.device)
    groups = group_images_by_parent(items)
    all_records: list[dict[str, str]] = []

    for parent, group_items in groups.items():
        stage_dir = ensure_dir(args.work_dir / "staging" / parent)
        mapping = stage_group_copy(group_items, stage_dir)
        cmd = [
            os.path.abspath(args.python_bin),
            str((args.repo_dir / "fawkes" / "protection.py").resolve()),
            "--directory",
            str(stage_dir),
            "--mode",
            args.mode,
            "--batch-size",
            str(int(args.batch_size)),
            "--format",
            args.format,
        ]
        if args.no_align:
            cmd.append("--no-align")
        if device == "cuda":
            cmd.extend(["--gpu", "0"])
        run_command(cmd, cwd=args.repo_dir, env_updates={"PYTHONPATH": str(args.repo_dir)})
        all_records.extend(copy_stage_outputs(stage_dir, args.output_dir, mapping, suffix="_cloaked"))

    write_json(args.output_dir / "_fawkes_summary.json", {"device": device, "records": all_records})


if __name__ == "__main__":
    main()
