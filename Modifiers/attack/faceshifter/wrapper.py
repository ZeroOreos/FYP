#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from Modifiers.attack.shared import build_extra_args, build_output_path, copy_image_file, external_backend_defaults
from Modifiers.attack.shared import external_backend_extra_lines, find_first_existing_file, find_latest_file
from Modifiers.attack.shared import load_pair_rows, make_record, prepare_pair_workdir, print_run_header
from Modifiers.attack.shared import require_existing_paths, resolve_external_backend_args, run_command, write_summary


DEFAULTS = external_backend_defaults(
    "faceshifter",
    "FaceShifter_upstream",
    default_entry_script="aei_inference.py",
    default_config="config/train.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FaceShifter attack probes with an external backend wrapper.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("pair_input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("records_out", type=Path)
    parser.add_argument("--repo-dir", type=Path, default=None)
    parser.add_argument("--entry-script", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--source-arg", type=str, default="--source_image")
    parser.add_argument("--target-arg", type=str, default="--target_image")
    parser.add_argument("--output-arg", type=str, default="--output_path")
    parser.add_argument("--checkpoint-arg", type=str, default="--checkpoint_path")
    parser.add_argument("--config-arg", type=str, default="--config")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return resolve_external_backend_args(
        parser.parse_args(),
        defaults=DEFAULTS,
        repo_env="FYP_FACESHIFTER_REPO_DIR",
        entry_env="FYP_FACESHIFTER_ENTRY_SCRIPT",
        checkpoint_env="FYP_FACESHIFTER_CHECKPOINT",
        config_env="FYP_FACESHIFTER_CONFIG",
    )


def build_command(args: argparse.Namespace, source_path: Path, target_path: Path, pair_dir: Path) -> tuple[list[str], Path]:
    if args.entry_script is None:
        raise ValueError("FaceShifter wrapper could not find an entry script. Put the backend source in Backends/sources/ or pass --entry-script.")

    result_path = pair_dir / "result.png"
    cmd = [
        args.python_bin,
        str(args.entry_script),
        args.source_arg,
        str(source_path),
        args.target_arg,
        str(target_path),
        args.output_arg,
        str(result_path),
    ]
    if args.config is not None:
        cmd.extend([args.config_arg, str(args.config)])
    if args.checkpoint is not None:
        cmd.extend([args.checkpoint_arg, str(args.checkpoint)])

    token_map = {
        "source": str(source_path),
        "target": str(target_path),
        "output": str(result_path),
        "workdir": str(pair_dir),
        "repo_dir": str(args.repo_dir) if args.repo_dir is not None else "",
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else "",
        "config": str(args.config) if args.config is not None else "",
    }
    cmd.extend(build_extra_args(args.extra_arg, token_map))
    return cmd, result_path


def resolve_output_file(result_path: Path, pair_dir: Path) -> Path | None:
    return find_first_existing_file([result_path]) or find_latest_file(pair_dir, ("**/*.png", "**/*.jpg", "**/*.jpeg"))


def main() -> None:
    args = parse_args()
    require_existing_paths(args.dataset_dir, args.pair_input)
    rows = load_pair_rows(args.pair_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print_run_header(
        method_name="faceshifter_wrapper",
        dataset_dir=args.dataset_dir,
        pair_input=args.pair_input,
        output_dir=args.output_dir,
        num_pairs=len(rows),
        extra_lines=external_backend_extra_lines(
            repo_dir=args.repo_dir,
            entry_script=args.entry_script,
            checkpoint=args.checkpoint,
            config=args.config,
            note="backend fidelity: best effort wrapper around external upstream repo",
        ),
    )

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_path = Path(row["attacker_image"]).resolve()
        target_path = Path(row["victim_image"]).resolve()
        output_path = build_output_path(args.output_dir, row)
        print(f"[PAIR] {index + 1}/{len(rows)} -> {output_path}")

        if output_path.exists() and not args.overwrite:
            records.append(make_record(index=index, row=row, output_path=output_path, status="skipped", message="output already exists", source_path=source_path, target_path=target_path))
            continue

        try:
            pair_dir = prepare_pair_workdir(args.work_dir, index)
            cmd, result_path = build_command(args, source_path, target_path, pair_dir)
            run_command(cmd, cwd=args.repo_dir, stage_name="faceshifter")
            generated = resolve_output_file(result_path, pair_dir)
            if generated is None:
                raise RuntimeError("FaceShifter backend completed but no output image was found.")
            copy_image_file(generated, output_path)
            records.append(make_record(index=index, row=row, output_path=output_path, status="ok", message="generated", source_path=source_path, target_path=target_path))
        except Exception as exc:  # pragma: no cover
            records.append(make_record(index=index, row=row, output_path=output_path, status="failed", message=str(exc), source_path=source_path, target_path=target_path))
            print(f"[WARN] Pair failed {index}: {exc}")

    write_summary(
        method_name="faceshifter_wrapper",
        dataset_dir=args.dataset_dir,
        pair_input=args.pair_input,
        output_dir=args.output_dir,
        records_out=args.records_out,
        args=args,
        records=records,
        extra_summary={"backend_type": "external_wrapper", "fidelity_note": "Best-effort wrapper around the original FaceShifter repository layout."},
    )


if __name__ == "__main__":
    main()
