#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from Modifiers.attack.shared import build_extra_args, build_output_path, copy_image_file, external_backend_defaults
from Modifiers.attack.shared import external_backend_extra_lines, find_first_existing_file, find_latest_file
from Modifiers.attack.shared import load_pair_rows, make_record, prepare_pair_workdir, print_run_header
from Modifiers.attack.shared import require_existing_paths, resolve_external_backend_args, run_command, write_summary
from Utility.runtime import resolve_torch_device


DEFAULTS = external_backend_defaults(
    "simswap",
    "SimSwap_upstream",
    default_entry_script="test_one_image.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SimSwap attack probes with an external backend wrapper.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("pair_input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("records_out", type=Path)
    parser.add_argument("--repo-dir", type=Path, default=None)
    parser.add_argument("--entry-script", type=Path, default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=None)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--name", type=str, default="people")
    parser.add_argument("--use-mask", action="store_true")
    parser.add_argument("--no-simswap-logo", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return resolve_external_backend_args(
        parser.parse_args(),
        defaults=DEFAULTS,
        repo_env="FYP_SIMSWAP_REPO_DIR",
        entry_env="FYP_SIMSWAP_ENTRY_SCRIPT",
    )


def build_command(args: argparse.Namespace, source_path: Path, target_path: Path, pair_dir: Path) -> tuple[list[str], Path]:
    if args.entry_script is None:
        raise ValueError("SimSwap wrapper could not find an entry script. Put the backend source in Backends/sources/ or pass --entry-script.")

    result_dir = pair_dir / "result"
    temp_dir = pair_dir / "temp"
    result_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = f"{result_dir}{os.sep}" if str(result_dir).endswith(os.sep) else f"{result_dir}{os.sep}"
    resolved_device = resolve_torch_device(args.device)
    gpu_ids = "0" if resolved_device == "cuda" else "-1"
    cmd = [
        str(Path(args.python_bin).absolute()),
        str(args.entry_script),
        "--isTrain",
        "false",
        "--name",
        args.name,
        "--gpu_ids",
        gpu_ids,
        "--pic_a_path",
        str(source_path),
        "--pic_b_path",
        str(target_path),
        "--output_path",
        output_path,
        "--temp_path",
        str(temp_dir),
        "--crop_size",
        str(int(args.crop_size)),
    ]
    if args.use_mask:
        cmd.append("--use_mask")
    if args.no_simswap_logo:
        cmd.append("--no_simswaplogo")

    token_map = {
        "source": str(source_path),
        "target": str(target_path),
        "output_dir": str(result_dir),
        "workdir": str(pair_dir),
        "repo_dir": str(args.repo_dir) if args.repo_dir is not None else "",
    }
    cmd.extend(build_extra_args(args.extra_arg, token_map))
    return cmd, result_dir


def resolve_output_file(result_dir: Path, pair_dir: Path) -> Path | None:
    return find_first_existing_file([
        result_dir / "result.jpg",
        result_dir / "result.png",
        result_dir / "swapped.jpg",
        result_dir / "swapped.png",
    ]) or find_latest_file(pair_dir, ("**/*.png", "**/*.jpg", "**/*.jpeg"))


def build_env_updates(args: argparse.Namespace) -> dict[str, str]:
    matplotlib_dir = args.work_dir / "_matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    return {
        "MPLCONFIGDIR": str(matplotlib_dir),
        "FYP_TORCH_DEVICE": resolve_torch_device(args.device),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    }


def main() -> None:
    args = parse_args()
    require_existing_paths(args.dataset_dir, args.pair_input)
    rows = load_pair_rows(args.pair_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print_run_header(
        method_name="simswap_wrapper",
        dataset_dir=args.dataset_dir,
        pair_input=args.pair_input,
        output_dir=args.output_dir,
        num_pairs=len(rows),
        extra_lines=external_backend_extra_lines(
            repo_dir=args.repo_dir,
            entry_script=args.entry_script,
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
            cmd, result_dir = build_command(args, source_path, target_path, pair_dir)
            run_command(cmd, cwd=args.repo_dir, env_updates=build_env_updates(args), stage_name="simswap")
            generated = resolve_output_file(result_dir, pair_dir)
            if generated is None:
                raise RuntimeError("SimSwap backend completed but no output image was found.")
            copy_image_file(generated, output_path)
            records.append(make_record(index=index, row=row, output_path=output_path, status="ok", message="generated", source_path=source_path, target_path=target_path))
        except Exception as exc:  # pragma: no cover
            records.append(make_record(index=index, row=row, output_path=output_path, status="failed", message=str(exc), source_path=source_path, target_path=target_path))
            print(f"[WARN] Pair failed {index}: {exc}")

    write_summary(
        method_name="simswap_wrapper",
        dataset_dir=args.dataset_dir,
        pair_input=args.pair_input,
        output_dir=args.output_dir,
        records_out=args.records_out,
        args=args,
        records=records,
        extra_summary={"backend_type": "external_wrapper", "fidelity_note": "Best-effort wrapper around the original SimSwap repository layout."},
    )


if __name__ == "__main__":
    main()
