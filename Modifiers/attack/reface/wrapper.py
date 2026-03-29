#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from Modifiers.attack.shared import build_output_path, copy_image_file, external_backend_defaults
from Modifiers.attack.shared import external_backend_extra_lines, find_latest_file, load_pair_rows, make_record
from Modifiers.attack.shared import prepare_pair_workdir, print_run_header, require_existing_paths
from Modifiers.attack.shared import resolve_external_backend_args, run_command, write_summary


DEFAULTS = external_backend_defaults(
    "reface",
    "REFace_upstream",
    default_entry_script="scripts/inference_swap_selected.py",
    default_checkpoint="last.ckpt",
    default_config="models/REFace/configs/project_ffhq.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate REFace attack probes with an external backend wrapper.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("pair_input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("records_out", type=Path)
    parser.add_argument("--repo-dir", type=Path, default=None)
    parser.add_argument("--entry-script", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("full", "autocast"), default="full")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return resolve_external_backend_args(
        parser.parse_args(),
        defaults=DEFAULTS,
        repo_env="FYP_REFACE_REPO_DIR",
        entry_env="FYP_REFACE_ENTRY_SCRIPT",
        checkpoint_env="FYP_REFACE_CHECKPOINT",
        config_env="FYP_REFACE_CONFIG",
    )


def build_pair_inputs(pair_dir: Path, source_path: Path, target_path: Path) -> tuple[Path, Path]:
    source_dir = pair_dir / "source"
    target_dir = pair_dir / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_copy = source_dir / source_path.name
    target_copy = target_dir / target_path.name
    copy_image_file(source_path, source_copy)
    copy_image_file(target_path, target_copy)
    return source_dir, target_dir


def build_command(
    args: argparse.Namespace,
    source_dir: Path,
    target_dir: Path,
    pair_dir: Path,
) -> tuple[list[str], Path]:
    if args.entry_script is None or args.config is None or args.checkpoint is None:
        raise ValueError("REFace wrapper needs an entry script, config, and checkpoint. Put the backend source in Backends/sources/ and weights in Backends/assets/, or pass them explicitly.")

    outdir = pair_dir / "run"
    base_dir = pair_dir / "base"
    cmd = [
        args.python_bin,
        str(args.entry_script),
        "--outdir",
        str(outdir),
        "--Base_dir",
        str(base_dir),
        "--target_folder",
        str(target_dir),
        "--src_folder",
        str(source_dir),
        "--config",
        str(args.config),
        "--ckpt",
        str(args.checkpoint),
        "--n_samples",
        "1",
        "--scale",
        str(float(args.scale)),
        "--ddim_steps",
        str(int(args.ddim_steps)),
        "--seed",
        str(int(args.seed)),
        "--precision",
        args.precision,
        "--skip_grid",
    ]
    return cmd, outdir


def resolve_output_file(outdir: Path, pair_dir: Path) -> Path | None:
    del pair_dir
    return find_latest_file(outdir, ("results/**/*.png", "results/**/*.jpg", "**/*.png", "**/*.jpg"))


def build_env_updates(args: argparse.Namespace) -> dict[str, str]:
    matplotlib_dir = args.work_dir / "_matplotlib"
    support_dir = args.checkpoint.parent / "support"
    hf_home = support_dir / "hf_home"
    hf_cache = support_dir / "hf_cache"
    taming_candidates = (
        args.repo_dir.parent / "dependencies" / "taming-transformers",
        args.repo_dir.parent / "taming-transformers",
    )
    taming_dir = next((path for path in taming_candidates if path.exists()), None)
    torch_home = support_dir / "torch_home"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    pythonpath_parts = [str(args.repo_dir)]
    if taming_dir is not None:
        pythonpath_parts.append(str(taming_dir))
    return {
        "MPLCONFIGDIR": str(matplotlib_dir),
        "PYTHONPATH": ":".join(pythonpath_parts),
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_cache),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "REFACE_CLIP_MODEL_DIR": str(hf_cache / "models--openai--clip-vit-large-patch14"),
        "TORCH_HOME": str(torch_home),
    }


def main() -> None:
    args = parse_args()
    require_existing_paths(args.dataset_dir, args.pair_input)
    rows = load_pair_rows(args.pair_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print_run_header(
        method_name="reface_wrapper",
        dataset_dir=args.dataset_dir,
        pair_input=args.pair_input,
        output_dir=args.output_dir,
        num_pairs=len(rows),
        extra_lines=external_backend_extra_lines(
            repo_dir=args.repo_dir,
            entry_script=args.entry_script,
            checkpoint=args.checkpoint,
            config=args.config,
            note="backend fidelity: best effort wrapper around the official REFace repository; CPU execution is a patched approximation of the original CUDA-first path",
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
            source_dir, target_dir = build_pair_inputs(pair_dir, source_path, target_path)
            cmd, outdir = build_command(args, source_dir, target_dir, pair_dir)
            run_command(cmd, cwd=args.repo_dir, env_updates=build_env_updates(args), stage_name="reface")
            generated = resolve_output_file(outdir, pair_dir)
            if generated is None:
                raise RuntimeError("REFace backend completed but no output image was found.")
            copy_image_file(generated, output_path)
            records.append(make_record(index=index, row=row, output_path=output_path, status="ok", message="generated", source_path=source_path, target_path=target_path))
        except Exception as exc:  # pragma: no cover
            records.append(make_record(index=index, row=row, output_path=output_path, status="failed", message=str(exc), source_path=source_path, target_path=target_path))
            print(f"[WARN] Pair failed {index}: {exc}")

    write_summary(
        method_name="reface_wrapper",
        dataset_dir=args.dataset_dir,
        pair_input=args.pair_input,
        output_dir=args.output_dir,
        records_out=args.records_out,
        args=args,
        records=records,
        extra_summary={
            "backend_type": "external_wrapper",
            "fidelity_note": "Best-effort wrapper around the official REFace repository layout. CPU execution here is a patched best attempt, not a byte-faithful reproduction of the original CUDA-first environment.",
        },
    )


if __name__ == "__main__":
    main()
