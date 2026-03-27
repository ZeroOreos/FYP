#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from Modifiers.attack.shared import build_output_path, copy_image_file, create_still_video_ffmpeg, external_backend_defaults
from Modifiers.attack.shared import external_backend_extra_lines, extract_first_frame_ffmpeg, find_first_existing_file, find_latest_file
from Modifiers.attack.shared import load_pair_rows, make_record, prepare_pair_workdir, print_run_header
from Modifiers.attack.shared import require_existing_paths, resolve_external_backend_args, run_command, write_summary


DEFAULTS = external_backend_defaults(
    "fomm",
    "FOMM_upstream",
    default_entry_script="demo.py",
    default_checkpoint="vox-cpk.pth.tar",
    default_config="config/vox-256.yaml",
)


def should_force_cpu(explicit: bool) -> bool:
    if explicit:
        return True
    try:
        import torch
    except Exception:  # pragma: no cover
        return False

    if torch.cuda.is_available():
        return False
    mps = getattr(torch.backends, "mps", None)
    return not bool(mps is not None and torch.backends.mps.is_available())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FOMM attack probes with an external backend wrapper.")
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
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--adapt-scale", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return resolve_external_backend_args(
        parser.parse_args(),
        defaults=DEFAULTS,
        repo_env="FYP_FOMM_REPO_DIR",
        entry_env="FYP_FOMM_ENTRY_SCRIPT",
        checkpoint_env="FYP_FOMM_CHECKPOINT",
        config_env="FYP_FOMM_CONFIG",
    )


def build_command(args: argparse.Namespace, source_path: Path, driving_video: Path, pair_dir: Path) -> tuple[list[str], Path]:
    if args.entry_script is None or args.config is None or args.checkpoint is None:
        raise ValueError("FOMM wrapper needs an entry script, config, and checkpoint. Put the upstream repo in external/ and weights in checkpoints/, or pass them explicitly.")
    result_video = pair_dir / "result.mp4"
    cmd = [
        args.python_bin,
        str(args.entry_script),
        "--config",
        str(args.config),
        "--driving_video",
        str(driving_video),
        "--source_image",
        str(source_path),
        "--checkpoint",
        str(args.checkpoint),
        "--result_video",
        str(result_video),
    ]
    if should_force_cpu(bool(args.force_cpu)):
        cmd.append("--cpu")
    if args.relative:
        cmd.append("--relative")
    if args.adapt_scale:
        cmd.append("--adapt_scale")
    cmd.extend(args.extra_arg)
    return cmd, result_video


def build_env_updates(args: argparse.Namespace) -> dict[str, str]:
    matplotlib_dir = args.work_dir / "_matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    return {"MPLCONFIGDIR": str(matplotlib_dir)}


def resolve_output_file(result_video: Path, pair_dir: Path) -> Path | None:
    if result_video.exists():
        return result_video
    return find_first_existing_file([pair_dir / "result.gif"]) or find_latest_file(pair_dir, ("**/*.mp4", "**/*.gif"))


def main() -> None:
    args = parse_args()
    require_existing_paths(args.dataset_dir, args.pair_input)
    rows = load_pair_rows(args.pair_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print_run_header(
        method_name="fomm_wrapper",
        dataset_dir=args.dataset_dir,
        pair_input=args.pair_input,
        output_dir=args.output_dir,
        num_pairs=len(rows),
        extra_lines=external_backend_extra_lines(
            repo_dir=args.repo_dir,
            entry_script=args.entry_script,
            checkpoint=args.checkpoint,
            config=args.config,
            note="backend fidelity: best effort wrapper around external upstream repo; CPU mode auto-enabled when CUDA/MPS are unavailable",
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
            driving_video = create_still_video_ffmpeg(target_path, pair_dir / "driving.mp4", seconds=float(args.seconds), fps=int(args.fps))
            cmd, result_video = build_command(args, source_path, driving_video, pair_dir)
            run_command(cmd, cwd=args.repo_dir, env_updates=build_env_updates(args), stage_name="fomm")
            generated_video = resolve_output_file(result_video, pair_dir)
            if generated_video is None:
                raise RuntimeError("FOMM backend completed but no output video was found.")
            frame_path = extract_first_frame_ffmpeg(generated_video, pair_dir / "frame.png")
            copy_image_file(frame_path, output_path)
            records.append(make_record(index=index, row=row, output_path=output_path, status="ok", message="generated", source_path=source_path, target_path=target_path))
        except Exception as exc:  # pragma: no cover
            records.append(make_record(index=index, row=row, output_path=output_path, status="failed", message=str(exc), source_path=source_path, target_path=target_path))
            print(f"[WARN] Pair failed {index}: {exc}")

    write_summary(
        method_name="fomm_wrapper",
        dataset_dir=args.dataset_dir,
        pair_input=args.pair_input,
        output_dir=args.output_dir,
        records_out=args.records_out,
        args=args,
        records=records,
        extra_summary={"backend_type": "external_wrapper", "fidelity_note": "Best-effort wrapper around the original First Order Motion Model repository layout."},
    )


if __name__ == "__main__":
    main()
