#!/usr/bin/env python3
# python3 Attack/materialize.py <dataset_dir> --step <attack_spec> --pair-input <atkpairs> [--generator-script <script>] -> attacked probe dataset + compact metadata

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from Attack.metadata import build_request_metadata, metadata_path, request_matches, write_attack_metadata
from Utility.paths import resolve_dataset_context
from Utility.runtime import ATTACK_METHODS, resolve_attack_generator_script, run_subprocess


@dataclass(frozen=True)
class AttackStep:
    name: str
    params: dict[str, Any]
    token: str


def parse_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in lowered for char in (".", "e")):
            return float(lowered)
        return int(lowered)
    except ValueError:
        return value


def format_token_value(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def parse_attack_step(raw: str) -> AttackStep:
    name, _, raw_params = raw.partition(":")
    name = name.strip()
    if not name:
        raise ValueError("attack step name cannot be empty")
    if name not in ATTACK_METHODS:
        raise ValueError(f"unknown attack method '{name}'")

    params: dict[str, Any] = {}
    if raw_params.strip():
        for entry in raw_params.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(f"invalid attack parameter '{entry}' in '{raw}'")
            key, value = entry.split("=", 1)
            params[key.strip()] = parse_scalar(value)

    token = name
    if params:
        suffix = "_".join(f"{key}{format_token_value(params[key])}" for key in sorted(params))
        token = f"{name}_{suffix}"
    return AttackStep(name=name, params=params, token=token)


def derive_output_variant_name(dataset_dir: Path, step: AttackStep) -> str:
    context = resolve_dataset_context(dataset_dir)
    existing_tokens: list[str] = []
    if context.variant_name != context.base_root_name and context.transform_chain != "clean":
        existing_tokens.extend(token for token in context.transform_chain.split("__") if token)
    return f"{'__'.join([step.token, *existing_tokens])}_{context.base_root_name}"


def derive_output_dataset_dir(dataset_dir: Path, step: AttackStep) -> Path:
    context = resolve_dataset_context(dataset_dir)
    return context.dataset_root.joinpath(*context.dataset_relative_parts[:-1], derive_output_variant_name(dataset_dir, step))


def load_attack_pairs(pair_input: Path) -> list[dict[str, str]]:
    data = np.load(pair_input, allow_pickle=True)
    required = ("victim_identity", "attacker_identity", "victim_image", "attacker_image")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"attack pair input missing required keys {missing}: {pair_input}")

    return [
        {
            "victim_identity": str(victim_id),
            "attacker_identity": str(attacker_id),
            "victim_image": str(victim_img),
            "attacker_image": str(attacker_img),
        }
        for victim_id, attacker_id, victim_img, attacker_img in zip(
            data["victim_identity"].tolist(),
            data["attacker_identity"].tolist(),
            data["victim_image"].tolist(),
            data["attacker_image"].tolist(),
        )
    ]


def expected_output_path(output_dir: Path, row: dict[str, str]) -> Path:
    return output_dir / row["victim_identity"] / Path(row["attacker_image"]).name


def run_generator(dataset_dir: Path, pair_input: Path, output_dir: Path, generator_script: Path, step: AttackStep) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "_generator_records.json"
    cmd = [
        sys.executable,
        str(generator_script),
        str(dataset_dir),
        str(pair_input),
        str(output_dir),
        str(records_path),
    ]
    for key, value in step.params.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])
    run_subprocess(cmd, f"attack generator -> {output_dir}")


def materialize_attack_dataset(
    dataset_dir: Path,
    step: AttackStep,
    pair_input: Path,
    generator_script: Path,
    seed: int,
    force: bool,
    output_dir_override: Path | None,
) -> Path:
    output_dir = output_dir_override.resolve() if output_dir_override is not None else derive_output_dataset_dir(dataset_dir, step)
    meta_path = metadata_path(output_dir)
    expected_request = build_request_metadata(dataset_dir, output_dir, step, pair_input, generator_script, seed)

    if output_dir.exists() or meta_path.exists():
        if not output_dir.exists() or not meta_path.exists():
            raise RuntimeError(
                f"attack output state is incomplete; expected both dataset dir and metadata: {output_dir} / {meta_path}"
            )
        if request_matches(meta_path, expected_request):
            print(f"[SKIP] attack dataset already matches request: {output_dir}")
            return meta_path
        if not force:
            raise RuntimeError(
                "existing attack dataset metadata does not match the current request; "
                "refuse to reuse or overwrite without explicit cleanup"
            )

    pair_rows = load_attack_pairs(pair_input)
    run_generator(dataset_dir, pair_input, output_dir, generator_script, step)

    failures: list[dict[str, str]] = []
    num_written = 0
    for row in pair_rows:
        output_image = expected_output_path(output_dir, row)
        if output_image.exists():
            num_written += 1
            continue
        failures.append({
            "output_image": str(output_image),
            "victim_identity": row["victim_identity"],
            "attacker_identity": row["attacker_identity"],
            "source_image": row["attacker_image"],
            "target_image": row["victim_image"],
            "error": "expected generated output was not found after generator completed",
        })

    write_attack_metadata(meta_path, expected_request, len(pair_rows), num_written, failures)

    print(f"[INFO] attack output: {output_dir}")
    print(f"[INFO] metadata: {meta_path}")
    print(f"[INFO] requested pairs: {len(pair_rows)}")
    print(f"[INFO] written: {num_written}")
    print(f"[INFO] failed: {len(failures)}")
    return meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build attack probe dataset.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--step", action="append", required=True)
    parser.add_argument("--pair-input", type=Path, required=True)
    parser.add_argument("--generator-script", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.pair_input = args.pair_input.resolve()
    if args.generator_script is not None:
        args.generator_script = args.generator_script.resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    if len(args.step) != 1:
        raise ValueError("attack materialization currently expects exactly one attack step per run")

    dataset_dir = args.dataset_dir
    pair_input = args.pair_input
    step = parse_attack_step(args.step[0])
    generator_script = resolve_attack_generator_script(step.name, args.generator_script)

    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not pair_input.exists():
        raise FileNotFoundError(pair_input)
    materialize_attack_dataset(
        dataset_dir=dataset_dir,
        step=step,
        pair_input=pair_input,
        generator_script=generator_script,
        seed=args.seed,
        force=bool(args.force),
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
