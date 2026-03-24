#!/usr/bin/env python3
# python3 Utility/attack.py <dataset_dir> --step <attack_spec> --pair-input <atkpairs> --generator-script <script> -> attacked probe dataset + compact metadata

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from Modifiers.attack.advfacegan import SPEC as ADVFACEGAN_SPEC
from Modifiers.attack.faceshifter import SPEC as FACESHIFTER_SPEC
from Modifiers.attack.fomm import SPEC as FOMM_SPEC
from Modifiers.attack.liveportrait import SPEC as LIVEPORTRAIT_SPEC
from Modifiers.attack.mipgan import SPEC as MIPGAN_SPEC
from Modifiers.attack.mordiff import SPEC as MORDIFF_SPEC
from Modifiers.attack.simswap import SPEC as SIMSWAP_SPEC
from Utility.pathfinder import resolve_dataset_context


METADATA_FILENAME = "attack_metadata.json"
ATTACK_SPECS = (
    SIMSWAP_SPEC,
    FACESHIFTER_SPEC,
    FOMM_SPEC,
    LIVEPORTRAIT_SPEC,
    ADVFACEGAN_SPEC,
    MIPGAN_SPEC,
    MORDIFF_SPEC,
)
ATTACK_REGISTRY = {spec["name"]: spec for spec in ATTACK_SPECS}


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


def _format_token_value(value: Any) -> str:
    text = str(value)
    return text.replace("-", "m").replace(".", "p")


def parse_attack_step(raw: str) -> AttackStep:
    if ":" in raw:
        name, raw_params = raw.split(":", 1)
    else:
        name, raw_params = raw, ""

    name = name.strip()
    if not name:
        raise ValueError("attack step name cannot be empty")
    if name not in ATTACK_REGISTRY:
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
        suffix = "_".join(f"{key}{_format_token_value(params[key])}" for key in sorted(params))
        token = f"{name}_{suffix}"
    return AttackStep(name=name, params=params, token=token)


def derive_output_variant_name(dataset_dir: Path, step: AttackStep) -> str:
    context = resolve_dataset_context(dataset_dir)
    existing_tokens: list[str] = []
    if context.variant_name != context.base_root_name and context.transform_chain != "clean":
        existing_tokens.extend(token for token in context.transform_chain.split("__") if token)
    tokens = [step.token] + existing_tokens
    return f"{'__'.join(tokens)}_{context.base_root_name}"


def derive_output_dataset_dir(dataset_dir: Path, step: AttackStep) -> Path:
    context = resolve_dataset_context(dataset_dir)
    return context.dataset_root.joinpath(*context.dataset_relative_parts[:-1], derive_output_variant_name(dataset_dir, step))


def metadata_path(output_dir: Path) -> Path:
    return output_dir / METADATA_FILENAME


def load_attack_pairs(pair_input: Path) -> list[dict[str, str]]:
    data = np.load(pair_input, allow_pickle=True)
    required = ("victim_identity", "attacker_identity", "victim_image", "attacker_image")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"attack pair input missing required keys {missing}: {pair_input}")

    victim_identity = [str(value) for value in data["victim_identity"].tolist()]
    attacker_identity = [str(value) for value in data["attacker_identity"].tolist()]
    victim_image = [str(value) for value in data["victim_image"].tolist()]
    attacker_image = [str(value) for value in data["attacker_image"].tolist()]

    rows: list[dict[str, str]] = []
    for victim_id, attacker_id, victim_img, attacker_img in zip(
        victim_identity,
        attacker_identity,
        victim_image,
        attacker_image,
    ):
        rows.append({
            "victim_identity": victim_id,
            "attacker_identity": attacker_id,
            "victim_image": victim_img,
            "attacker_image": attacker_img,
        })
    return rows


def expected_output_path(output_dir: Path, row: dict[str, str]) -> Path:
    return output_dir / row["victim_identity"] / Path(row["attacker_image"]).name


def infer_attack_family(step_name: str) -> str:
    return ATTACK_REGISTRY[step_name].family


def build_request_metadata(
    dataset_dir: Path,
    output_dir: Path,
    step: AttackStep,
    pair_input: Path,
    generator_script: Path,
    seed: int,
) -> dict[str, Any]:
    context = resolve_dataset_context(dataset_dir)
    return {
        "input_dataset_dir": str(dataset_dir.resolve()),
        "output_dataset_dir": str(output_dir.resolve()),
        "base_dataset": context.base_dataset_name,
        "referenced_base_root": context.base_root_name,
        "variant_name": output_dir.name,
        "attack_family": infer_attack_family(step.name),
        "attack_method": step.name,
        "attack_category": ATTACK_REGISTRY[step.name].category,
        "paper_title": ATTACK_REGISTRY[step.name].paper_title,
        "paper_url": ATTACK_REGISTRY[step.name].paper_url,
        "code_url": ATTACK_REGISTRY[step.name].code_url,
        "generator": str(generator_script.resolve()),
        "seed": seed,
        "pair_input": str(pair_input.resolve()),
        "generator_params": step.params,
        "folder_identity_rule": "folder name is victim / claimed identity",
        "filename_rule": "output filename preserves attacker/source filename",
        "attacker_resolution_rule": "resolve attacker from pair file and preserved filename",
        "path_semantics_are_primary": True,
    }


def compare_existing_metadata(existing_path: Path, expected: dict[str, Any]) -> bool:
    with open(existing_path, "r", encoding="utf-8") as handle:
        existing = json.load(handle)
    for key, value in expected.items():
        if existing.get(key) != value:
            return False
    return True


def run_generator(dataset_dir: Path, pair_input: Path, output_dir: Path, generator_script: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_records = output_dir / "_generator_records.json"
    cmd = [
        sys.executable,
        str(generator_script),
        str(dataset_dir),
        str(pair_input),
        str(output_dir),
        str(temp_records),
    ]
    print("\n[RUN] attack generator")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
        if compare_existing_metadata(meta_path, expected_request):
            print(f"[SKIP] attack dataset already matches request: {output_dir}")
            return meta_path
        if not force:
            raise RuntimeError(
                "existing attack dataset metadata does not match the current request; "
                "refuse to reuse or overwrite without explicit cleanup"
            )

    pair_rows = load_attack_pairs(pair_input)
    run_generator(dataset_dir, pair_input, output_dir, generator_script)

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

    metadata = {
        **expected_request,
        "num_requested_pairs": len(pair_rows),
        "num_written": num_written,
        "num_failed": len(failures),
        "failures": failures,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"[INFO] attack output: {output_dir}")
    print(f"[INFO] metadata: {meta_path}")
    print(f"[INFO] requested pairs: {len(pair_rows)}")
    print(f"[INFO] written: {num_written}")
    print(f"[INFO] failed: {len(failures)}")
    return meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize an attack probe dataset from attack pairs.")
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--step", action="append", required=True)
    parser.add_argument("--pair-input", type=str, required=True)
    parser.add_argument("--generator-script", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.step) != 1:
        raise ValueError("attack materialization currently expects exactly one attack step per run")

    dataset_dir = Path(args.dataset_dir).resolve()
    pair_input = Path(args.pair_input).resolve()
    generator_script = Path(args.generator_script).resolve()
    step = parse_attack_step(args.step[0])

    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not pair_input.exists():
        raise FileNotFoundError(pair_input)
    if not generator_script.exists():
        raise FileNotFoundError(generator_script)

    materialize_attack_dataset(
        dataset_dir=dataset_dir,
        step=step,
        pair_input=pair_input,
        generator_script=generator_script,
        seed=args.seed,
        force=bool(args.force),
        output_dir_override=Path(args.output_dir).resolve() if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
