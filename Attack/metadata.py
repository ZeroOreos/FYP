#!/usr/bin/env python3
# Attack metadata contract and request-matching helpers.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Utility.paths import resolve_dataset_context


METADATA_FILENAME = "attack_metadata.json"


def metadata_path(output_dir: Path) -> Path:
    return output_dir / METADATA_FILENAME


def build_request_metadata(
    dataset_dir: Path,
    output_dir: Path,
    step: Any,
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
        "attack_method": step.name,
        "generator": str(generator_script.resolve()),
        "seed": seed,
        "pair_input": str(pair_input.resolve()),
        "generator_params": step.params,
        "folder_identity_rule": "folder name is victim / claimed identity",
        "filename_rule": "output filename preserves attacker/source filename",
        "attacker_resolution_rule": "resolve attacker from pair file and preserved filename",
        "path_semantics_are_primary": True,
    }


def request_matches(metadata_file: Path, expected: dict[str, Any]) -> bool:
    with open(metadata_file, "r", encoding="utf-8") as handle:
        existing = json.load(handle)
    return all(existing.get(key) == value for key, value in expected.items())


def write_attack_metadata(
    metadata_file: Path,
    expected_request: dict[str, Any],
    num_requested_pairs: int,
    num_written: int,
    failures: list[dict[str, str]],
) -> Path:
    metadata = {
        **expected_request,
        "num_requested_pairs": num_requested_pairs,
        "num_written": num_written,
        "num_failed": len(failures),
        "failures": failures,
    }
    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_file
