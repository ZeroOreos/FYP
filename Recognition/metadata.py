#!/usr/bin/env python3
# Recognition transform metadata contract and request-matching helpers.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Utility.paths import resolve_dataset_context


METADATA_FILENAME = "transform.json"


def metadata_path(output_dir: Path) -> Path:
    return output_dir / METADATA_FILENAME


def build_transform_metadata(
    dataset_dir: Path,
    output_dir: Path,
    steps: list[Any],
    seed: int,
    num_workers: int,
) -> dict[str, Any]:
    context = resolve_dataset_context(dataset_dir)
    return {
        "input_dataset_dir": str(dataset_dir.resolve()),
        "output_dataset_dir": str(output_dir.resolve()),
        "base_dataset": context.base_dataset_name,
        "referenced_base_root": context.base_root_name,
        "variant_name": output_dir.name,
        "variant_type": "single" if len(steps) == 1 and context.variant_type == "clean" else "hybrid",
        "pipeline": [
            {
                "modifier": step.spec_name,
                "token": step.token,
                "params": step.params,
            }
            for step in steps
        ],
        "seed": seed,
        "num_workers": num_workers,
    }


def request_matches(metadata_file: Path, expected: dict[str, Any]) -> bool:
    with open(metadata_file, "r", encoding="utf-8") as handle:
        return json.load(handle) == expected


def write_transform_metadata(
    metadata_file: Path,
    metadata: dict[str, Any],
) -> Path:
    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_file
