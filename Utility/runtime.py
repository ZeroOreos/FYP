#!/usr/bin/env python3
# Shared runtime config and process helpers for pipeline entry points.

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypedDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
PAIRS_ROOT = DATASET_ROOT / "pairs"
RESULTS_ROOT = PROJECT_ROOT / "Results"

THROTTLE_BATCH_SIZE = 32
MODEL_THROTTLE_DELAYS = {
    "InsightFace": 0.02,
    "FaceNet": 0.5,
}
ALLOW_MISSING_PAIRS = True


class ModelSpec(TypedDict):
    name: str
    generate_script: Path


MODELS: list[ModelSpec] = [
    {
        "name": "InsightFace",
        "generate_script": PROJECT_ROOT / "Models" / "InsightFace" / "generate.py",
    },
    {
        "name": "FaceNet",
        "generate_script": PROJECT_ROOT / "Models" / "FaceNet" / "generate.py",
    },
]
VERIFY_SCRIPT = PROJECT_ROOT / "Recognition" / "evaluate.py"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_input_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")


def run_subprocess(cmd: list[str], stage_name: str) -> None:
    print(f"\n[RUN] {stage_name}")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def validate_model_registry(models: list[ModelSpec]) -> None:
    if not models:
        raise ValueError("MODELS is empty.")

    names = [model["name"] for model in models]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate model names: {names}")

    for model in models:
        if not model["generate_script"].exists():
            raise FileNotFoundError(f"{model['name']} generate.py not found")
    if not VERIFY_SCRIPT.exists():
        raise FileNotFoundError(f"Shared verify.py not found: {VERIFY_SCRIPT}")
