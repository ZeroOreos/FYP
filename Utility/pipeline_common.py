#!/usr/bin/env python3
# dataset_dir + model registry -> cached pairs / embeddings / metrics paths and subprocess runs

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from Utility.pathfinder import resolve_dataset_context


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
PAIRS_ROOT = DATASET_ROOT / "pairs"
RESULTS_ROOT = PROJECT_ROOT / "Results"
PAIRS_SCRIPT = PROJECT_ROOT / "Utility" / "pairs.py"

THROTTLE_BATCH_SIZE = 32
MODEL_THROTTLE_DELAYS = {
    "InsightFace": 0.02,
    "FaceNet": 0.5,
}
ALLOW_MISSING_PAIRS = True


class ModelSpec(TypedDict):
    name: str
    generate_script: Path
    evaluate_script: Path


MODELS: list[ModelSpec] = [
    {
        "name": "InsightFace",
        "generate_script": PROJECT_ROOT / "Models" / "InsightFace" / "generate.py",
        "evaluate_script": PROJECT_ROOT / "Models" / "InsightFace" / "evaluate.py",
    },
    {
        "name": "FaceNet",
        "generate_script": PROJECT_ROOT / "Models" / "FaceNet" / "generate.py",
        "evaluate_script": PROJECT_ROOT / "Models" / "FaceNet" / "evaluate.py",
    },
]


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
        if not model["evaluate_script"].exists():
            raise FileNotFoundError(f"{model['name']} evaluate.py not found")


def pairs_output_path(dataset_dir: Path) -> Path:
    context = resolve_dataset_context(dataset_dir)
    return context.dataset_root / "pairs" / context.pair_filename


def model_results_dir(variant_name: str, model_name: str) -> Path:
    return RESULTS_ROOT / variant_name / model_name


def embeddings_output_path(variant_name: str, model_name: str) -> Path:
    return model_results_dir(variant_name, model_name) / "embeddings.npz"


def metrics_output_path(variant_name: str, model_name: str) -> Path:
    return model_results_dir(variant_name, model_name) / "metrics.json"


def maybe_run_pairs(dataset_dir: Path) -> Path:
    ensure_dir(PAIRS_ROOT)
    context = resolve_dataset_context(dataset_dir)
    out_path = pairs_output_path(dataset_dir)

    if out_path.exists():
        print(f"[SKIP] pairs already exists for base root '{context.base_root_name}': {out_path}")
        return out_path

    cmd = [sys.executable, str(PAIRS_SCRIPT), str(dataset_dir), "--pairs-out", str(out_path)]
    run_subprocess(cmd, f"pairs.py -> {out_path}")
    return out_path


def maybe_run_generate(
    dataset_dir: Path,
    variant_name: str,
    model: ModelSpec,
    throttle_enabled: bool = False,
) -> Path:
    model_name = model["name"]
    script_path = model["generate_script"]
    out_path = embeddings_output_path(variant_name, model_name)

    ensure_dir(out_path.parent)

    if out_path.exists():
        print(f"[SKIP] {model_name} embeddings exist")
        return out_path

    cmd = [sys.executable, str(script_path), str(dataset_dir), str(out_path)]

    if throttle_enabled:
        throttle_delay = MODEL_THROTTLE_DELAYS.get(model_name, 0.0)
        if model_name == "FaceNet":
            cmd.append(str(THROTTLE_BATCH_SIZE))
            cmd.append(str(throttle_delay))
        elif model_name == "InsightFace":
            cmd.append(str(throttle_delay))

    run_subprocess(cmd, f"{model_name} generate -> {out_path}")
    return out_path


def maybe_run_evaluate(variant_name: str, model: ModelSpec, pairs_file: Path, embeddings_file: Path) -> Path:
    model_name = model["name"]
    script_path = model["evaluate_script"]
    out_path = metrics_output_path(variant_name, model_name)

    ensure_dir(out_path.parent)

    if out_path.exists():
        print(f"[SKIP] {model_name} metrics exist")
        return out_path

    cmd = [
        sys.executable,
        str(script_path),
        str(pairs_file),
        str(embeddings_file),
        str(out_path),
    ]

    if ALLOW_MISSING_PAIRS:
        cmd.append("--allow-missing-pairs")

    run_subprocess(cmd, f"{model_name} evaluate -> {out_path}")
    return out_path
