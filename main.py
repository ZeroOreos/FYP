#!/usr/bin/env python3
"""
main.py

Orchestrator for the FYP verification workflow.

Stages:
1) pairs.py -> Dataset/pairs/<derived_name>_pairs.npz
2) Models/<Model>/generate.py -> Results/<mod_type>/<Model>/embeddings.npz
3) Models/<Model>/evaluate.py -> Results/<mod_type>/<Model>/metrics.json
4) Rebuild Results/compiled_results.csv from all available metrics files
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, TypedDict



PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "Dataset"
PAIRS_ROOT = DATASET_ROOT / "pairs"

RESULTS_ROOT = PROJECT_ROOT / "Results"
COMPILED_CSV = RESULTS_ROOT / "compiled_results.csv"

PAIRS_SCRIPT = PROJECT_ROOT / "pairs.py"



THROTTLE_ENABLED = False
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


MODELS: List[ModelSpec] = [
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


def derive_pairs_output_path(dataset_dir: Path) -> Path:
    """
    Create deterministic pair output path under Dataset/pairs/ using the
    dataset path relative to Dataset/.

    Example:
        Dataset/CelebA/val -> Dataset/pairs/CelebA_val_pairs.npz
    """
    dataset_dir = dataset_dir.resolve()
    parts = list(dataset_dir.parts)

    if "Dataset" in parts:
        idx = parts.index("Dataset")
        relative_parts = parts[idx + 1 :]
        dataset_root = Path(*parts[: idx + 1])
    else:
        relative_parts = parts[-2:]
        dataset_root = dataset_dir.parent

    name = "_".join(relative_parts)
    return dataset_root / "pairs" / f"{name}_pairs.npz"


def get_modification_type(dataset_dir: Path) -> str:
    return dataset_dir.name


def validate_input_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")


def run_subprocess(cmd: List[str], stage_name: str) -> None:
    print(f"\n[RUN] {stage_name}")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def validate_model_registry(models: List[ModelSpec]) -> None:
    if not models:
        raise ValueError("MODELS is empty.")

    names = [m["name"] for m in models]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate model names: {names}")

    for model in models:
        if not model["generate_script"].exists():
            raise FileNotFoundError(f"{model['name']} generate.py not found")
        if not model["evaluate_script"].exists():
            raise FileNotFoundError(f"{model['name']} evaluate.py not found")


def pairs_output_path(dataset_dir: Path) -> Path:
    return derive_pairs_output_path(dataset_dir)


def model_results_dir(mod_type: str, model_name: str) -> Path:
    return RESULTS_ROOT / mod_type / model_name


def embeddings_output_path(mod_type: str, model_name: str) -> Path:
    return model_results_dir(mod_type, model_name) / "embeddings.npz"


def metrics_output_path(mod_type: str, model_name: str) -> Path:
    return model_results_dir(mod_type, model_name) / "metrics.json"



def maybe_run_pairs(dataset_dir: Path) -> Path:
    ensure_dir(PAIRS_ROOT)
    out_path = pairs_output_path(dataset_dir)

    if out_path.exists():
        print(f"[SKIP] pairs already exists: {out_path}")
        return out_path

    cmd = [sys.executable, str(PAIRS_SCRIPT), str(dataset_dir), "--pairs-out", str(out_path)]
    run_subprocess(cmd, f"pairs.py -> {out_path}")
    return out_path


def maybe_run_generate(dataset_dir: Path, mod_type: str, model: ModelSpec, throttle_enabled: bool = False) -> Path:
    model_name = model["name"]
    script_path = model["generate_script"]
    out_path = embeddings_output_path(mod_type, model_name)

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


def maybe_run_evaluate(mod_type: str, model: ModelSpec, pairs_file: Path, embeddings_file: Path) -> Path:
    model_name = model["name"]
    script_path = model["evaluate_script"]
    out_path = metrics_output_path(mod_type, model_name)

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



def collect_metrics_files() -> List[Path]:
    if not RESULTS_ROOT.exists():
        return []
    return sorted(RESULTS_ROOT.glob("*/*/metrics.json"))


def rebuild_compiled_csv() -> None:
    ensure_dir(RESULTS_ROOT)

    metrics_files = collect_metrics_files()

    fieldnames = [
        "modification_type",
        "model",
        "num_embeddings",
        "num_identities",
        "num_pairs",
        "num_genuine_pairs",
        "num_impostor_pairs",
        "best_threshold",
        "best_accuracy",
        "far",
        "frr",
        "precision",
        "recall",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "eer",
        "eer_threshold",
        "far_at_eer",
        "frr_at_eer",
    ]

    rows = []

    for path in metrics_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows.append({
            "modification_type": path.parent.parent.name,
            "model": path.parent.name,            **data,
        })

    rows.sort(key=lambda r: (r["modification_type"], r["model"]))

    with open(COMPILED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Compiled CSV refreshed: {COMPILED_CSV}")



def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <dataset_dir> [--throttle]")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1]).resolve()
    validate_input_dataset(dataset_dir)
    validate_model_registry(MODELS)

    throttle_enabled = "--throttle" in sys.argv
    
    if throttle_enabled:
        delays_str = ", ".join([f"{k}={v}s" for k, v in MODEL_THROTTLE_DELAYS.items()])
        print(f"[INFO] Throttling enabled: batch_size={THROTTLE_BATCH_SIZE}, delays=[{delays_str}]")

    mod_type = get_modification_type(dataset_dir)

    ensure_dir(PAIRS_ROOT)
    ensure_dir(RESULTS_ROOT)

    pairs_file = maybe_run_pairs(dataset_dir)

    for model in MODELS:
        print(f"\n===== MODEL: {model['name']} =====")

        emb_file = maybe_run_generate(dataset_dir, mod_type, model, throttle_enabled)
        maybe_run_evaluate(mod_type, model, pairs_file, emb_file)

    rebuild_compiled_csv()

    print("\n[INFO] Workflow complete.")


if __name__ == "__main__":
    main()
