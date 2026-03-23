#!/usr/bin/env python3
"""
main.py

Orchestrator for the FYP verification workflow.

Stages:
1) pairs.py -> Dataset/pairs/<derived_base_name>_pairs.npz
2) Models/<Model>/generate.py -> Results/<variant_name>/<Model>/embeddings.npz
3) Models/<Model>/evaluate.py -> Results/<variant_name>/<Model>/metrics.json
4) Rebuild Results/compiled_results.csv from all available metrics files
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from pathfinder import resolve_dataset_context


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
PAIRS_ROOT = DATASET_ROOT / "pairs"
RESULTS_ROOT = PROJECT_ROOT / "Results"
COMPILED_CSV = RESULTS_ROOT / "compiled_results.csv"
PAIRS_SCRIPT = PROJECT_ROOT / "pairs.py"

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


def maybe_run_generate(dataset_dir: Path, variant_name: str, model: ModelSpec, throttle_enabled: bool = False) -> Path:
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


def collect_metrics_files() -> List[Path]:
    if not RESULTS_ROOT.exists():
        return []
    return sorted(RESULTS_ROOT.glob("*/*/metrics.json"))


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _variant_metadata_from_results_path(metrics_path: Path, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
    variant_name = metrics_path.parent.parent.name
    pair_file = Path(metrics_data.get("pair_file", "")) if metrics_data.get("pair_file") else None
    referenced_base_root = variant_name.rsplit("_", 1)[-1] if "_" in variant_name else variant_name

    transform_tokens: List[str] = []
    if variant_name != referenced_base_root:
        suffix = f"__{referenced_base_root}"
        if variant_name.endswith(suffix):
            prefix = variant_name[: -len(suffix)]
            transform_tokens = [token for token in prefix.split("__") if token]
        else:
            suffix = f"_{referenced_base_root}"
            prefix = variant_name[: -len(suffix)] if variant_name.endswith(suffix) else variant_name
            transform_tokens = [prefix] if prefix else []

    if variant_name == referenced_base_root:
        variant_type = "clean"
    elif len(transform_tokens) <= 1:
        variant_type = "single"
    else:
        variant_type = "hybrid"

    base_dataset = ""
    inferred_pair_file = ""
    if pair_file is not None and pair_file.name.endswith("_pairs.npz"):
        stem = pair_file.name[: -len("_pairs.npz")]
        stem_parts = stem.split("_")
        if len(stem_parts) >= 2:
            base_dataset = stem_parts[0]
            inferred_pair_file = str(pair_file)
            referenced_base_root = stem_parts[-1]

    return {
        "base_dataset": base_dataset,
        "referenced_base_root": referenced_base_root,
        "variant_name": variant_name,
        "variant_type": variant_type,
        "num_transforms": len(transform_tokens),
        "transform_chain": "__".join(transform_tokens) if transform_tokens else "clean",
        "pair_file": inferred_pair_file,
    }


def _flatten_tar_at_far(pooled_metrics: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, entry in pooled_metrics.get("tar_at_far", {}).items():
        normalized = key.replace("@", "_").replace("=", "_").replace(".", "p")
        flattened[f"{normalized}_tar"] = entry.get("tar")
        flattened[f"{normalized}_threshold"] = entry.get("threshold")
        flattened[f"{normalized}_actual_far"] = entry.get("actual_far")
    return flattened


def rebuild_compiled_csv() -> None:
    ensure_dir(RESULTS_ROOT)

    metrics_files = collect_metrics_files()
    rows: List[Dict[str, Any]] = []

    base_fieldnames = [
        "base_dataset",
        "referenced_base_root",
        "variant_name",
        "variant_type",
        "num_transforms",
        "transform_chain",
        "model",
        "pair_file",
        "embeddings_file",
        "num_embeddings",
        "embedding_dim",
        "num_pairs",
        "num_genuine_pairs",
        "num_impostor_pairs",
        "best_threshold",
        "accuracy",
        "auc",
        "eer",
        "far",
        "frr",
        "tar",
        "precision",
        "recall",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "eer_threshold",
        "far_at_eer",
        "frr_at_eer",
        "runtime_seconds",
    ]
    extra_fieldnames: List[str] = []

    for path in metrics_files:
        data = _load_json(path)
        pooled = data.get("pooled_metrics", {})
        variant_meta = _variant_metadata_from_results_path(path, data)
        tar_fields = _flatten_tar_at_far(pooled)
        extra_fieldnames.extend([name for name in tar_fields if name not in extra_fieldnames])

        row = {
            **variant_meta,
            "model": data.get("model", path.parent.name),
            "pair_file": data.get("pair_file") or variant_meta["pair_file"],
            "embeddings_file": data.get("embeddings_file", ""),
            "num_embeddings": data.get("num_embeddings"),
            "embedding_dim": data.get("embedding_dim"),
            "num_pairs": data.get("num_pairs", pooled.get("num_pairs")),
            "num_genuine_pairs": data.get("num_genuine_pairs", pooled.get("num_genuine_pairs")),
            "num_impostor_pairs": data.get("num_impostor_pairs", pooled.get("num_impostor_pairs")),
            "best_threshold": pooled.get("best_threshold"),
            "accuracy": pooled.get("best_accuracy"),
            "auc": pooled.get("auc"),
            "eer": pooled.get("eer"),
            "far": pooled.get("far"),
            "frr": pooled.get("frr"),
            "tar": pooled.get("tar"),
            "precision": pooled.get("precision"),
            "recall": pooled.get("recall"),
            "f1": pooled.get("f1"),
            "tp": pooled.get("tp"),
            "tn": pooled.get("tn"),
            "fp": pooled.get("fp"),
            "fn": pooled.get("fn"),
            "eer_threshold": pooled.get("eer_threshold"),
            "far_at_eer": pooled.get("far_at_eer"),
            "frr_at_eer": pooled.get("frr_at_eer"),
            "runtime_seconds": data.get("runtime_seconds"),
            **tar_fields,
        }
        rows.append(row)

    fieldnames = base_fieldnames + extra_fieldnames
    rows.sort(key=lambda r: (r["base_dataset"], r["variant_name"], r["model"]))

    with open(COMPILED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Compiled CSV refreshed: {COMPILED_CSV}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <dataset_dir> [--throttle]")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1]).resolve()
    validate_input_dataset(dataset_dir)
    validate_model_registry(MODELS)

    throttle_enabled = "--throttle" in sys.argv
    context = resolve_dataset_context(dataset_dir)

    if throttle_enabled:
        delays_str = ", ".join([f"{k}={v}s" for k, v in MODEL_THROTTLE_DELAYS.items()])
        print(f"[INFO] Throttling enabled: batch_size={THROTTLE_BATCH_SIZE}, delays=[{delays_str}]")

    print(f"[INFO] Variant: {context.variant_name}")
    print(f"[INFO] Referenced base root: {context.base_root_name}")
    print(f"[INFO] Shared pair file: {pairs_output_path(dataset_dir)}")

    ensure_dir(PAIRS_ROOT)
    ensure_dir(RESULTS_ROOT)

    pairs_file = maybe_run_pairs(dataset_dir)

    for model in MODELS:
        print(f"\n===== MODEL: {model['name']} =====")
        emb_file = maybe_run_generate(dataset_dir, context.variant_name, model, throttle_enabled)
        maybe_run_evaluate(context.variant_name, model, pairs_file, emb_file)

    rebuild_compiled_csv()
    print("\n[INFO] Workflow complete.")


if __name__ == "__main__":
    main()
