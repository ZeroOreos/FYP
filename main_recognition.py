#!/usr/bin/env python3
# python3 main_recognition.py <dataset_dir> -> recognition pipeline

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Utility.paths import derive_pairs_output_path, resolve_dataset_context
from Utility.results_compile import rebuild_compiled_csv, rebuild_parsed_results
from Utility.runtime import ALLOW_MISSING_PAIRS, MODELS, PAIRS_ROOT, PROJECT_ROOT, RESULTS_ROOT
from Utility.runtime import DEFAULT_ONNX_PROVIDER, DEFAULT_TORCH_DEVICE
from Utility.runtime import THROTTLE_BATCH_SIZE, MODEL_THROTTLE_DELAYS
from Utility.runtime import ensure_dir, run_subprocess, runtime_env_overrides, validate_input_dataset, validate_model_registry


PAIRS_SCRIPT = PROJECT_ROOT / "Recognition" / "pairs.py"
VERIFY_SCRIPT = PROJECT_ROOT / "Recognition" / "evaluate.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recognition pipeline.")
    parser.add_argument("dataset_dir", type=Path, help="Dataset root.")
    parser.add_argument("--torch-device", choices=("auto", "cuda", "mps", "cpu"), default=DEFAULT_TORCH_DEVICE)
    parser.add_argument("--onnx-provider", choices=("auto", "coreml", "cuda", "cpu"), default=DEFAULT_ONNX_PROVIDER)
    args = parser.parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    return args


def embeddings_output_path(variant_name: str, model_name: str) -> Path:
    return RESULTS_ROOT / variant_name / model_name / "embeddings.npz"


def metrics_output_path(variant_name: str, model_name: str) -> Path:
    return RESULTS_ROOT / variant_name / model_name / "metrics.json"


def maybe_run_pairs(dataset_dir: Path, env_overrides: dict[str, str]) -> Path:
    pairs_file = derive_pairs_output_path(dataset_dir)
    ensure_dir(PAIRS_ROOT)
    if pairs_file.exists():
        return pairs_file

    run_subprocess(
        [sys.executable, str(PAIRS_SCRIPT), str(dataset_dir), "--pairs-out", str(pairs_file)],
        f"pairs.py -> {pairs_file}",
        extra_env=env_overrides,
    )
    return pairs_file


def maybe_run_generate(
    dataset_dir: Path,
    variant_name: str,
    model: dict[str, Path],
    env_overrides: dict[str, str],
    throttle: bool = False,
) -> Path:
    embeddings_file = embeddings_output_path(variant_name, model["name"])
    ensure_dir(embeddings_file.parent)
    if embeddings_file.exists():
        print(f"[SKIP] {model['name']} embeddings exist")
        return embeddings_file

    cmd = [sys.executable, str(model["generate_script"]), str(dataset_dir), str(embeddings_file)]
    if throttle:
        delay = MODEL_THROTTLE_DELAYS.get(model["name"], 0.0)
        if model["name"] == "FaceNet":
            cmd.extend([str(THROTTLE_BATCH_SIZE), str(delay)])
        elif model["name"] == "InsightFace":
            cmd.append(str(delay))

    run_subprocess(cmd, f"{model['name']} generate -> {embeddings_file}", extra_env=env_overrides)
    return embeddings_file


def maybe_run_evaluate(
    variant_name: str,
    model: dict[str, Path],
    pairs_file: Path,
    embeddings_file: Path,
    env_overrides: dict[str, str],
) -> Path:
    metrics_file = metrics_output_path(variant_name, model["name"])
    ensure_dir(metrics_file.parent)
    if metrics_file.exists():
        print(f"[SKIP] {model['name']} metrics exist")
        return metrics_file

    cmd = [
        sys.executable,
        str(VERIFY_SCRIPT),
        str(pairs_file),
        str(embeddings_file),
        str(metrics_file),
        "--model-name",
        model["name"],
    ]
    if ALLOW_MISSING_PAIRS:
        cmd.append("--allow-missing-pairs")

    run_subprocess(cmd, f"{model['name']} evaluate -> {metrics_file}", extra_env=env_overrides)
    return metrics_file


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    validate_input_dataset(dataset_dir)
    validate_model_registry(MODELS)

    context = resolve_dataset_context(dataset_dir)
    env_overrides = runtime_env_overrides(
        torch_device=args.torch_device,
        onnx_provider=args.onnx_provider,
    )
    print(f"[INFO] Variant: {context.variant_name}")
    print(f"[INFO] Referenced base root: {context.base_root_name}")
    print(f"[INFO] Shared pair file: {derive_pairs_output_path(dataset_dir)}")
    print(f"[INFO] Torch device preference: {args.torch_device}")
    print(f"[INFO] ONNX provider preference: {args.onnx_provider}")

    ensure_dir(PAIRS_ROOT)
    ensure_dir(RESULTS_ROOT)

    pairs_file = maybe_run_pairs(dataset_dir, env_overrides)
    for model in MODELS:
        print(f"\n===== MODEL: {model['name']} =====")
        embeddings_file = maybe_run_generate(dataset_dir, context.variant_name, model, env_overrides)
        maybe_run_evaluate(context.variant_name, model, pairs_file, embeddings_file, env_overrides)

    rebuild_compiled_csv()
    rebuild_parsed_results()
    print("\n[INFO] Workflow complete.")


if __name__ == "__main__":
    main()
