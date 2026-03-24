#!/usr/bin/env python3
# python3 main_recognition.py <dataset_dir> [--throttle] -> cached recognition orchestration and result refresh

import sys
from pathlib import Path

from Utility.pathfinder import resolve_dataset_context
from Utility.pipeline_common import MODELS, MODEL_THROTTLE_DELAYS, PAIRS_ROOT, RESULTS_ROOT, THROTTLE_BATCH_SIZE
from Utility.pipeline_common import ensure_dir, maybe_run_evaluate, maybe_run_generate, maybe_run_pairs
from Utility.pipeline_common import pairs_output_path, validate_input_dataset, validate_model_registry
from Utility.results_compile import rebuild_compiled_csv, rebuild_parsed_results


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 main_recognition.py <dataset_dir> [--throttle]")
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
    rebuild_parsed_results()
    print("\n[INFO] Workflow complete.")


if __name__ == "__main__":
    main()
