#!/usr/bin/env python3
# python3 main_attack.py <gallery_dir> [attack options] -> cached attack pairing, probe generation, embeddings, attack metrics, parsed results

from __future__ import annotations

import argparse
from pathlib import Path

from Utility.pathfinder import resolve_dataset_context
from Utility.pipeline_common import MODELS, PAIRS_ROOT, RESULTS_ROOT
from Utility.pipeline_common import ensure_dir, maybe_run_generate, maybe_run_pairs
from Utility.pipeline_common import validate_input_dataset, validate_model_registry
from Utility.attack_pipeline import derive_probe_dir, ensure_gallery_embeddings
from Utility.attack_pipeline import maybe_run_attack_evaluate, maybe_run_attack_generate, maybe_run_attack_pair
from Utility.attack_pipeline import attack_metrics_output
from Utility.results_compile import rebuild_compiled_csv, rebuild_parsed_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached attack-verification orchestration.")
    parser.add_argument("gallery_dir", type=str, help="Clean gallery dataset root.")
    parser.add_argument("--probe-dir", type=str, default=None, help="Existing attack probe dataset root.")
    parser.add_argument("--attack-method", type=str, default=None, help="Attack method token used to derive the probe root.")
    parser.add_argument(
        "--attack-generator-script",
        type=str,
        default=None,
        help="Optional generator script with contract: <gallery_dir> <atkpairs_npz> <probe_dir> <manifest_out>.",
    )
    parser.add_argument("--pair-model", type=str, default="InsightFace", help="Model used to generate attack pairs.")
    parser.add_argument("--top-k", type=int, default=5, help="Nearest non-match identities kept per victim.")
    parser.add_argument(
        "--samples-per-identity-pair",
        type=int,
        default=3,
        help="Specific source-target image pairs kept per attacker-victim identity pair.",
    )
    parser.add_argument("--pairing-mode", choices=("hard", "semi_hard"), default="hard")
    parser.add_argument("--min-identity-sim", type=float, default=None)
    parser.add_argument("--min-image-sim", type=float, default=None)
    parser.add_argument("--throttle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gallery_dir = Path(args.gallery_dir).resolve()
    validate_input_dataset(gallery_dir)
    validate_model_registry(MODELS)
    ensure_dir(RESULTS_ROOT)
    ensure_dir(PAIRS_ROOT)

    throttle_enabled = args.throttle
    pairs_file = maybe_run_pairs(gallery_dir)
    gallery_embeddings = ensure_gallery_embeddings(gallery_dir, throttle_enabled)

    pair_model = args.pair_model
    if pair_model not in gallery_embeddings:
        raise ValueError(f"pair model not found in registry: {pair_model}")
    attack_pairs_npz, _ = maybe_run_attack_pair(
        pair_model,
        gallery_embeddings[pair_model],
        top_k=max(1, args.top_k),
        samples_per_identity_pair=max(1, args.samples_per_identity_pair),
        pairing_mode=args.pairing_mode,
        min_identity_sim=args.min_identity_sim,
        min_image_sim=args.min_image_sim,
    )

    if args.probe_dir:
        probe_dir = Path(args.probe_dir).resolve()
    elif args.attack_method:
        probe_dir = derive_probe_dir(gallery_dir, args.attack_method)
    else:
        raise ValueError("provide --probe-dir or --attack-method")

    generator_script = Path(args.attack_generator_script).resolve() if args.attack_generator_script else None
    attack_manifest = maybe_run_attack_generate(gallery_dir, probe_dir, attack_pairs_npz, generator_script)
    probe_context = resolve_dataset_context(probe_dir)

    for model in MODELS:
        model_name = model["name"]
        probe_embeddings = maybe_run_generate(
            probe_dir,
            probe_context.variant_name,
            model,
            throttle_enabled=throttle_enabled,
        )
        maybe_run_attack_evaluate(
            gallery_embeddings=gallery_embeddings[model_name],
            probe_embeddings=probe_embeddings,
            attack_manifest=attack_manifest,
            metrics_out=attack_metrics_output(probe_context.variant_name, model_name),
            pairs_file=pairs_file,
        )

    rebuild_compiled_csv()
    rebuild_parsed_results()
    print("\n[INFO] Attack workflow complete.")


if __name__ == "__main__":
    main()
