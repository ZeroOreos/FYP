#!/usr/bin/env python3
# python3 main_attack.py <gallery_dir> [attack options] -> cached attack pairing, probe materialization, embeddings, attack metrics, parsed results

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Utility.paths import derive_pairs_output_path, resolve_dataset_context
from Utility.results_compile import rebuild_compiled_csv, rebuild_parsed_results
from Utility.runtime import MODELS, PAIRS_ROOT, PROJECT_ROOT, RESULTS_ROOT
from Utility.runtime import ensure_dir, run_subprocess, validate_input_dataset, validate_model_registry


PAIR_SCRIPT = PROJECT_ROOT / "Attack" / "pairs.py"
ATTACK_EVAL_SCRIPT = PROJECT_ROOT / "Attack" / "evaluate.py"
MODIFIER_SCRIPT = PROJECT_ROOT / "modifier.py"
ATTACK_METADATA_FILENAME = "attack_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached attack-verification orchestration.")
    parser.add_argument("gallery_dir", type=Path, help="Clean gallery dataset root.")

    probe_group = parser.add_mutually_exclusive_group(required=True)
    probe_group.add_argument("--probe-dir", type=Path, help="Existing attack probe dataset root.")
    probe_group.add_argument("--attack-method", type=str, help="Attack method token used to derive the probe root.")

    parser.add_argument("--attack-generator-script", type=Path, default=None, help="Optional generator backend script.")
    parser.add_argument("--pair-model", type=str, default="InsightFace", help="Model used to generate attack pairs.")
    parser.add_argument("--top-k", type=int, default=5, help="Nearest non-match identities kept per victim.")
    parser.add_argument("--samples-per-identity-pair", type=int, default=3, help="Specific source-target image pairs kept.")
    parser.add_argument("--pairing-mode", choices=("hard", "semi_hard"), default="hard")
    parser.add_argument("--min-identity-sim", type=float, default=None)
    parser.add_argument("--min-image-sim", type=float, default=None)

    args = parser.parse_args()
    args.gallery_dir = args.gallery_dir.resolve()
    if args.probe_dir is not None:
        args.probe_dir = args.probe_dir.resolve()
    if args.attack_generator_script is not None:
        args.attack_generator_script = args.attack_generator_script.resolve()
    return args


def embeddings_output_path(variant_name: str, model_name: str) -> Path:
    return RESULTS_ROOT / variant_name / model_name / "embeddings.npz"


def attack_metrics_output(variant_name: str, model_name: str) -> Path:
    return RESULTS_ROOT / variant_name / model_name / "metrics.json"


def maybe_run_pairs(dataset_dir: Path) -> Path:
    pairs_file = derive_pairs_output_path(dataset_dir)
    ensure_dir(PAIRS_ROOT)
    if pairs_file.exists():
        return pairs_file

    run_subprocess(
        [sys.executable, str(PROJECT_ROOT / "Recognition" / "pairs.py"), str(dataset_dir), "--pairs-out", str(pairs_file)],
        f"pairs.py -> {pairs_file}",
    )
    return pairs_file


def maybe_run_generate(dataset_dir: Path, variant_name: str, model: dict[str, Path]) -> Path:
    embeddings_file = embeddings_output_path(variant_name, model["name"])
    ensure_dir(embeddings_file.parent)
    if embeddings_file.exists():
        print(f"[SKIP] {model['name']} embeddings exist")
        return embeddings_file

    run_subprocess(
        [sys.executable, str(model["generate_script"]), str(dataset_dir), str(embeddings_file)],
        f"{model['name']} generate -> {embeddings_file}",
    )
    return embeddings_file


def ensure_gallery_embeddings(gallery_dir: Path) -> dict[str, Path]:
    gallery_context = resolve_dataset_context(gallery_dir)
    return {
        model["name"]: maybe_run_generate(gallery_dir, gallery_context.variant_name, model)
        for model in MODELS
    }


def attack_pairs_output(model_results_dir: Path) -> tuple[Path, Path]:
    variant_name = model_results_dir.parent.name
    model_name = model_results_dir.name
    prefix = f"{variant_name}_{model_name}_atkpairs"
    return model_results_dir / f"{prefix}.npz", model_results_dir / f"{prefix}.json"


def maybe_run_attack_pair(args: argparse.Namespace, gallery_embeddings: Path) -> tuple[Path, Path]:
    out_npz, out_json = attack_pairs_output(gallery_embeddings.parent)
    if out_npz.exists() and out_json.exists():
        print(f"[SKIP] {args.pair_model} attack pairs exist")
        return out_npz, out_json

    cmd = [
        sys.executable,
        str(PAIR_SCRIPT),
        str(gallery_embeddings.parent),
        "--top-k",
        str(max(1, args.top_k)),
        "--samples-per-identity-pair",
        str(max(1, args.samples_per_identity_pair)),
        "--pairing-mode",
        args.pairing_mode,
    ]
    if args.min_identity_sim is not None:
        cmd.extend(["--min-identity-sim", str(args.min_identity_sim)])
    if args.min_image_sim is not None:
        cmd.extend(["--min-image-sim", str(args.min_image_sim)])

    run_subprocess(cmd, f"attack pair -> {out_npz}")
    return out_npz, out_json


def derive_probe_dir(gallery_dir: Path, attack_method: str) -> Path:
    context = resolve_dataset_context(gallery_dir)
    return context.dataset_root / context.base_dataset_name / f"{attack_method}_{context.base_root_name}"


def maybe_materialize_attack_probe(
    gallery_dir: Path,
    probe_dir: Path,
    attack_pairs_npz: Path,
    attack_method: str,
    generator_script: Path | None,
) -> Path:
    metadata_path = probe_dir / ATTACK_METADATA_FILENAME
    if probe_dir.exists() and metadata_path.exists():
        print(f"[SKIP] attack probe dataset exists: {probe_dir}")
        return metadata_path
    if generator_script is None:
        raise FileNotFoundError("probe dataset / attack metadata missing and no --attack-generator-script was provided")

    run_subprocess(
        [
            sys.executable,
            str(MODIFIER_SCRIPT),
            str(gallery_dir),
            "--step",
            attack_method,
            "--pair-input",
            str(attack_pairs_npz),
            "--generator-script",
            str(generator_script),
            "--output-dir",
            str(probe_dir),
        ],
        f"attack materialize -> {probe_dir}",
    )
    return metadata_path


def maybe_run_attack_evaluate(
    model_name: str,
    gallery_embeddings: Path,
    probe_embeddings: Path,
    attack_metadata: Path,
    pairs_file: Path,
) -> Path:
    metrics_file = attack_metrics_output(probe_embeddings.parent.parent.name, model_name)
    ensure_dir(metrics_file.parent)
    if metrics_file.exists():
        print(f"[SKIP] {model_name} attack metrics exist")
        return metrics_file

    run_subprocess(
        [
            sys.executable,
            str(ATTACK_EVAL_SCRIPT),
            str(gallery_embeddings),
            str(probe_embeddings),
            str(attack_metadata),
            str(metrics_file),
            "--pairs-file",
            str(pairs_file),
        ],
        f"attack evaluate -> {metrics_file}",
    )
    return metrics_file


def main() -> None:
    args = parse_args()
    validate_input_dataset(args.gallery_dir)
    validate_model_registry(MODELS)

    ensure_dir(PAIRS_ROOT)
    ensure_dir(RESULTS_ROOT)

    pairs_file = maybe_run_pairs(args.gallery_dir)
    gallery_embeddings = ensure_gallery_embeddings(args.gallery_dir)
    if args.pair_model not in gallery_embeddings:
        raise ValueError(f"pair model not found in registry: {args.pair_model}")

    attack_pairs_npz, _ = maybe_run_attack_pair(args, gallery_embeddings[args.pair_model])
    probe_dir = args.probe_dir or derive_probe_dir(args.gallery_dir, args.attack_method)
    attack_method = args.attack_method or probe_dir.name.rsplit("_", 1)[0]
    attack_metadata = maybe_materialize_attack_probe(
        args.gallery_dir,
        probe_dir,
        attack_pairs_npz,
        attack_method,
        args.attack_generator_script,
    )

    probe_context = resolve_dataset_context(probe_dir)
    for model in MODELS:
        probe_embeddings = maybe_run_generate(probe_dir, probe_context.variant_name, model)
        maybe_run_attack_evaluate(
            model["name"],
            gallery_embeddings[model["name"]],
            probe_embeddings,
            attack_metadata,
            pairs_file,
        )

    rebuild_compiled_csv()
    rebuild_parsed_results()
    print("\n[INFO] Attack workflow complete.")


if __name__ == "__main__":
    main()
