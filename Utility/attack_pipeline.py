#!/usr/bin/env python3
# gallery_dir + attack config -> cached attack pairs, probe materialization, probe metrics paths and subprocess runs

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from Utility.pathfinder import resolve_dataset_context
from Utility.pipeline_common import MODELS, RESULTS_ROOT
from Utility.pipeline_common import maybe_run_generate, run_subprocess, validate_model_registry


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATTACK_PAIR_SCRIPT = PROJECT_ROOT / "Utility" / "attack_pair.py"
ATTACK_EVAL_SCRIPT = PROJECT_ROOT / "Evaluators" / "attack.py"
MODIFIER_SCRIPT = PROJECT_ROOT / "modifier.py"
ATTACK_METADATA_FILENAME = "attack_metadata.json"


def attack_pairs_output(model_results_dir: Path) -> tuple[Path, Path]:
    variant_name = model_results_dir.parent.name
    model_name = model_results_dir.name
    prefix = f"{variant_name}_{model_name}_atkpairs"
    return model_results_dir / f"{prefix}.npz", model_results_dir / f"{prefix}.json"


def derive_probe_dir(gallery_dir: Path, attack_method: str) -> Path:
    context = resolve_dataset_context(gallery_dir)
    variant_name = f"{attack_method}_{context.base_root_name}"
    return context.dataset_root / context.base_dataset_name / variant_name


def ensure_gallery_embeddings(gallery_dir: Path, throttle_enabled: bool) -> dict[str, Path]:
    validate_model_registry(MODELS)
    outputs: dict[str, Path] = {}
    gallery_context = resolve_dataset_context(gallery_dir)
    for model in MODELS:
        outputs[model["name"]] = maybe_run_generate(
            gallery_dir,
            gallery_context.variant_name,
            model,
            throttle_enabled=throttle_enabled,
        )
    return outputs


def maybe_run_attack_pair(
    model_name: str,
    gallery_embeddings: Path,
    top_k: int,
    samples_per_identity_pair: int,
    pairing_mode: str,
    min_identity_sim: Optional[float],
    min_image_sim: Optional[float],
) -> tuple[Path, Path]:
    model_results_dir = gallery_embeddings.parent
    out_npz, out_json = attack_pairs_output(model_results_dir)
    if out_npz.exists() and out_json.exists():
        print(f"[SKIP] {model_name} attack pairs exist")
        return out_npz, out_json

    cmd = [
        sys.executable,
        str(ATTACK_PAIR_SCRIPT),
        str(model_results_dir),
        "--top-k",
        str(top_k),
        "--samples-per-identity-pair",
        str(samples_per_identity_pair),
        "--pairing-mode",
        pairing_mode,
    ]
    if min_identity_sim is not None:
        cmd.extend(["--min-identity-sim", str(min_identity_sim)])
    if min_image_sim is not None:
        cmd.extend(["--min-image-sim", str(min_image_sim)])

    run_subprocess(cmd, f"attack_pair.py -> {out_npz}")
    return out_npz, out_json


def maybe_run_attack_generate(
    gallery_dir: Path,
    probe_dir: Path,
    attack_pairs_npz: Path,
    attack_method: str,
    attack_generator_script: Optional[Path],
) -> Path:
    metadata_path = probe_dir / ATTACK_METADATA_FILENAME
    if probe_dir.exists() and metadata_path.exists():
        print(f"[SKIP] attack probe dataset exists: {probe_dir}")
        return metadata_path

    if attack_generator_script is None:
        raise FileNotFoundError(
            "probe dataset / attack metadata missing and no --attack-generator-script was provided"
        )

    cmd = [
        sys.executable,
        str(MODIFIER_SCRIPT),
        str(gallery_dir),
        "--step",
        attack_method,
        "--pair-input",
        str(attack_pairs_npz),
        "--generator-script",
        str(attack_generator_script),
        "--output-dir",
        str(probe_dir),
    ]
    run_subprocess(cmd, f"modifier.py attack -> {probe_dir}")
    return metadata_path


def attack_metrics_output(probe_variant_name: str, model_name: str) -> Path:
    return RESULTS_ROOT / probe_variant_name / model_name / "metrics.json"


def maybe_run_attack_evaluate(
    gallery_embeddings: Path,
    probe_embeddings: Path,
    attack_metadata: Path,
    metrics_out: Path,
    pairs_file: Path,
) -> Path:
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    if metrics_out.exists():
        print(f"[SKIP] {metrics_out.parent.name} attack metrics exist")
        return metrics_out

    cmd = [
        sys.executable,
        str(ATTACK_EVAL_SCRIPT),
        str(gallery_embeddings),
        str(probe_embeddings),
        str(attack_metadata),
        str(metrics_out),
        "--pairs-file",
        str(pairs_file),
    ]
    run_subprocess(cmd, f"attack.py -> {metrics_out}")
    return metrics_out
