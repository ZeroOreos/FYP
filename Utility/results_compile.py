#!/usr/bin/env python3
# Results/*/*/metrics.json + Dataset/*/*/transform.json -> compiled_results.csv and parsed result tables

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

from Utility.paths import resolve_dataset_context
from Utility.runtime import ensure_dir, run_subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
RESULTS_ROOT = PROJECT_ROOT / "Results"
COMPILED_CSV = RESULTS_ROOT / "compiled_results.csv"
RESULTS_PARSE_SCRIPT = PROJECT_ROOT / "Utility" / "results_parse.py"


def collect_metrics_files() -> list[Path]:
    if not RESULTS_ROOT.exists():
        return []
    return sorted(RESULTS_ROOT.glob("*/*/metrics.json"))


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _variant_metadata_from_results_path(metrics_path: Path, metrics_data: dict[str, Any]) -> dict[str, Any]:
    variant_name = metrics_path.parent.parent.name
    pair_file = Path(metrics_data.get("pair_file", "")) if metrics_data.get("pair_file") else None
    dataset_dir = next(iter(sorted(DATASET_ROOT.glob(f"*/{variant_name}"))), None)
    if dataset_dir is not None:
        context = resolve_dataset_context(dataset_dir)
        return {
            "base_dataset": context.base_dataset_name,
            "referenced_base_root": context.base_root_name,
            "variant_name": context.variant_name,
            "variant_type": context.variant_type,
            "num_transforms": context.num_transforms,
            "transform_chain": context.transform_chain,
            "pair_file": str(pair_file) if pair_file is not None else "",
        }

    referenced_base_root = variant_name.rsplit("_", 1)[-1] if "_" in variant_name else variant_name
    transform_chain = "clean"
    if variant_name != referenced_base_root:
        prefix = variant_name[: -(len(referenced_base_root) + 1)] if variant_name.endswith(f"_{referenced_base_root}") else variant_name
        transform_chain = prefix if prefix else "clean"
    transform_tokens = [token for token in transform_chain.split("__") if token] if transform_chain != "clean" else []
    variant_type = "clean" if transform_chain == "clean" else ("single" if len(transform_tokens) == 1 else "hybrid")
    base_dataset = ""
    inferred_pair_file = str(pair_file) if pair_file is not None else ""
    if pair_file is not None and pair_file.name.endswith("_pairs.npz"):
        stem_parts = pair_file.name[: -len("_pairs.npz")].split("_")
        if len(stem_parts) >= 2:
            base_dataset = stem_parts[0]
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


def _transform_metadata_path(variant_meta: dict[str, Any]) -> Optional[Path]:
    base_dataset = variant_meta.get("base_dataset", "")
    variant_name = variant_meta.get("variant_name", "")
    if not base_dataset or not variant_name:
        return None

    candidate = DATASET_ROOT / base_dataset / variant_name / "transform.json"
    if candidate.exists():
        return candidate
    attack_candidate = DATASET_ROOT / base_dataset / variant_name / "attack_metadata.json"
    if attack_candidate.exists():
        return attack_candidate
    return None


def _load_transform_metadata(variant_meta: dict[str, Any]) -> dict[str, Any]:
    metadata_path = _transform_metadata_path(variant_meta)
    if metadata_path is None:
        return {}
    return _load_json(metadata_path)


def _flatten_transform_metadata(transform_meta: dict[str, Any]) -> dict[str, Any]:
    if not transform_meta:
        return {
            "transform_metadata_file": "",
            "pipeline_summary": "",
            "pipeline_json": "",
            "latest_transform": "",
            "oldest_transform": "",
            "pipeline_seed": "",
        }

    pipeline = transform_meta.get("pipeline", [])
    if not pipeline:
        metadata_file = ""
        output_dir = transform_meta.get("output_dataset_dir", "")
        if output_dir:
            if transform_meta.get("attack_method"):
                metadata_file = str(Path(output_dir) / "attack_metadata.json")
            else:
                metadata_file = str(Path(output_dir) / "transform.json")
        return {
            "transform_metadata_file": transform_meta.get("transform_metadata_file") or metadata_file,
            "pipeline_summary": transform_meta.get("attack_method", ""),
            "pipeline_json": "",
            "latest_transform": transform_meta.get("attack_method", ""),
            "oldest_transform": transform_meta.get("attack_method", ""),
            "pipeline_seed": transform_meta.get("seed", ""),
        }

    tokens = [step.get("token", "") for step in pipeline]
    flattened = {
        "transform_metadata_file": transform_meta.get("transform_metadata_file")
        or (
            str(Path(transform_meta.get("output_dataset_dir", "")) / "transform.json")
            if transform_meta.get("output_dataset_dir")
            else ""
        ),
        "pipeline_summary": " -> ".join(tokens),
        "pipeline_json": json.dumps(pipeline, separators=(",", ":")),
        "latest_transform": tokens[0] if tokens else "",
        "oldest_transform": tokens[-1] if tokens else "",
        "pipeline_seed": transform_meta.get("seed", ""),
    }

    for idx, step in enumerate(pipeline, start=1):
        flattened[f"step_{idx}_token"] = step.get("token", "")
        flattened[f"step_{idx}_modifier"] = step.get("modifier", "")
        flattened[f"step_{idx}_params"] = json.dumps(step.get("params", {}), separators=(",", ":"))

    return flattened


def _primary_metrics(data: dict[str, Any]) -> dict[str, Any]:
    pooled = data.get("pooled_metrics")
    if isinstance(pooled, dict) and pooled:
        return pooled
    return data


def _flatten_tar_at_far(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, entry in metrics.get("tar_at_far", {}).items():
        normalized = key.replace("@", "_").replace("=", "_").replace(".", "p")
        flattened[f"{normalized}_tar"] = entry.get("tar")
        flattened[f"{normalized}_threshold"] = entry.get("threshold")
        flattened[f"{normalized}_actual_far"] = entry.get("actual_far")

        target_far = entry.get("target_far")
        if target_far is not None:
            compact_far = str(target_far).replace(".", "p")
            flattened[f"tar_far_{compact_far}"] = entry.get("tar")
            flattened[f"tar_far_{compact_far}_threshold"] = entry.get("threshold")
            flattened[f"tar_far_{compact_far}_actual_far"] = entry.get("actual_far")
    return flattened


def _flatten_reproducibility(data: dict[str, Any]) -> dict[str, Any]:
    repro = data.get("reproducibility", {})
    return {
        "dataset_hash": repro.get("dataset_hash", ""),
        "pair_seed": repro.get("pair_seed", ""),
        "num_repeats": repro.get("num_repeats", ""),
        "num_folds": repro.get("num_folds", ""),
        "eval_python_version": repro.get("python_version", ""),
        "eval_numpy_version": repro.get("numpy_version", ""),
    }


def _flatten_crossval(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("crossval", {}).get("summary_over_all_test_folds", {})
    fields: dict[str, Any] = {}
    for metric in ("accuracy", "auc", "eer", "far", "frr", "tar", "precision", "recall", "f1"):
        entry = summary.get(metric, {})
        fields[f"cv_{metric}_mean"] = entry.get("mean", "")
        fields[f"cv_{metric}_std"] = entry.get("std", "")

    tar_summary = summary.get("tar_at_far", {})
    for key, entry in tar_summary.items():
        normalized = key.replace("@", "_").replace("=", "_").replace(".", "p")
        fields[f"cv_{normalized}_mean"] = entry.get("mean", "")
        fields[f"cv_{normalized}_std"] = entry.get("std", "")
    return fields


def _flatten_bootstrap_ci(data: dict[str, Any]) -> dict[str, Any]:
    bootstrap = data.get("bootstrap_ci", {})
    fields: dict[str, Any] = {
        "bootstrap_n_valid": bootstrap.get("n_bootstrap_valid", ""),
    }
    for metric in ("best_accuracy", "auc", "eer", "far", "frr", "f1"):
        entry = bootstrap.get(metric, {})
        prefix = metric.replace("best_accuracy", "accuracy")
        fields[f"boot_{prefix}_mean"] = entry.get("mean", "")
        fields[f"boot_{prefix}_std"] = entry.get("std", "")
        fields[f"boot_{prefix}_ci95_low"] = entry.get("ci95_low", "")
        fields[f"boot_{prefix}_ci95_high"] = entry.get("ci95_high", "")

    for key, entry in bootstrap.get("tar_at_far", {}).items():
        normalized = key.replace("@", "_").replace("=", "_").replace(".", "p")
        fields[f"boot_{normalized}_mean"] = entry.get("mean", "")
        fields[f"boot_{normalized}_std"] = entry.get("std", "")
        fields[f"boot_{normalized}_ci95_low"] = entry.get("ci95_low", "")
        fields[f"boot_{normalized}_ci95_high"] = entry.get("ci95_high", "")
    return fields


def _flatten_attack_margin_bins(data: dict[str, Any]) -> dict[str, Any]:
    bins = data.get("clean_margin_bins", {})
    if not isinstance(bins, dict):
        return {"clean_margin_bins_json": ""}

    fields: dict[str, Any] = {
        "clean_margin_bins_json": json.dumps(bins, separators=(",", ":")),
    }
    for bucket_name, bucket_data in bins.items():
        if not isinstance(bucket_data, dict):
            continue
        prefix = f"clean_margin_{bucket_name}"
        fields[f"{prefix}_count"] = bucket_data.get("count", "")
        fields[f"{prefix}_attack_success_rate"] = bucket_data.get("attack_success_rate", "")
        fields[f"{prefix}_mean_attack_delta_vs_clean"] = bucket_data.get("mean_attack_delta_vs_clean", "")
        fields[f"{prefix}_mean_source_preservation_score"] = bucket_data.get("mean_source_preservation_score", "")
    return fields


def rebuild_compiled_csv() -> None:
    ensure_dir(RESULTS_ROOT)

    metrics_files = collect_metrics_files()
    rows: list[dict[str, Any]] = []

    base_fieldnames = [
        "evaluation_mode",
        "base_dataset",
        "referenced_base_root",
        "variant_name",
        "variant_type",
        "num_transforms",
        "transform_chain",
        "metrics_file",
        "transform_metadata_file",
        "pipeline_summary",
        "pipeline_json",
        "latest_transform",
        "oldest_transform",
        "pipeline_seed",
        "model",
        "pair_file",
        "embeddings_file",
        "dataset_hash",
        "pair_seed",
        "num_repeats",
        "num_folds",
        "eval_python_version",
        "eval_numpy_version",
        "num_embeddings",
        "embedding_dim",
        "num_pairs",
        "num_attack_samples",
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
        "score_min",
        "score_mean",
        "score_std",
        "score_max",
        "runtime_seconds",
        "limitations",
        "bootstrap_n_valid",
        "attack_success_rate",
        "victim_accept_rate",
        "attacker_accept_rate",
        "mean_clean_impostor_similarity",
        "mean_clean_margin",
        "num_preaccepted_clean_pairs",
        "mean_attack_target_similarity",
        "mean_attack_delta_vs_clean",
        "mean_margin_closure",
        "mean_residual_margin",
        "mean_source_preservation_score",
        "source_preservation_accept_rate",
        "constrained_attack_success_rate",
        "clean_margin_bins_json",
        "attack_success_rate_at_far_0p1",
        "attack_success_rate_at_far_0p01",
        "attack_success_rate_at_far_0p001",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_auc_mean",
        "cv_auc_std",
        "cv_eer_mean",
        "cv_eer_std",
    ]
    extra_fieldnames: list[str] = []

    for path in metrics_files:
        data = _load_json(path)
        metrics = _primary_metrics(data)
        variant_meta = _variant_metadata_from_results_path(path, data)
        transform_meta = _load_transform_metadata(variant_meta)
        transform_fields = _flatten_transform_metadata(transform_meta)
        repro_fields = _flatten_reproducibility(data)
        crossval_fields = _flatten_crossval(data)
        bootstrap_fields = _flatten_bootstrap_ci(data)
        tar_fields = _flatten_tar_at_far(metrics)
        attack_margin_fields = _flatten_attack_margin_bins(data)
        extra_fieldnames.extend(
            [name for name in transform_fields if name not in base_fieldnames and name not in extra_fieldnames]
        )
        extra_fieldnames.extend(
            [name for name in repro_fields if name not in base_fieldnames and name not in extra_fieldnames]
        )
        extra_fieldnames.extend(
            [name for name in crossval_fields if name not in base_fieldnames and name not in extra_fieldnames]
        )
        extra_fieldnames.extend(
            [name for name in bootstrap_fields if name not in base_fieldnames and name not in extra_fieldnames]
        )
        extra_fieldnames.extend(
            [name for name in attack_margin_fields if name not in base_fieldnames and name not in extra_fieldnames]
        )
        extra_fieldnames.extend([name for name in tar_fields if name not in extra_fieldnames])

        row = {
            **variant_meta,
            "evaluation_mode": data.get("evaluation_mode", "recognition"),
            "metrics_file": str(path),
            **transform_fields,
            **repro_fields,
            **crossval_fields,
            **bootstrap_fields,
            **attack_margin_fields,
            "model": data.get("model", path.parent.name),
            "pair_file": data.get("pair_file") or variant_meta["pair_file"],
            "embeddings_file": data.get("embeddings_file", ""),
            "num_embeddings": data.get("num_embeddings"),
            "embedding_dim": data.get("embedding_dim"),
            "num_pairs": data.get("num_pairs", metrics.get("num_pairs")),
            "num_attack_samples": data.get("num_attack_samples", ""),
            "num_genuine_pairs": data.get("num_genuine_pairs", metrics.get("num_genuine_pairs")),
            "num_impostor_pairs": data.get("num_impostor_pairs", metrics.get("num_impostor_pairs")),
            "best_threshold": metrics.get("best_threshold", data.get("best_threshold")),
            "accuracy": metrics.get("best_accuracy", data.get("best_accuracy")),
            "auc": metrics.get("auc", data.get("auc")),
            "eer": metrics.get("eer", data.get("eer")),
            "far": metrics.get("far", data.get("far")),
            "frr": metrics.get("frr", data.get("frr")),
            "tar": metrics.get("tar", data.get("tar")),
            "precision": metrics.get("precision", data.get("precision")),
            "recall": metrics.get("recall", data.get("recall")),
            "f1": metrics.get("f1", data.get("f1")),
            "tp": metrics.get("tp", data.get("tp")),
            "tn": metrics.get("tn", data.get("tn")),
            "fp": metrics.get("fp", data.get("fp")),
            "fn": metrics.get("fn", data.get("fn")),
            "eer_threshold": metrics.get("eer_threshold", data.get("eer_threshold")),
            "far_at_eer": metrics.get("far_at_eer", data.get("far_at_eer")),
            "frr_at_eer": metrics.get("frr_at_eer", data.get("frr_at_eer")),
            "score_min": metrics.get("score_min", ""),
            "score_mean": metrics.get("score_mean", ""),
            "score_std": metrics.get("score_std", ""),
            "score_max": metrics.get("score_max", ""),
            "runtime_seconds": data.get("runtime_seconds"),
            "limitations": " | ".join(data.get("limitations", [])),
            "attack_success_rate": data.get("attack_success_rate", ""),
            "victim_accept_rate": data.get("victim_accept_rate", ""),
            "attacker_accept_rate": data.get("attacker_accept_rate", ""),
            "mean_clean_impostor_similarity": data.get("mean_clean_impostor_similarity", ""),
            "mean_clean_margin": data.get("mean_clean_margin", ""),
            "num_preaccepted_clean_pairs": data.get("num_preaccepted_clean_pairs", ""),
            "mean_attack_target_similarity": data.get("mean_attack_target_similarity", ""),
            "mean_attack_delta_vs_clean": data.get("mean_attack_delta_vs_clean", ""),
            "mean_margin_closure": data.get("mean_margin_closure", ""),
            "mean_residual_margin": data.get("mean_residual_margin", ""),
            "mean_source_preservation_score": data.get("mean_source_preservation_score", ""),
            "source_preservation_accept_rate": data.get("source_preservation_accept_rate", ""),
            "constrained_attack_success_rate": data.get("constrained_attack_success_rate", ""),
            "attack_success_rate_at_far_0p1": data.get("attack_success_rate_at_far_0p1", ""),
            "attack_success_rate_at_far_0p01": data.get("attack_success_rate_at_far_0p01", ""),
            "attack_success_rate_at_far_0p001": data.get("attack_success_rate_at_far_0p001", ""),
            **tar_fields,
        }
        rows.append(row)

    fieldnames = base_fieldnames + extra_fieldnames
    rows.sort(key=lambda row: (row["base_dataset"], row["variant_name"], row["model"]))

    with open(COMPILED_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Compiled CSV refreshed: {COMPILED_CSV}")


def rebuild_parsed_results() -> None:
    if not RESULTS_PARSE_SCRIPT.exists():
        print(f"[WARN] results parse script missing: {RESULTS_PARSE_SCRIPT}")
        return
    cmd = [sys.executable, str(RESULTS_PARSE_SCRIPT), str(COMPILED_CSV)]
    run_subprocess(cmd, "results_parse.py")
