#!/usr/bin/env python3
# python3 results_parse.py <compiled_results.csv> -> Results/parsed/*.csv for readable variant, model, delta, and transform summaries

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


CORE_METRICS = [
    "accuracy",
    "auc",
    "eer",
    "tar_far_0p1",
    "tar_far_0p01",
    "tar_far_0p001",
    "f1",
    "precision",
    "recall",
    "runtime_seconds",
]


def _to_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("base_dataset", ""),
        row.get("referenced_base_root", ""),
        row.get("variant_type", ""),
        row.get("variant_name", ""),
        row.get("model", ""),
    )


def build_model_results_long(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        selected.append({
            "base_dataset": row.get("base_dataset", ""),
            "referenced_base_root": row.get("referenced_base_root", ""),
            "variant_name": row.get("variant_name", ""),
            "variant_type": row.get("variant_type", ""),
            "pipeline_summary": row.get("pipeline_summary", "") or row.get("transform_chain", ""),
            "latest_transform": row.get("latest_transform", ""),
            "model": row.get("model", ""),
            "num_embeddings": row.get("num_embeddings", ""),
            "num_pairs": row.get("num_pairs", ""),
            "accuracy": row.get("accuracy", ""),
            "auc": row.get("auc", ""),
            "eer": row.get("eer", ""),
            "tar_far_0p1": row.get("tar_far_0p1", ""),
            "tar_far_0p01": row.get("tar_far_0p01", ""),
            "tar_far_0p001": row.get("tar_far_0p001", ""),
            "f1": row.get("f1", ""),
            "best_threshold": row.get("best_threshold", ""),
            "runtime_seconds": row.get("runtime_seconds", ""),
            "metrics_file": row.get("metrics_file", ""),
            "transform_metadata_file": row.get("transform_metadata_file", ""),
        })
    selected.sort(key=_sort_key)
    return selected


def build_variant_overview(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("base_dataset", ""), row.get("referenced_base_root", ""), row.get("variant_name", ""))
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for _, group in sorted(grouped.items()):
        anchor = group[0]
        out = {
            "base_dataset": anchor.get("base_dataset", ""),
            "referenced_base_root": anchor.get("referenced_base_root", ""),
            "variant_name": anchor.get("variant_name", ""),
            "variant_type": anchor.get("variant_type", ""),
            "pipeline_summary": anchor.get("pipeline_summary", "") or anchor.get("transform_chain", ""),
            "latest_transform": anchor.get("latest_transform", ""),
            "num_transforms": anchor.get("num_transforms", ""),
            "num_pairs": anchor.get("num_pairs", ""),
        }
        for row in sorted(group, key=lambda r: r.get("model", "")):
            model = row.get("model", "")
            for metric in ("accuracy", "auc", "eer", "tar_far_0p01", "tar_far_0p001", "runtime_seconds"):
                out[f"{model}_{metric}"] = row.get(metric, "")
        output.append(out)
    return output


def build_clean_variant_deltas(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("variant_type") == "clean":
            key = (row.get("base_dataset", ""), row.get("referenced_base_root", ""), row.get("model", ""))
            baselines[key] = row

    output: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("base_dataset", ""), row.get("referenced_base_root", ""), row.get("model", ""))
        clean = baselines.get(key)
        out = {
            "base_dataset": row.get("base_dataset", ""),
            "referenced_base_root": row.get("referenced_base_root", ""),
            "variant_name": row.get("variant_name", ""),
            "variant_type": row.get("variant_type", ""),
            "pipeline_summary": row.get("pipeline_summary", "") or row.get("transform_chain", ""),
            "latest_transform": row.get("latest_transform", ""),
            "model": row.get("model", ""),
            "clean_variant_name": clean.get("variant_name", "") if clean else "",
            "accuracy": row.get("accuracy", ""),
            "auc": row.get("auc", ""),
            "eer": row.get("eer", ""),
            "tar_far_0p01": row.get("tar_far_0p01", ""),
            "tar_far_0p001": row.get("tar_far_0p001", ""),
        }
        for metric in ("accuracy", "auc", "eer", "tar_far_0p01", "tar_far_0p001"):
            current = _to_float(row.get(metric))
            baseline = _to_float(clean.get(metric)) if clean else None
            out[f"{metric}_delta_vs_clean"] = "" if current is None or baseline is None else current - baseline
        output.append(out)
    output.sort(key=_sort_key)
    return output


def build_transform_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    delta_rows = build_clean_variant_deltas(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        latest_transform = row.get("latest_transform", "") or "clean"
        grouped[(latest_transform, row.get("model", ""))].append(row)

    output: list[dict[str, Any]] = []
    for (latest_transform, model), group in sorted(grouped.items()):
        out: dict[str, Any] = {
            "latest_transform": latest_transform,
            "model": model,
            "num_variants": len(group),
        }
        for metric in ("accuracy", "auc", "eer", "tar_far_0p01", "tar_far_0p001"):
            values = [_to_float(row.get(metric)) for row in group]
            values = [value for value in values if value is not None]
            out[f"mean_{metric}"] = "" if not values else sum(values) / len(values)

            delta_values = [_to_float(row.get(f"{metric}_delta_vs_clean")) for row in group]
            delta_values = [value for value in delta_values if value is not None]
            out[f"mean_{metric}_delta_vs_clean"] = "" if not delta_values else sum(delta_values) / len(delta_values)
        output.append(out)
    return output


def build_metric_rankings(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ranked = [dict(row) for row in rows]
    ranked.sort(
        key=lambda row: (
            row.get("base_dataset", ""),
            row.get("referenced_base_root", ""),
            -(_to_float(row.get("accuracy")) or float("-inf")),
            -(_to_float(row.get("auc")) or float("-inf")),
            (_to_float(row.get("eer")) if _to_float(row.get("eer")) is not None else float("inf")),
        )
    )
    output: list[dict[str, Any]] = []
    current_group = None
    rank = 0
    for row in ranked:
        group = (row.get("base_dataset", ""), row.get("referenced_base_root", ""))
        if group != current_group:
            current_group = group
            rank = 1
        else:
            rank += 1
        output.append({
            "base_dataset": row.get("base_dataset", ""),
            "referenced_base_root": row.get("referenced_base_root", ""),
            "rank": rank,
            "variant_name": row.get("variant_name", ""),
            "variant_type": row.get("variant_type", ""),
            "pipeline_summary": row.get("pipeline_summary", "") or row.get("transform_chain", ""),
            "model": row.get("model", ""),
            "accuracy": row.get("accuracy", ""),
            "auc": row.get("auc", ""),
            "eer": row.get("eer", ""),
            "tar_far_0p01": row.get("tar_far_0p01", ""),
            "tar_far_0p001": row.get("tar_far_0p001", ""),
        })
    return output


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 results_parse.py <compiled_results.csv>")

    compiled_path = Path(sys.argv[1]).resolve()
    if not compiled_path.exists():
        raise FileNotFoundError(compiled_path)

    rows = _read_csv(compiled_path)
    parsed_root = compiled_path.parent / "parsed"

    model_results_long = build_model_results_long(rows)
    _write_csv(
        parsed_root / "model_results_long.csv",
        model_results_long,
        list(model_results_long[0].keys()) if model_results_long else [],
    )

    variant_overview = build_variant_overview(rows)
    _write_csv(
        parsed_root / "variant_overview.csv",
        variant_overview,
        list(variant_overview[0].keys()) if variant_overview else [],
    )

    clean_variant_deltas = build_clean_variant_deltas(rows)
    _write_csv(
        parsed_root / "clean_variant_deltas.csv",
        clean_variant_deltas,
        list(clean_variant_deltas[0].keys()) if clean_variant_deltas else [],
    )

    transform_summary = build_transform_summary(rows)
    _write_csv(
        parsed_root / "transform_summary.csv",
        transform_summary,
        list(transform_summary[0].keys()) if transform_summary else [],
    )

    metric_rankings = build_metric_rankings(rows)
    _write_csv(
        parsed_root / "metric_rankings.csv",
        metric_rankings,
        list(metric_rankings[0].keys()) if metric_rankings else [],
    )

    print(f"[INFO] Parsed result tables written to: {parsed_root}")


if __name__ == "__main__":
    main()
