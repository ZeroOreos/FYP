#!/usr/bin/env python3
# Example: python scripts/aggregate_seed_runs.py --help
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_METRICS = (
    "val_clean.accuracy",
    "val_robust.accuracy",
    "val_robust.attack_success_rate",
    "val_verification.roc_auc",
    "val_verification.eer",
    "val_verification.tar@far=0.0001",
    "val_verification.tar@far=1e-05",
    "val_clean.ensemble.gain_vs_best_single",
    "val_robust.ensemble.gain_vs_best_single",
    "val_clean.recognizer_ensemble.gain_vs_best_single",
    "val_robust.recognizer_ensemble.gain_vs_best_single",
    "performance.global_img_s",
    "performance.avg_step_ms",
    "memory.cuda_peak_mb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate repeated-seed runs into mean/std and 95% CI summaries.")
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        required=True,
        help="Completed run directory. Pass multiple times.",
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=None,
        help="Metric path to aggregate from latest_epoch or summary. Defaults to a core paper set.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the aggregation JSON.",
    )
    return parser.parse_args()


def _load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _dig(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _metric_value(summary: dict[str, Any], metric: str) -> float | None:
    latest_epoch = summary.get("latest_epoch") or (summary.get("history") or [None])[-1]
    for root in (latest_epoch, summary):
        if isinstance(root, dict):
            value = _dig(root, metric)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
    return None


def _aggregate(values: list[float]) -> dict[str, float]:
    n = len(values)
    avg = mean(values)
    if n == 1:
        return {
            "n": 1.0,
            "mean": avg,
            "std": 0.0,
            "ci95_half_width": 0.0,
            "min": values[0],
            "max": values[0],
        }
    sigma = stdev(values)
    ci95 = 1.96 * sigma / math.sqrt(n)
    return {
        "n": float(n),
        "mean": avg,
        "std": sigma,
        "ci95_half_width": ci95,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    metrics = tuple(args.metrics or DEFAULT_METRICS)
    run_dirs = [Path(item).resolve() for item in args.run_dirs]
    summaries = {str(run_dir): _load_summary(run_dir) for run_dir in run_dirs}

    aggregated: dict[str, Any] = {
        "run_dirs": [str(path) for path in run_dirs],
        "metrics": {},
    }
    for metric in metrics:
        collected: list[float] = []
        per_run: dict[str, float | None] = {}
        for run_dir, summary in summaries.items():
            value = _metric_value(summary, metric)
            per_run[run_dir] = value
            if value is not None:
                collected.append(value)
        aggregated["metrics"][metric] = {
            "per_run": per_run,
            "aggregate": _aggregate(collected) if collected else None,
        }

    rendered = json.dumps(aggregated, indent=2, sort_keys=True)
    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
