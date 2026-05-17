#!/usr/bin/env python3
# Example: python scripts/summarize_train_log.py --help
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_/.:-]+)=([^\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize training throughput and timing from a train.log file.")
    parser.add_argument("log_path", type=Path, help="Path to train.log or a tee-captured stdout log.")
    parser.add_argument(
        "--tail",
        type=int,
        default=50,
        help="How many most-recent [BATCH] lines to average. Default: 50.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def _to_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _extract_batch_metrics(lines: list[str]) -> list[dict[str, float | str]]:
    metrics: list[dict[str, float | str]] = []
    for line in lines:
        if "[BATCH]" not in line:
            continue
        entry: dict[str, float | str] = {}
        for key, value in KEY_VALUE_RE.findall(line):
            numeric = _to_float(value)
            entry[key] = numeric if numeric is not None else value
        if entry:
            metrics.append(entry)
    return metrics


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _latest(entries: list[dict[str, float | str]], key: str) -> float | str | None:
    for entry in reversed(entries):
        if key in entry:
            return entry[key]
    return None


def _summarize(entries: list[dict[str, float | str]]) -> dict[str, object]:
    numeric_keys = (
        "global_img_s",
        "opt_steps_s",
        "avg_wait_ms",
        "avg_step_ms",
        "loss",
        "acc",
    )
    summary: dict[str, object] = {
        "num_batch_lines": len(entries),
        "latest_step": _latest(entries, "step"),
        "latest_epoch": _latest(entries, "epoch"),
        "latest_stage": _latest(entries, "stage"),
        "latest_accum": _latest(entries, "accum"),
    }
    for key in numeric_keys:
        values = [float(entry[key]) for entry in entries if isinstance(entry.get(key), (int, float))]
        avg = _average(values)
        if avg is not None:
            summary[f"{key}_avg"] = avg
        latest = _latest(entries, key)
        if latest is not None:
            summary[f"{key}_latest"] = latest
    return summary


def main() -> None:
    args = parse_args()
    lines = args.log_path.read_text(encoding="utf-8").splitlines()
    batch_entries = _extract_batch_metrics(lines)
    if not batch_entries:
        raise SystemExit(f"No [BATCH] lines found in {args.log_path}")
    tail_entries = batch_entries[-max(1, args.tail):]
    summary = {
        "log_path": str(args.log_path.resolve()),
        "tail_window": max(1, args.tail),
        **_summarize(tail_entries),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"log_path={summary['log_path']}")
    print(f"tail_window={summary['tail_window']} batch_lines={summary['num_batch_lines']}")
    for key in (
        "latest_epoch",
        "latest_step",
        "latest_stage",
        "latest_accum",
        "global_img_s_avg",
        "global_img_s_latest",
        "opt_steps_s_avg",
        "avg_wait_ms_avg",
        "avg_step_ms_avg",
        "loss_avg",
        "loss_latest",
        "acc_avg",
        "acc_latest",
    ):
        if key in summary:
            print(f"{key}={summary[key]}")


if __name__ == "__main__":
    main()
