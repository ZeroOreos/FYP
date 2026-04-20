#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


EXPORT_FILES = (
    "config.snapshot.json",
    "history.json",
    "metrics.jsonl",
    "repro_state.json",
    "summary.json",
    "train.log",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect run metrics and metadata into a portable export bundle.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, default=None, help="Defaults to <run-dir>/exports.")
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--bundle", action="store_true", help="Create a .tar.gz bundle alongside the export directory.")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history", [])
    if not history:
        return {}
    latest = history[-1]
    return {
        "epoch": latest.get("epoch"),
        "training_stage": latest.get("training_stage"),
        "train_loss": latest.get("train_loss"),
        "train_accuracy": latest.get("train_accuracy"),
        "val_clean_accuracy": latest.get("val_clean", {}).get("accuracy"),
        "val_robust_accuracy": latest.get("val_robust", {}).get("accuracy"),
        "val_verification_auc": latest.get("val_verification", {}).get("auc"),
        "val_verification_accuracy": latest.get("val_verification", {}).get("accuracy"),
    }


def _copy_if_present(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    export_root = (args.export_root or (run_dir / "exports")).resolve()
    export_dir = export_root / run_dir.name
    export_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else {"output_dir": str(run_dir), "history": []}
    manifest = {
        "run_dir": str(run_dir),
        "export_dir": str(export_dir),
        "latest_metrics": _latest_metrics(summary),
        "files": [],
    }

    for filename in EXPORT_FILES:
        source = run_dir / filename
        target = export_dir / filename
        if source.exists():
            _copy_if_present(source, target)
            manifest["files"].append(filename)

    if args.include_checkpoints:
        checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir.exists():
            shutil.copytree(checkpoint_dir, export_dir / "checkpoints", dirs_exist_ok=True)
            manifest["files"].append("checkpoints/")

    manifest_path = export_dir / "export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    latest_metrics_path = export_dir / "latest_metrics.json"
    with latest_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest["latest_metrics"], handle, indent=2, sort_keys=True)
        handle.write("\n")

    bundle_path = None
    if args.bundle:
        bundle_path = export_root / f"{run_dir.name}.tar.gz"
        with tarfile.open(bundle_path, "w:gz") as handle:
            handle.add(export_dir, arcname=run_dir.name)

    result = {
        "run_dir": str(run_dir),
        "export_dir": str(export_dir),
        "bundle_path": str(bundle_path) if bundle_path is not None else None,
        "latest_metrics_path": str(latest_metrics_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
