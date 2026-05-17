#!/usr/bin/env python3
# Example: python scripts/bootstrap_backends.py --help
"""Audit and lightly prepare backend source, asset, and workdir layout.

This script intentionally does not clone repositories or download weights.
It exists to keep the backend layout explicit, predictable, and easy to audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "Backends" / "manifest.json"


def load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_str: str) -> Path:
    return PROJECT_ROOT / path_str


def resolve_required_path(required_path_str: str, entry: dict, active_repo: Path) -> Path:
    required_path = resolve_path(required_path_str)
    canonical_repo = resolve_path(entry["repo_dir"])
    try:
        relative_to_repo = required_path.relative_to(canonical_repo)
    except ValueError:
        return required_path
    return active_repo / relative_to_repo


def collect_entry_issues(entry: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    repo_dir = resolve_path(entry["repo_dir"])
    legacy_dirs = [resolve_path(path_str) for path_str in entry.get("legacy_repo_dirs", [])]
    repo_exists = repo_dir.exists()
    legacy_exists = [path for path in legacy_dirs if path.exists()]

    if not repo_exists and not legacy_exists:
        errors.append(f"missing repo_dir: {entry['repo_dir']}")

    active_repo = repo_dir if repo_exists else (legacy_exists[0] if legacy_exists else repo_dir)

    nested_git = active_repo / ".git"
    if nested_git.exists():
        warnings.append(f"nested git repo present: {nested_git.relative_to(PROJECT_ROOT)}")

    for required_path_str in entry.get("required_paths", []):
        required_path = resolve_required_path(required_path_str, entry, active_repo)
        if not required_path.exists():
            errors.append(
                f"missing required path: {required_path.relative_to(PROJECT_ROOT)}"
            )

    checkpoint_dir_str = entry.get("checkpoint_dir")
    if checkpoint_dir_str:
        checkpoint_dir = resolve_path(checkpoint_dir_str)
        if not checkpoint_dir.exists():
            warnings.append(f"missing checkpoint dir: {checkpoint_dir_str}")

    for runtime_output_str in entry.get("runtime_outputs", []):
        runtime_output = resolve_path(runtime_output_str)
        if runtime_output.exists():
            warnings.append(f"runtime output present inside backend source tree: {runtime_output_str}")

    for tolerated_path_str in entry.get("tolerated_legacy_paths", []):
        tolerated_path = resolve_path(tolerated_path_str)
        if not tolerated_path.exists():
            warnings.append(f"missing tolerated legacy path: {tolerated_path_str}")

    if repo_exists and legacy_exists:
        warnings.append(
            "both canonical and legacy dependency paths exist; keep one source of truth"
        )

    return errors, warnings


def print_report(manifest: dict) -> tuple[int, int]:
    total_errors = 0
    total_warnings = 0

    print("[backends] manifest:", MANIFEST_PATH.relative_to(PROJECT_ROOT))
    print("[backends] entries:", len(manifest.get("entries", [])))

    for entry in manifest.get("entries", []):
        print()
        print(
            f"- {entry['name']} [{entry['kind']}] status={entry['status']}"
        )
        errors, warnings = collect_entry_issues(entry)
        total_errors += len(errors)
        total_warnings += len(warnings)
        if not errors and not warnings:
            print("  ok")
            continue
        for item in errors:
            print(f"  error: {item}")
        for item in warnings:
            print(f"  warn: {item}")

    print()
    print(f"[backends] errors={total_errors} warnings={total_warnings}")
    return total_errors, total_warnings


def prepare_layout(manifest: dict) -> None:
    layout = manifest.get("layout", {})
    for key in (
        "recognition_sources_root",
        "dependencies_root",
        "assets_root",
        "workdirs_root",
        "attack_sources_root",
        "attack_primary_sources_root",
        "attack_surrogate_sources_root",
        "attack_assets_root",
        "attack_workdirs_root",
        "recognition_assets_root",
        "recognition_workdirs_root",
    ):
        path_str = layout.get(key)
        if not path_str:
            continue
        resolve_path(path_str).mkdir(parents=True, exist_ok=True)

    for entry in manifest.get("entries", []):
        checkpoint_dir_str = entry.get("checkpoint_dir")
        if checkpoint_dir_str:
            resolve_path(checkpoint_dir_str).mkdir(parents=True, exist_ok=True)
        workdir_dir_str = entry.get("workdir_dir")
        if workdir_dir_str:
            workdir = resolve_path(workdir_dir_str)
        elif entry.get("kind") == "backend":
            workdir = resolve_path(layout["workdirs_root"]) / entry["name"]
        else:
            workdir = None
        if workdir is not None:
            workdir.mkdir(parents=True, exist_ok=True)
            gitkeep = workdir / ".gitkeep"
            gitkeep.touch(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and prepare backend layout.")
    parser.add_argument("command", choices=("audit", "prepare"))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when warnings are present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest()

    if args.command == "prepare":
        prepare_layout(manifest)

    errors, warnings = print_report(manifest)
    if errors or (args.strict and warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
