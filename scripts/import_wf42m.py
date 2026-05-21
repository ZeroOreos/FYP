#!/usr/bin/env python3
# Example: python scripts/import_wf42m.py --help
"""Prepare a canonical WebFace42M dataset root.

This does not attempt to download WebFace42M. It creates the canonical local target
directory and can register or symlink an existing local source tree when available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = PROJECT_ROOT / "Dataset" / "WebFace42M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the local WebFace42M dataset root.")
    parser.add_argument("--source-root", type=Path, default=None, help="Optional existing local WebFace42M identity-root source.")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET, help="Local target root to create.")
    parser.add_argument("--link-source", action="store_true", help="Create a raw/ symlink to --source-root.")
    return parser.parse_args()


def write_manifest(target_root: Path, source_root: Path | None, linked: bool) -> None:
    manifest = {
        "dataset_name": "WebFace42M",
        "target_root": str(target_root),
        "source_root": None if source_root is None else str(source_root),
        "linked_source": linked,
        "status": "prepared" if source_root is None else "source_registered",
        "notes": [
            "The published ceiling targets WebFace42M with RetinaFace-style 5-point alignment and 112x112 crops.",
            "This repo keeps a smaller local proxy backbone and embedding dimension for practical iteration.",
            "Actual WebFace42M contents are not bundled in the repo and must come from a local source prepared separately.",
        ],
    }
    with (target_root / "import_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def write_readme(target_root: Path) -> None:
    readme = target_root / "README.md"
    readme.write_text(
        "# WebFace42M Local Target\n\n"
        "This folder is the canonical adjacent import target for the WF42M ceiling pipeline.\n\n"
        "- Expected published ceiling: RetinaFace-class 5-point alignment, 112x112 normalized crop\n"
        "- Local proxy setting: smaller backbone and embedding dimension until full-scale WF42M is practical\n"
        "- Practical local alternative under about 20GB: aligned WebFace4M\n"
        "- Put or link a local WF42M identity-root source here via `scripts/import_wf42m.py`\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    target_root = args.target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    write_readme(target_root)

    linked = False
    if args.source_root is not None:
        source_root = args.source_root.resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"Source root not found: {source_root}")
        if args.link_source:
            raw_link = target_root / "raw"
            if raw_link.exists() or raw_link.is_symlink():
                raw_link.unlink()
            raw_link.symlink_to(source_root)
            linked = True
        write_manifest(target_root, source_root, linked)
    else:
        write_manifest(target_root, None, linked)

    print(f"[INFO] WebFace42M target root: {target_root}")
    if args.source_root is not None:
        print(f"[INFO] Registered source:      {args.source_root.resolve()}")
        if linked:
            print(f"[INFO] Created raw symlink:   {target_root / 'raw'}")
    else:
        print("[INFO] No source root supplied; prepared adjacent target only.")


if __name__ == "__main__":
    main()
