#!/usr/bin/env python3
# Example: python scripts/extract_webface4m_images.py --help
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract WebFace4M shard JPEGs into an identity-root image tree.")
    parser.add_argument("--target-root", type=Path, default=Path("Dataset/WebFace4M"))
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to <target-root>/images.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _label_map(handle: tarfile.TarFile) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for member in handle.getmembers():
        if not member.isfile() or not member.name.endswith(".cls"):
            continue
        file_obj = handle.extractfile(member)
        if file_obj is None:
            continue
        mapping[Path(member.name).stem] = file_obj.read().decode("utf-8").strip()
    return mapping


def main() -> None:
    args = parse_args()
    target_root = args.target_root.resolve()
    raw_root = target_root / "raw"
    output_root = (args.output_root or (target_root / "images")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(raw_root.glob("*.tar.gz"))
    if not shard_paths:
        raise RuntimeError(f"No shard files found in {raw_root}")

    written = 0
    skipped = 0
    missing_labels = 0
    for shard_path in shard_paths:
        with tarfile.open(shard_path, mode="r:gz") as handle:
            labels = _label_map(handle)
            for member in handle.getmembers():
                if not member.isfile() or not member.name.endswith(".jpg"):
                    continue
                key = Path(member.name).stem
                label_name = labels.get(key)
                if label_name is None:
                    missing_labels += 1
                    continue
                destination = output_root / label_name / f"{key}.jpg"
                if destination.exists() and not args.overwrite:
                    skipped += 1
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                file_obj = handle.extractfile(member)
                if file_obj is None:
                    continue
                destination.write_bytes(file_obj.read())
                written += 1

    summary = {
        "target_root": str(target_root),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "written_images": written,
        "skipped_existing": skipped,
        "missing_labels": missing_labels,
    }
    (target_root / "extract_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
