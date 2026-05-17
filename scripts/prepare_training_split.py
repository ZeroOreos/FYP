#!/usr/bin/env python3
# Example: python scripts/prepare_training_split.py --help
"""Create a small train/val identity split for local training smoke runs."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a symlinked identity split from any identity-root dataset.")
    parser.add_argument("source_dir", type=Path, help="Source identity-root dataset.")
    parser.add_argument("output_root", type=Path, help="Output root that will contain train/ and val/")
    parser.add_argument("--num-identities", type=int, default=32, help="Number of identities to include.")
    parser.add_argument("--train-images-per-id", type=int, default=4, help="Train images per identity.")
    parser.add_argument("--val-images-per-id", type=int, default=2, help="Validation images per identity.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_identity_dirs(source_dir: Path) -> list[Path]:
    identities = []
    for identity_dir in sorted(source_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        image_count = len([path for path in identity_dir.iterdir() if path.suffix.lower() in VALID_EXTS])
        if image_count >= 2:
            identities.append(identity_dir)
    if not identities:
        raise RuntimeError(f"No usable identities found in: {source_dir}")
    return identities


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_split_for_identity(
    identity_dir: Path,
    train_dir: Path,
    val_dir: Path,
    train_images_per_id: int,
    val_images_per_id: int,
) -> tuple[int, int]:
    images = [path for path in sorted(identity_dir.iterdir()) if path.is_file() and path.suffix.lower() in VALID_EXTS]
    if len(images) < 2:
        return 0, 0

    train_count = min(train_images_per_id, max(1, len(images) - 1))
    remaining = images[train_count:]
    if not remaining:
        remaining = images[-1:]
    val_count = min(val_images_per_id, len(remaining))

    train_identity_dir = train_dir / identity_dir.name
    val_identity_dir = val_dir / identity_dir.name
    ensure_clean_dir(train_identity_dir)
    ensure_clean_dir(val_identity_dir)

    for image_path in images[:train_count]:
        target = train_identity_dir / image_path.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(image_path.resolve())

    for image_path in remaining[:val_count]:
        target = val_identity_dir / image_path.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(image_path.resolve())

    return train_count, val_count


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_root = args.output_root.resolve()
    train_dir = output_root / "train"
    val_dir = output_root / "val"

    identities = collect_identity_dirs(source_dir)
    rng = random.Random(args.seed)
    rng.shuffle(identities)
    selected = identities[: max(1, min(args.num_identities, len(identities)))]

    ensure_clean_dir(train_dir)
    ensure_clean_dir(val_dir)

    total_train = 0
    total_val = 0
    kept_ids = 0
    for identity_dir in selected:
        train_count, val_count = build_split_for_identity(
            identity_dir=identity_dir,
            train_dir=train_dir,
            val_dir=val_dir,
            train_images_per_id=max(1, args.train_images_per_id),
            val_images_per_id=max(1, args.val_images_per_id),
        )
        if train_count > 0 and val_count > 0:
            kept_ids += 1
            total_train += train_count
            total_val += val_count

    print(f"[INFO] Source dir:      {source_dir}")
    print(f"[INFO] Output root:     {output_root}")
    print(f"[INFO] Identities kept: {kept_ids}")
    print(f"[INFO] Train images:    {total_train}")
    print(f"[INFO] Val images:      {total_val}")


if __name__ == "__main__":
    main()
