#!/usr/bin/env python3
# Example: python scripts/create_verification_pairs.py --help
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create verification pairs from an identity-root dataset.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--positive-pairs-per-id", type=int, default=4)
    parser.add_argument("--negative-pairs-per-id", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prefer-image-paths",
        dest="prefer_image_paths",
        action="store_true",
        help="When reading a manifest, prefer extracted image paths over shard::member references when available.",
    )
    parser.add_argument(
        "--prefer-shard-refs",
        dest="prefer_image_paths",
        action="store_false",
        help="When reading a manifest, keep shard::member references even if extracted image paths exist.",
    )
    parser.set_defaults(prefer_image_paths=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional extracted-image root used to resolve rel_path entries when image_path is missing.",
    )
    return parser.parse_args()


def _manifest_identity_samples(
    manifest_path: Path,
    *,
    prefer_image_paths: bool = True,
    image_root: Path | None = None,
) -> tuple[dict[str, list[str]], str]:
    mapping: dict[str, list[str]] = {}
    reference_mode = "shard"
    inferred_image_root = image_root.resolve() if image_root is not None else (manifest_path.resolve().parent.parent / "images")
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            label_name = str(item["label_name"])
            sample_ref = f"{Path(item['shard_path']).resolve()}::{item['key']}.jpg"
            if prefer_image_paths:
                image_path_value = item.get("image_path")
                candidate_path: Path | None = None
                if image_path_value is not None:
                    candidate_path = Path(str(image_path_value)).expanduser().resolve()
                elif item.get("rel_path") is not None and inferred_image_root.exists():
                    candidate_path = (inferred_image_root / str(item["rel_path"])).resolve()
                if candidate_path is not None and candidate_path.is_file():
                    sample_ref = str(candidate_path)
                    reference_mode = "image_path"
            mapping.setdefault(label_name, []).append(sample_ref)
    mapping = {identity: sorted(samples) for identity, samples in mapping.items() if len(samples) >= 2}
    if not mapping:
        raise RuntimeError(f"No usable manifest identities found in {manifest_path}")
    return mapping, reference_mode


def _identity_images(dataset_dir: Path) -> dict[str, list[Path]]:
    mapping: dict[str, list[Path]] = {}
    for identity_dir in sorted(dataset_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        images = sorted(
            path.resolve()
            for path in identity_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTS
        )
        if len(images) >= 2:
            mapping[identity_dir.name] = images
    if not mapping:
        raise RuntimeError(f"No usable identity directories found in {dataset_dir}")
    return mapping


def _dataset_hash(mapping: dict[str, list[Path]] | dict[str, list[str]]) -> str:
    digest = hashlib.sha256()
    for identity, paths in sorted(mapping.items()):
        digest.update(identity.encode("utf-8"))
        for path in paths:
            digest.update(str(path).encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_npz = args.output_npz.resolve()
    metadata_json = (args.metadata_json or output_npz.with_suffix(".json")).resolve()

    rng = random.Random(args.seed)
    reference_mode = "directory"
    if dataset_dir.is_file() and dataset_dir.suffix.lower() == ".jsonl":
        mapping, reference_mode = _manifest_identity_samples(
            dataset_dir,
            prefer_image_paths=bool(args.prefer_image_paths),
            image_root=args.image_root,
        )
    else:
        mapping = _identity_images(dataset_dir)
    identities = sorted(mapping)
    img1_paths: list[str] = []
    img2_paths: list[str] = []
    labels: list[int] = []
    repeat_ids: list[int] = []
    fold_ids: list[int] = []

    for repeat in range(1):
        for identity in identities:
            images = mapping[identity]
            pos_candidates = [(images[i], images[j]) for i in range(len(images)) for j in range(i + 1, len(images))]
            rng.shuffle(pos_candidates)
            for path_a, path_b in pos_candidates[: max(1, args.positive_pairs_per_id)]:
                img1_paths.append(str(path_a))
                img2_paths.append(str(path_b))
                labels.append(1)
                repeat_ids.append(repeat)
                fold_ids.append(rng.randrange(args.folds))

            other_identities = [item for item in identities if item != identity]
            rng.shuffle(other_identities)
            for other_identity in other_identities[: max(1, args.negative_pairs_per_id)]:
                path_a = rng.choice(images)
                path_b = rng.choice(mapping[other_identity])
                img1_paths.append(str(path_a))
                img2_paths.append(str(path_b))
                labels.append(0)
                repeat_ids.append(repeat)
                fold_ids.append(rng.randrange(args.folds))

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        img1_paths=np.asarray(img1_paths, dtype=object),
        img2_paths=np.asarray(img2_paths, dtype=object),
        labels=np.asarray(labels, dtype=np.int8),
        repeat_ids=np.asarray(repeat_ids, dtype=np.int64),
        fold_ids=np.asarray(fold_ids, dtype=np.int16),
        dataset_hash=_dataset_hash(mapping),
        seed=np.int64(args.seed),
        num_repeats=np.int64(1),
        num_folds=np.int64(args.folds),
    )
    metadata = {
        "dataset_dir": str(dataset_dir),
        "pairs_out": str(output_npz),
        "dataset_hash": _dataset_hash(mapping),
        "seed": args.seed,
        "folds": args.folds,
        "positive_pairs_per_id": args.positive_pairs_per_id,
        "negative_pairs_per_id": args.negative_pairs_per_id,
        "reference_mode": reference_mode,
        "prefer_image_paths": bool(args.prefer_image_paths),
        "image_root": str(args.image_root.resolve()) if args.image_root is not None else None,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote pairs: {output_npz}")
    print(f"[INFO] Metadata:    {metadata_json}")


if __name__ == "__main__":
    main()
