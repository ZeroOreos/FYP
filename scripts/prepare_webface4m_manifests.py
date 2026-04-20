#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val manifests for shard-backed WebFace4M training.")
    parser.add_argument("--target-root", type=Path, default=Path("Dataset/WebFace4M"))
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--min-val-per-class", type=int, default=1)
    parser.add_argument("--max-val-per-class", type=int, default=5)
    return parser.parse_args()


def _iter_cls_members(shard_path: Path):
    with tarfile.open(shard_path, mode="r:gz") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith(".cls"):
                continue
            file_obj = handle.extractfile(member)
            if file_obj is None:
                continue
            label_name = file_obj.read().decode("utf-8").strip()
            key = Path(member.name).stem
            yield key, label_name


def _stable_score(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16 ** 12 - 1)


def main() -> None:
    args = parse_args()
    target_root = args.target_root.resolve()
    raw_root = target_root / "raw"
    manifest_root = target_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(raw_root.glob("*.tar.gz"))
    if not shard_paths:
        raise RuntimeError(f"No shard files found in {raw_root}")

    class_counts: Counter[str] = Counter()
    total_samples = 0
    for shard_path in shard_paths:
        for _, label_name in _iter_cls_members(shard_path):
            class_counts[label_name] += 1
            total_samples += 1

    if not class_counts:
        raise RuntimeError(f"No .cls label entries found in shard set: {raw_root}")

    ordered_labels = sorted(class_counts.keys(), key=lambda item: int(item))
    class_to_idx = {label_name: idx for idx, label_name in enumerate(ordered_labels)}

    val_quota: dict[str, int] = {}
    for label_name, count in class_counts.items():
        if count <= 1:
            val_quota[label_name] = 0
            continue
        quota = int(round(count * args.val_fraction))
        quota = max(args.min_val_per_class, quota)
        quota = min(args.max_val_per_class, quota, count - 1)
        val_quota[label_name] = max(0, quota)

    train_manifest = manifest_root / "train.jsonl"
    val_manifest = manifest_root / "val.jsonl"
    val_assigned: defaultdict[str, int] = defaultdict(int)
    seen_count: defaultdict[str, int] = defaultdict(int)
    train_samples = 0
    val_samples = 0

    with train_manifest.open("w", encoding="utf-8") as train_handle, val_manifest.open("w", encoding="utf-8") as val_handle:
        for shard_path in shard_paths:
            for key, label_name in _iter_cls_members(shard_path):
                seen_count[label_name] += 1
                total_for_label = class_counts[label_name]
                remaining_for_label = total_for_label - seen_count[label_name]
                remaining_needed = val_quota[label_name] - val_assigned[label_name]
                split = "train"
                if remaining_needed > 0:
                    threshold = val_quota[label_name] / float(total_for_label)
                    if _stable_score(key) < threshold or remaining_for_label < remaining_needed:
                        split = "val"
                        val_assigned[label_name] += 1
                payload = {
                    "shard_path": str(shard_path.resolve()),
                    "key": key,
                    "label_name": label_name,
                    "label_idx": class_to_idx[label_name],
                    "rel_path": f"{label_name}/{key}.jpg",
                }
                if split == "val":
                    val_handle.write(json.dumps(payload, sort_keys=True) + "\n")
                    val_samples += 1
                else:
                    train_handle.write(json.dumps(payload, sort_keys=True) + "\n")
                    train_samples += 1

    (manifest_root / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "dataset_name": "WebFace4M",
        "raw_root": str(raw_root),
        "num_shards": len(shard_paths),
        "classes": len(class_to_idx),
        "total_samples": total_samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "val_fraction": args.val_fraction,
        "min_val_per_class": args.min_val_per_class,
        "max_val_per_class": args.max_val_per_class,
    }
    (manifest_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
