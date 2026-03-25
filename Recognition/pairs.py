# python3 Recognition/pairs.py <dataset_dir> -> pair data and pair-file helpers
import argparse
import hashlib
import itertools
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np


if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from Utility.paths import derive_pairs_output_path, resolve_dataset_context


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SEED = 42
DEFAULT_REPEATS = 1
DEFAULT_FOLDS = 5


def collect_identity_images(dataset_dir: Path):
    identities = {}

    for id_dir in sorted(dataset_dir.iterdir()):
        if not id_dir.is_dir():
            continue

        images = sorted(
            p for p in id_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        )

        if len(images) >= 2:
            identities[id_dir.name] = images

    if not identities:
        raise RuntimeError("No valid identities with >=2 images found.")

    return identities


def build_dataset_manifest_hash(dataset_dir: Path, identities):
    h = hashlib.sha256()

    for identity, images in sorted(identities.items()):
        h.update(identity.encode("utf-8"))
        for img in images:
            rel = img.relative_to(dataset_dir).as_posix()
            size = img.stat().st_size
            h.update(rel.encode("utf-8"))
            h.update(str(size).encode("utf-8"))

    return h.hexdigest()


def generate_genuine_pairs(identities):
    pairs = []

    for identity in sorted(identities.keys()):
        img_list = identities[identity]
        for a, b in itertools.combinations(img_list, 2):
            pairs.append((a, b, 1))

    return pairs


def estimate_max_impostor_pairs(identities):
    id_names = sorted(identities.keys())
    total = 0

    for i in range(len(id_names)):
        n1 = len(identities[id_names[i]])
        for j in range(i + 1, len(id_names)):
            n2 = len(identities[id_names[j]])
            total += n1 * n2

    return total


def generate_impostor_pairs(identities, target_count, rng):
    id_names = sorted(identities.keys())
    pairs = set()

    max_possible = estimate_max_impostor_pairs(identities)

    if target_count > max_possible:
        target_count = max_possible

    max_attempts = max(target_count * 20, 10000)
    attempts = 0

    while len(pairs) < target_count and attempts < max_attempts:
        id1, id2 = rng.sample(id_names, 2)
        img1 = rng.choice(identities[id1])
        img2 = rng.choice(identities[id2])

        a, b = sorted((str(img1), str(img2)))
        pairs.add((a, b))
        attempts += 1

    return [(Path(a), Path(b), 0) for a, b in sorted(pairs)]


def assign_folds(labels, folds, rng):
    labels = np.asarray(labels)
    fold_ids = np.empty(len(labels), dtype=np.int16)

    for cls in [0, 1]:
        cls_idx = np.where(labels == cls)[0].tolist()
        rng.shuffle(cls_idx)

        for i, idx in enumerate(cls_idx):
            fold_ids[idx] = i % folds

    return fold_ids


def pairs_to_arrays(pairs):
    img1_paths = np.array([str(a) for a, _, _ in pairs], dtype=object)
    img2_paths = np.array([str(b) for _, b, _ in pairs], dtype=object)
    labels = np.array([lbl for _, _, lbl in pairs], dtype=np.int8)
    return img1_paths, img2_paths, labels


def main():
    parser = argparse.ArgumentParser(description="Generate balanced face verification pairs.")
    parser.add_argument("dataset_dir", type=Path, help="Path to dataset root (identity/image structure).")
    parser.add_argument("--pairs-out", type=Path, default=None, help="Optional custom output path.")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--max-genuine", type=int, default=None)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    context = resolve_dataset_context(dataset_dir)

    if args.pairs_out is None:
        pairs_out = derive_pairs_output_path(dataset_dir)
    else:
        pairs_out = args.pairs_out.resolve()

    print(f"[INFO] Dataset: {dataset_dir}")
    print(f"[INFO] Variant root: {context.variant_name}")
    print(f"[INFO] Referenced base root: {context.base_root_name}")
    print(f"[INFO] Output:  {pairs_out}")

    identities = collect_identity_images(dataset_dir)

    dataset_hash = build_dataset_manifest_hash(dataset_dir, identities)

    all_genuine_pairs = generate_genuine_pairs(identities)

    records_img1 = []
    records_img2 = []
    records_labels = []
    records_repeat = []
    records_fold = []

    for repeat_idx in range(args.repeats):
        rng = random.Random(args.seed + repeat_idx)

        genuine_pairs = list(all_genuine_pairs)
        rng.shuffle(genuine_pairs)

        if args.max_genuine is not None:
            genuine_pairs = genuine_pairs[:args.max_genuine]

        impostor_pairs = generate_impostor_pairs(
            identities,
            len(genuine_pairs),
            rng
        )

        combined = genuine_pairs + impostor_pairs
        rng.shuffle(combined)

        img1, img2, labels = pairs_to_arrays(combined)

        fold_rng = random.Random(args.seed + repeat_idx + 100000)
        folds = assign_folds(labels, args.folds, fold_rng)

        records_img1.append(img1)
        records_img2.append(img2)
        records_labels.append(labels)
        records_repeat.append(np.full(len(labels), repeat_idx))
        records_fold.append(folds)

    img1_paths = np.concatenate(records_img1)
    img2_paths = np.concatenate(records_img2)
    labels = np.concatenate(records_labels)
    repeat_ids = np.concatenate(records_repeat)
    fold_ids = np.concatenate(records_fold)

    pairs_out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        pairs_out,
        img1_paths=img1_paths,
        img2_paths=img2_paths,
        labels=labels,
        repeat_ids=repeat_ids,
        fold_ids=fold_ids,
        dataset_hash=dataset_hash,
        seed=args.seed,
        num_repeats=args.repeats,
        num_folds=args.folds,
    )

    metadata = {
        "dataset_dir": str(dataset_dir),
        "variant_name": context.variant_name,
        "base_dataset": context.base_dataset_name,
        "referenced_base_root": context.base_root_name,
        "variant_type": context.variant_type,
        "transform_chain": context.transform_chain,
        "num_transforms": context.num_transforms,
        "pairs_out": str(pairs_out),
        "dataset_hash": dataset_hash,
        "seed": args.seed,
        "repeats": args.repeats,
        "folds": args.folds,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }

    metadata_path = pairs_out.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved NPZ: {pairs_out}")
    print(f"[INFO] Saved JSON: {metadata_path}")


if __name__ == "__main__":
    main()
