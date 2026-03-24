#!/usr/bin/env python3
# python3 Utility/attack_pair.py <embeddings_src> [options] -> <variant>_<model>_atkpairs.npz/.json beside embeddings

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "Results"


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return array / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    sims = np.sum(left[:, None, :] * right[None, :, :], axis=2, dtype=np.float64)
    return np.clip(sims, -1.0, 1.0)


def _to_text_list(values: np.ndarray) -> list[str]:
    return [str(value) for value in values.tolist()]


def resolve_embeddings_path(src: Path) -> Path:
    if src.is_file():
        return src
    candidate = src / "embeddings.npz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"embeddings file not found from source: {src}")


def derive_output_prefix(embeddings_path: Path) -> tuple[Path, str]:
    if embeddings_path.parent.parent == RESULTS_ROOT or RESULTS_ROOT in embeddings_path.parents:
        model_name = embeddings_path.parent.name
        variant_name = embeddings_path.parent.parent.name
        prefix = f"{variant_name}_{model_name}"
    else:
        prefix = embeddings_path.stem
    return embeddings_path.parent, prefix


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = {"embeddings", "image_paths", "labels"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"missing keys in embeddings file: {sorted(missing)}")
    return {
        "embeddings": np.asarray(data["embeddings"], dtype=np.float32),
        "image_paths": np.asarray(data["image_paths"], dtype=object),
        "labels": np.asarray(data["labels"], dtype=object),
    }


def build_identity_index(labels: list[str], image_paths: list[str], embeddings: np.ndarray) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(label, {"indices": []})
        grouped[label]["indices"].append(idx)

    for label, payload in grouped.items():
        indices = payload["indices"]
        id_embeddings = embeddings[indices]
        centroid = _normalize_vector(id_embeddings.mean(axis=0))
        payload["centroid"] = centroid
        payload["images"] = [image_paths[i] for i in indices]
        payload["image_embeddings"] = embeddings[indices]
    return grouped


def filter_finite_embeddings(
    embeddings: np.ndarray,
    image_paths: list[str],
    labels: list[str],
) -> tuple[np.ndarray, list[str], list[str], int]:
    finite_mask = np.isfinite(embeddings).all(axis=1)
    filtered_embeddings = embeddings[finite_mask]
    filtered_image_paths = [path for idx, path in enumerate(image_paths) if bool(finite_mask[idx])]
    filtered_labels = [label for idx, label in enumerate(labels) if bool(finite_mask[idx])]
    removed = int((~finite_mask).sum())
    return filtered_embeddings, filtered_image_paths, filtered_labels, removed


def build_attack_pairs(
    identity_index: dict[str, dict[str, Any]],
    top_k: int,
    samples_per_identity_pair: int,
    pairing_mode: str,
    min_identity_sim: Optional[float],
    min_image_sim: Optional[float],
) -> list[dict[str, Any]]:
    identities = sorted(identity_index.keys())
    centroids = np.stack([identity_index[label]["centroid"] for label in identities], axis=0).astype(np.float64)
    centroid_sims = _cosine_matrix(centroids, centroids)

    records: list[dict[str, Any]] = []
    for victim_idx, victim_label in enumerate(identities):
        sims = centroid_sims[victim_idx].copy()
        sims[victim_idx] = -np.inf
        ranked_indices = np.argsort(sims)[::-1]
        ranked_indices = [idx for idx in ranked_indices if np.isfinite(sims[idx])]
        if min_identity_sim is not None:
            ranked_indices = [idx for idx in ranked_indices if float(sims[idx]) >= min_identity_sim]
        if pairing_mode == "semi_hard":
            ranked_indices = ranked_indices[: max(top_k * 3, top_k)]
        else:
            ranked_indices = ranked_indices[:top_k]

        if pairing_mode == "semi_hard" and len(ranked_indices) > top_k:
            rng = np.random.default_rng(42 + victim_idx)
            ranked_indices = sorted(rng.choice(ranked_indices, size=top_k, replace=False).tolist())

        victim_payload = identity_index[victim_label]
        victim_embeddings = victim_payload["image_embeddings"]
        victim_images = victim_payload["images"]

        for attacker_idx in ranked_indices:
            attacker_label = identities[attacker_idx]
            attacker_payload = identity_index[attacker_label]
            image_sims = _cosine_matrix(
                attacker_payload["image_embeddings"].astype(np.float64),
                victim_embeddings.astype(np.float64),
            )
            flat_order = np.argsort(image_sims, axis=None)[::-1]
            used_pairs = 0
            for flat_idx in flat_order:
                source_row, target_col = np.unravel_index(flat_idx, image_sims.shape)
                image_similarity = float(image_sims[source_row, target_col])
                if min_image_sim is not None and image_similarity < min_image_sim:
                    continue
                records.append({
                    "victim_identity": victim_label,
                    "attacker_identity": attacker_label,
                    "victim_image": victim_images[target_col],
                    "attacker_image": attacker_payload["images"][source_row],
                    "identity_similarity": float(centroid_sims[victim_idx, attacker_idx]),
                    "image_similarity": image_similarity,
                    "claim_identity": victim_label,
                    "pairing_mode": pairing_mode,
                })
                used_pairs += 1
                if used_pairs >= samples_per_identity_pair:
                    break
    return records


def save_attack_pairs(
    records: list[dict[str, Any]],
    out_npz: Path,
    out_json: Path,
    metadata: dict[str, Any],
) -> None:
    arrays = {
        "victim_identity": np.array([row["victim_identity"] for row in records], dtype=object),
        "attacker_identity": np.array([row["attacker_identity"] for row in records], dtype=object),
        "victim_image": np.array([row["victim_image"] for row in records], dtype=object),
        "attacker_image": np.array([row["attacker_image"] for row in records], dtype=object),
        "claim_identity": np.array([row["claim_identity"] for row in records], dtype=object),
        "identity_similarity": np.array([row["identity_similarity"] for row in records], dtype=np.float32),
        "image_similarity": np.array([row["image_similarity"] for row in records], dtype=np.float32),
        "pairing_mode": np.array([row["pairing_mode"] for row in records], dtype=object),
    }
    np.savez_compressed(out_npz, **arrays)

    metadata["num_attack_pairs"] = len(records)
    metadata["records_preview"] = records[: min(10, len(records))]
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hard or semi-hard face-swap attack pair candidates from embeddings.")
    parser.add_argument("embeddings_src", type=str, help="Embeddings .npz path or result directory containing embeddings.npz")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--samples-per-identity-pair", type=int, default=3)
    parser.add_argument("--pairing-mode", choices=("hard", "semi_hard"), default="hard")
    parser.add_argument("--min-identity-sim", type=float, default=None)
    parser.add_argument("--min-image-sim", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings_path = resolve_embeddings_path(Path(args.embeddings_src).resolve())
    output_dir, prefix = derive_output_prefix(embeddings_path)
    out_npz = output_dir / f"{prefix}_atkpairs.npz"
    out_json = output_dir / f"{prefix}_atkpairs.json"

    if out_npz.exists() or out_json.exists():
        print(f"[WARN] attack pair output already exists, skipping: {out_npz}")
        return

    loaded = load_embeddings(embeddings_path)
    embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
    image_paths = _to_text_list(loaded["image_paths"])
    labels = _to_text_list(loaded["labels"])
    embeddings, image_paths, labels, removed_nonfinite = filter_finite_embeddings(embeddings, image_paths, labels)
    embeddings = _normalize_rows(embeddings)

    identity_index = build_identity_index(labels, image_paths, embeddings)
    records = build_attack_pairs(
        identity_index=identity_index,
        top_k=max(1, args.top_k),
        samples_per_identity_pair=max(1, args.samples_per_identity_pair),
        pairing_mode=args.pairing_mode,
        min_identity_sim=args.min_identity_sim,
        min_image_sim=args.min_image_sim,
    )

    metadata = {
        "embeddings_file": str(embeddings_path),
        "output_npz": str(out_npz),
        "output_json": str(out_json),
        "top_k": args.top_k,
        "samples_per_identity_pair": args.samples_per_identity_pair,
        "pairing_mode": args.pairing_mode,
        "min_identity_sim": args.min_identity_sim,
        "min_image_sim": args.min_image_sim,
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "num_identities": len(identity_index),
        "removed_nonfinite_embeddings": removed_nonfinite,
        "pairing_basis": "identity centroid cosine similarity + image cosine similarity",
    }
    save_attack_pairs(records, out_npz, out_json, metadata)

    print(f"[INFO] embeddings: {embeddings_path}")
    print(f"[INFO] attack pairs: {len(records)}")
    print(f"[INFO] npz: {out_npz}")
    print(f"[INFO] json: {out_json}")


if __name__ == "__main__":
    main()
