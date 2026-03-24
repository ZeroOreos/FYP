# python3 attack_verification.py <gallery_embeddings> <probe_embeddings> <attack_manifest> <metrics_out> [--pairs-file <pairs>] -> attack metrics

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


FAR_TARGETS = (0.1, 0.01, 0.001)


def _load_embeddings(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    data = np.load(path, allow_pickle=True)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    image_paths = [str(value) for value in data["image_paths"].tolist()]
    labels = [str(value) for value in data["labels"].tolist()]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    embeddings = embeddings / norms
    return embeddings, image_paths, labels


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        if "records" in data:
            return list(data["records"])
        if "records_preview" in data:
            return list(data["records_preview"])
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported attack manifest format: {path}")


def _identity_centroids(embeddings: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(label, []).append(idx)

    centroids: dict[str, np.ndarray] = {}
    for label, indices in grouped.items():
        centroid = embeddings[indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-12:
            centroid = centroid / norm
        centroids[label] = centroid.astype(np.float32)
    return centroids


def _embedding_map(image_paths: list[str], embeddings: np.ndarray) -> dict[str, np.ndarray]:
    return {path: embeddings[idx] for idx, path in enumerate(image_paths)}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a.astype(np.float64) * b.astype(np.float64), dtype=np.float64))


def _load_thresholds_from_pairs(pairs_file: Optional[Path], embedding_map: dict[str, np.ndarray]) -> dict[str, float]:
    if pairs_file is None or not pairs_file.exists():
        return {}

    data = np.load(pairs_file, allow_pickle=True)
    img1_paths = [str(value) for value in data["img1_paths"].tolist()]
    img2_paths = [str(value) for value in data["img2_paths"].tolist()]
    labels = np.asarray(data["labels"], dtype=np.int32)

    scores: list[float] = []
    pair_labels: list[int] = []
    for img1, img2, label in zip(img1_paths, img2_paths, labels):
        emb1 = embedding_map.get(img1)
        emb2 = embedding_map.get(img2)
        if emb1 is None or emb2 is None:
            continue
        scores.append(_cosine(emb1, emb2))
        pair_labels.append(int(label))

    if not scores:
        return {}

    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(pair_labels, dtype=np.int32)
    genuine = scores_array[labels_array == 1]
    impostor = scores_array[labels_array == 0]
    thresholds = np.unique(scores_array)

    best_accuracy = -1.0
    best_threshold = float(thresholds[0])
    for threshold in thresholds:
        preds = scores_array >= threshold
        accuracy = float((preds == labels_array).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    result = {"best_threshold": best_threshold}
    for far_target in FAR_TARGETS:
        eligible = thresholds[(impostor >= thresholds[:, None]).mean(axis=1) <= far_target]
        threshold = float(eligible[0]) if len(eligible) else float(thresholds.max())
        result[f"far_{str(far_target).replace('.', 'p')}"] = threshold
    return result


def evaluate_attack(
    gallery_embeddings: Path,
    probe_embeddings: Path,
    attack_manifest: Path,
    metrics_out: Path,
    pairs_file: Optional[Path],
) -> None:
    start = time.time()
    gallery_emb, gallery_paths, gallery_labels = _load_embeddings(gallery_embeddings)
    probe_emb, probe_paths, _ = _load_embeddings(probe_embeddings)
    manifest = _load_manifest(attack_manifest)

    gallery_map = _embedding_map(gallery_paths, gallery_emb)
    probe_map = _embedding_map(probe_paths, probe_emb)
    centroids = _identity_centroids(gallery_emb, gallery_labels)
    thresholds = _load_thresholds_from_pairs(pairs_file, gallery_map)
    default_threshold = thresholds.get("best_threshold", 0.3)

    scored_records: list[dict[str, Any]] = []
    skipped_missing_probe = 0
    skipped_missing_victim = 0
    for record in manifest:
        output_image = str(record.get("output_image", ""))
        probe_vector = probe_map.get(output_image)
        victim_identity = str(record.get("victim_identity") or record.get("claimed_identity") or "")
        attacker_identity = str(record.get("attacker_identity") or "")
        victim_centroid = centroids.get(victim_identity)
        attacker_centroid = centroids.get(attacker_identity)
        if probe_vector is None:
            skipped_missing_probe += 1
            continue
        if victim_centroid is None:
            skipped_missing_victim += 1
            continue

        victim_score = _cosine(probe_vector, victim_centroid)
        attacker_score = _cosine(probe_vector, attacker_centroid) if attacker_centroid is not None else None
        scored_records.append({
            "output_image": output_image,
            "victim_identity": victim_identity,
            "attacker_identity": attacker_identity,
            "victim_score": victim_score,
            "attacker_score": attacker_score,
        })

    victim_scores = np.asarray([row["victim_score"] for row in scored_records], dtype=np.float64)
    attacker_scores = np.asarray(
        [row["attacker_score"] for row in scored_records if row["attacker_score"] is not None],
        dtype=np.float64,
    )

    metrics: dict[str, Any] = {
        "evaluation_mode": "attack",
        "gallery_embeddings": str(gallery_embeddings),
        "probe_embeddings": str(probe_embeddings),
        "attack_manifest": str(attack_manifest),
        "pair_file": str(pairs_file) if pairs_file is not None else "",
        "num_manifest_records": len(manifest),
        "num_attack_samples": len(scored_records),
        "skipped_missing_probe_embeddings": skipped_missing_probe,
        "skipped_missing_victim_centroid": skipped_missing_victim,
        "attack_success_rate": float((victim_scores >= default_threshold).mean()) if len(victim_scores) else 0.0,
        "victim_accept_rate": float((victim_scores >= default_threshold).mean()) if len(victim_scores) else 0.0,
        "attacker_accept_rate": float((attacker_scores >= default_threshold).mean()) if len(attacker_scores) else 0.0,
        "best_threshold": default_threshold,
        "runtime_seconds": time.time() - start,
        "thresholds": thresholds,
    }

    for far_target in FAR_TARGETS:
        key = f"far_{str(far_target).replace('.', 'p')}"
        threshold = thresholds.get(key)
        if threshold is not None:
            metrics[f"attack_success_rate_at_far_{str(far_target).replace('.', 'p')}"] = float(
                (victim_scores >= threshold).mean()
            ) if len(victim_scores) else 0.0

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate attack probes against clean gallery embeddings.")
    parser.add_argument("gallery_embeddings", type=str)
    parser.add_argument("probe_embeddings", type=str)
    parser.add_argument("attack_manifest", type=str)
    parser.add_argument("metrics_out", type=str)
    parser.add_argument("--pairs-file", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_attack(
        gallery_embeddings=Path(args.gallery_embeddings).resolve(),
        probe_embeddings=Path(args.probe_embeddings).resolve(),
        attack_manifest=Path(args.attack_manifest).resolve(),
        metrics_out=Path(args.metrics_out).resolve(),
        pairs_file=Path(args.pairs_file).resolve() if args.pairs_file else None,
    )


if __name__ == "__main__":
    main()
