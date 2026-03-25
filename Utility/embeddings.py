#!/usr/bin/env python3
# Shared embedding loading and lookup helpers for evaluation pipelines.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


EMBEDDING_KEYS = ("embeddings", "embedding", "embs", "x", "features", "feats")
IMAGE_PATH_KEYS = ("image_paths", "paths", "img_paths", "filenames", "files")
LABEL_KEYS = ("labels", "label", "ids", "identities")


def l2_normalize(values: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.clip(norms, eps, None)


def load_embeddings(path: Path, *, require_labels: bool = False) -> tuple[np.ndarray, list[str], Optional[list[str]]]:
    data = np.load(path, allow_pickle=True)

    embeddings = next((np.asarray(data[key], dtype=np.float32) for key in EMBEDDING_KEYS if key in data.files), None)
    if embeddings is None:
        raise KeyError(f"No embeddings key found. Keys: {list(data.files)}")

    raw_paths = next((data[key] for key in IMAGE_PATH_KEYS if key in data.files), None)
    if raw_paths is None:
        raise KeyError(
            "No image path key found in embeddings file. "
            "Expected one of: image_paths, paths, img_paths, filenames, files. "
            f"Keys: {list(data.files)}"
        )
    image_paths = [str(Path(value).resolve()) for value in raw_paths.tolist()]

    if len(embeddings) != len(image_paths):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) does not match image path count ({len(image_paths)})"
        )

    raw_labels = next((data[key] for key in LABEL_KEYS if key in data.files), None)
    if raw_labels is None:
        if require_labels:
            raise KeyError(f"No labels key found in embeddings file. Keys: {list(data.files)}")
        labels = None
    else:
        labels = [str(value) for value in raw_labels.tolist()]
        if len(labels) != len(image_paths):
            raise ValueError(
                f"Label count ({len(labels)}) does not match image path count ({len(image_paths)})"
            )

    return l2_normalize(embeddings), image_paths, labels


def build_path_to_index(image_paths: list[str]) -> tuple[dict[str, int], int]:
    path_to_index: dict[str, int] = {}
    duplicates = 0
    for index, image_path in enumerate(image_paths):
        if image_path in path_to_index:
            duplicates += 1
        path_to_index[image_path] = index
    return path_to_index, duplicates


def build_embedding_map(image_paths: list[str], embeddings: np.ndarray) -> dict[str, np.ndarray]:
    return {path: embeddings[index] for index, path in enumerate(image_paths)}
