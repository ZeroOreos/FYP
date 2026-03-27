# python3 generate.py <dataset_dir> <embeddings_out> [batch_delay] -> embeddings.npz

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

from Utility.runtime import resolve_onnx_providers


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_DELAY = 0.0


def collect_items(dataset_dir: Path) -> List[Tuple[Path, str]]:
    items = []
    for identity_dir in sorted(dataset_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        label = identity_dir.name
        for img_path in sorted(identity_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in VALID_EXTS:
                items.append((img_path, label))
    return items


def build_app(provider: str | None = None) -> tuple[FaceAnalysis, list[str]]:
    providers = resolve_onnx_providers(provider)
    app = FaceAnalysis(
        name="antelopev2",
        providers=providers,
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app, providers


def load_image(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    return img


def extract_embedding(app: FaceAnalysis, img_path: Path) -> np.ndarray:
    img = load_image(img_path)
    faces = app.get(img)

    if not faces:
        raise RuntimeError(f"No face detected in image: {img_path}")

    best_face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    emb = np.asarray(best_face.embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def generate_embeddings(items, app, batch_delay=BATCH_DELAY):
    embeddings = []
    image_paths = []
    labels = []
    skipped = []

    for img_path, label in tqdm(items, desc="Generating embeddings"):
        try:
            emb = extract_embedding(app, img_path)
            embeddings.append(emb)
            image_paths.append(str(img_path))
            labels.append(label)
            
            if batch_delay > 0:
                time.sleep(batch_delay)
        except Exception as e:
            skipped.append((str(img_path), str(e)))

    if not embeddings:
        raise RuntimeError("No valid embeddings were generated. All images failed.")

    embeddings = np.vstack(embeddings).astype(np.float32)
    image_paths = np.array(image_paths, dtype=object)
    labels = np.array(labels, dtype=object)

    return embeddings, image_paths, labels, skipped


def main():
    parser = argparse.ArgumentParser(description="Generate InsightFace embeddings.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("embeddings_out", type=Path)
    parser.add_argument("batch_delay", type=float, nargs="?", default=BATCH_DELAY)
    parser.add_argument("--provider", choices=("auto", "coreml", "cuda", "cpu"), default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    embeddings_out = args.embeddings_out.resolve()
    batch_delay = float(args.batch_delay)

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    embeddings_out.parent.mkdir(parents=True, exist_ok=True)

    items = collect_items(dataset_dir)
    identities = sorted({label for _, label in items})

    print(f"[INFO] Dataset dir:     {dataset_dir}")
    print(f"[INFO] Embeddings out: {embeddings_out}")
    print(f"[INFO] Images found:   {len(items)}")
    print(f"[INFO] Identities:     {len(identities)}")

    app, providers = build_app(args.provider)
    print(f"[INFO] Providers:       {providers}")

    embeddings, image_paths, labels, skipped = generate_embeddings(items, app, batch_delay)

    np.savez_compressed(
        embeddings_out,
        embeddings=embeddings,
        image_paths=image_paths,
        labels=labels,
    )

    print(f"[INFO] Saved embeddings: {embeddings.shape}")

    if skipped:
        skipped_file = embeddings_out.parent / "skipped_images.txt"
        with open(skipped_file, "w", encoding="utf-8") as f:
            for path, err in skipped:
                f.write(f"{path}\t{err}\n")

        skipped_labels = [Path(p).parent.name for p, _ in skipped]
        kept_identities = set(labels.tolist())
        all_identities = set([label for _, label in items])
        lost_identities = sorted(all_identities - kept_identities)

        print(f"[WARN] Skipped images: {len(skipped)}")
        print(f"[WARN] Skip log: {skipped_file}")
        print(f"[WARN] Remaining ids: {len(kept_identities)} / {len(all_identities)}")

        if lost_identities:
            print(f"[WARN] Lost ids with zero valid images: {len(lost_identities)}")
    else:
        print("[INFO] No skipped images.")


if __name__ == "__main__":
    main()
