# python3 generate.py <dataset_dir> <embeddings_out> [batch_size] [batch_delay] -> embeddings.npz

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 160
BATCH_SIZE = 64
BATCH_DELAY = 0.0


def collect_images(dataset_dir: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []

    for id_dir in sorted(dataset_dir.iterdir()):
        if not id_dir.is_dir():
            continue

        identity = id_dir.name

        for img_path in sorted(id_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in VALID_EXTS:
                items.append((img_path.resolve(), identity))

    if not items:
        raise RuntimeError(f"No valid images found in dataset: {dataset_dir}")

    return items


def preprocess_image(img_path: Path) -> torch.Tensor:
    """Load one image and convert it to FaceNet input tensor."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    arr = np.asarray(img, dtype=np.float32)
    arr = (arr / 255.0 - 0.5) / 0.5

    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def build_model(device: torch.device) -> InceptionResnetV1:
    return InceptionResnetV1(pretrained="vggface2").eval().to(device)


def generate_embeddings(
    items: List[Tuple[Path, str]],
    model: InceptionResnetV1,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    batch_delay: float = BATCH_DELAY,
):
    embeddings_list = []
    image_paths = []
    labels = []

    with torch.no_grad():
        for start in tqdm(range(0, len(items), batch_size), desc="Generating embeddings"):
            batch_items = items[start:start + batch_size]

            batch_tensors = []
            for img_path, identity in batch_items:
                tensor = preprocess_image(img_path)
                batch_tensors.append(tensor)
                image_paths.append(str(img_path))
                labels.append(identity)

            batch = torch.stack(batch_tensors).to(device)
            embs = model(batch).cpu().numpy().astype(np.float32)

            embeddings_list.append(embs)
            
            if batch_delay > 0:
                time.sleep(batch_delay)

    embeddings = np.concatenate(embeddings_list, axis=0)

    return (
        embeddings,
        np.asarray(image_paths, dtype=str),
        np.asarray(labels, dtype=str),
    )


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 FaceNet/generate.py <dataset_dir> <embeddings_out> [batch_size] [batch_delay]")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1]).resolve()
    embeddings_out = Path(sys.argv[2]).resolve()
    
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else BATCH_SIZE
    batch_delay = float(sys.argv[4]) if len(sys.argv) > 4 else BATCH_DELAY

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Dataset dir:     {dataset_dir}")
    print(f"[INFO] Embeddings out: {embeddings_out}")
    print(f"[INFO] Device:         {device}")

    items = collect_images(dataset_dir)
    print(f"[INFO] Images found:   {len(items)}")
    print(f"[INFO] Identities:     {len(set(label for _, label in items))}")

    model = build_model(device)

    embeddings, image_paths, labels = generate_embeddings(
        items=items,
        model=model,
        device=device,
        batch_size=batch_size,
        batch_delay=batch_delay,
    )

    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    print(f"[INFO] Image paths:      {image_paths.shape}")
    print(f"[INFO] Labels:           {labels.shape}")

    embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        embeddings_out,
        embeddings=embeddings,
        image_paths=image_paths,
        labels=labels,
    )

    print(f"[INFO] Saved embeddings to: {embeddings_out}")


if __name__ == "__main__":
    main()
