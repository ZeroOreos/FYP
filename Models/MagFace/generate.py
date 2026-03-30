# python3 generate.py <dataset_dir> <embeddings_out> [batch_size] [batch_delay] -> embeddings.npz

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_PROJECT_ROOT))

from Utility.runtime import PROJECT_ROOT, resolve_torch_device


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 112
BATCH_SIZE = 64
BATCH_DELAY = 0.0
DEFAULT_ARCH = "iresnet100"
DEFAULT_EMBEDDING_SIZE = 512


def collect_images(dataset_dir: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for identity_dir in sorted(dataset_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        for img_path in sorted(identity_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in VALID_EXTS:
                items.append((img_path.resolve(), identity_dir.name))
    if not items:
        raise RuntimeError(f"No valid images found in dataset: {dataset_dir}")
    return items


def resolve_source_root(explicit: Path | None) -> Path:
    if explicit is not None:
        source_root = explicit.resolve()
    else:
        source_root = Path(
            os.environ.get(
                "FYP_MAGFACE_SOURCE_ROOT",
                PROJECT_ROOT / "Backends" / "sources" / "MagFace_upstream",
            )
        ).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"MagFace source root not found: {source_root}")
    return source_root


def resolve_checkpoint(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    asset_root = Path(
        os.environ.get(
            "FYP_MAGFACE_ASSET_ROOT",
            PROJECT_ROOT / "Backends" / "assets" / "recognition" / "magface",
        )
    ).resolve()
    return asset_root / "pretrained" / "magface_iresnet100_ms1mv2.pth"


def import_upstream_iresnet(source_root: Path):
    sys.path.insert(0, str(source_root))
    try:
        from models import iresnet  # type: ignore
    finally:
        sys.path.pop(0)
    return iresnet


def build_model(arch: str, embedding_size: int, source_root: Path):
    iresnet = import_upstream_iresnet(source_root)
    builders = {
        "iresnet18": iresnet.iresnet18,
        "iresnet34": iresnet.iresnet34,
        "iresnet50": iresnet.iresnet50,
        "iresnet100": iresnet.iresnet100,
    }
    if arch not in builders:
        raise ValueError(f"Unsupported MagFace architecture: {arch}")
    return builders[arch](pretrained=False, num_classes=embedding_size)


def load_model(
    *,
    source_root: Path,
    architecture: str,
    embedding_size: int,
    checkpoint_path: Path,
    allow_random_init: bool,
    device: torch.device,
):
    model = build_model(architecture, embedding_size, source_root)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {}
        model_keys = model.state_dict()
        for key, value in state_dict.items():
            candidates = [
                key.removeprefix("features.module."),
                key.removeprefix("features."),
                key.removeprefix("module."),
                ".".join(key.split(".")[1:]),
                "features." + ".".join(key.split(".")[2:]),
                key,
            ]
            for candidate in candidates:
                if candidate in model_keys and value.shape == model_keys[candidate].shape:
                    cleaned[candidate] = value
                    break
        if len(cleaned) != len(model_keys):
            missing = len(model_keys) - len(cleaned)
            raise RuntimeError(
                f"MagFace checkpoint did not cover the full model state: missing {missing} tensors."
            )
        model.load_state_dict(cleaned, strict=True)
        checkpoint_used = checkpoint_path
        random_init = False
    else:
        if not allow_random_init:
            raise FileNotFoundError(
                f"MagFace checkpoint not found: {checkpoint_path}. "
                "Pass --allow-random-init only for smoke testing."
            )
        checkpoint_used = None
        random_init = True

    model.eval().to(device)
    return model, checkpoint_used, random_init


def preprocess_image(img_path: Path) -> torch.Tensor:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float() / 255.0


def generate_embeddings(
    items: list[tuple[Path, str]],
    model,
    device: torch.device,
    batch_size: int,
    batch_delay: float,
):
    embeddings_list = []
    image_paths: list[str] = []
    labels: list[str] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(items), batch_size), desc="Generating embeddings"):
            batch_items = items[start:start + batch_size]
            batch = torch.stack([preprocess_image(path) for path, _ in batch_items]).to(device)
            features = F.normalize(model(batch), dim=1)
            embeddings_list.append(features.cpu().numpy().astype(np.float32))
            image_paths.extend(str(path) for path, _ in batch_items)
            labels.extend(label for _, label in batch_items)
            if batch_delay > 0:
                time.sleep(batch_delay)

    embeddings = np.concatenate(embeddings_list, axis=0)
    return embeddings, np.asarray(image_paths, dtype=str), np.asarray(labels, dtype=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MagFace embeddings.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("embeddings_out", type=Path)
    parser.add_argument("batch_size", type=int, nargs="?", default=BATCH_SIZE)
    parser.add_argument("batch_delay", type=float, nargs="?", default=BATCH_DELAY)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=None)
    parser.add_argument("--architecture", default=DEFAULT_ARCH)
    parser.add_argument("--embedding-size", type=int, default=DEFAULT_EMBEDDING_SIZE)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    embeddings_out = args.embeddings_out.resolve()
    source_root = resolve_source_root(args.source_root)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    device = torch.device(resolve_torch_device(args.device))

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    items = collect_images(dataset_dir)
    model, checkpoint_used, random_init = load_model(
        source_root=source_root,
        architecture=args.architecture,
        embedding_size=int(args.embedding_size),
        checkpoint_path=checkpoint_path,
        allow_random_init=args.allow_random_init,
        device=device,
    )

    print(f"[INFO] Dataset dir:     {dataset_dir}")
    print(f"[INFO] Embeddings out: {embeddings_out}")
    print(f"[INFO] Device:         {device}")
    print(f"[INFO] Source root:    {source_root}")
    print(f"[INFO] Architecture:   {args.architecture}")
    print(f"[INFO] Images found:   {len(items)}")
    print(f"[INFO] Identities:     {len({label for _, label in items})}")
    if checkpoint_used is not None:
        print(f"[INFO] Checkpoint:     {checkpoint_used}")
    elif random_init:
        print("[WARN] Checkpoint missing. Using random initialization for smoke test only.")

    embeddings, image_paths, labels = generate_embeddings(
        items=items,
        model=model,
        device=device,
        batch_size=int(args.batch_size),
        batch_delay=float(args.batch_delay),
    )

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
