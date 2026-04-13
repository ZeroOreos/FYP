#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_NAMES = ["original", "Deepfakes", "FaceSwap", "NeuralTextures", "Face2Face"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tiny FF++-style HDF5 smoke fixture for M2F2-Det using distinct frames."
    )
    parser.add_argument("image_dir", type=Path, help="Directory containing source images.")
    parser.add_argument("output_dir", type=Path, help="Output directory for FF++_*.h5 smoke files.")
    parser.add_argument("--split-a", default="000", help="First original clip key.")
    parser.add_argument("--split-b", default="001", help="Second original clip key.")
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=21,
        help="Number of distinct frames per clip. Keep this small for smoke tests.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square resize target used by the smoke fixture.",
    )
    return parser.parse_args()


def collect_images(image_dir: Path) -> list[Path]:
    paths = [p for p in sorted(image_dir.rglob("*")) if p.is_file() and p.suffix.lower() in VALID_EXTS]
    if not paths:
        raise RuntimeError(f"No input images found in {image_dir}")
    return paths


def load_distinct_frames(paths: list[Path], count: int, image_size: int) -> list[np.ndarray]:
    if len(paths) < count:
        raise RuntimeError(f"Need at least {count} distinct images, found {len(paths)}")
    frames: list[np.ndarray] = []
    for path in paths[:count]:
        image = Image.open(path).convert("RGB").resize((image_size, image_size))
        frames.append(np.asarray(image, dtype=np.uint8))
    return frames


def transform_frames(frames: list[np.ndarray], mode: str) -> np.ndarray:
    output: list[np.ndarray] = []
    for frame in frames:
        image = Image.fromarray(frame)
        if mode == "original":
            transformed = image
        elif mode == "Deepfakes":
            transformed = ImageEnhance.Sharpness(image).enhance(0.8)
        elif mode == "FaceSwap":
            transformed = ImageOps.mirror(image)
        elif mode == "NeuralTextures":
            transformed = ImageEnhance.Color(image).enhance(0.7)
        elif mode == "Face2Face":
            transformed = ImageEnhance.Contrast(image).enhance(1.15)
        else:
            raise ValueError(mode)
        output.append(np.asarray(transformed, dtype=np.uint8))
    return np.stack(output, axis=0)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = collect_images(args.image_dir.resolve())
    required = args.frames_per_clip * 2
    base_frames = load_distinct_frames(all_images, required, args.image_size)
    frames_a = base_frames[: args.frames_per_clip]
    frames_b = base_frames[args.frames_per_clip : required]

    split = [[args.split_a, args.split_b]]
    (output_dir / "test.json").write_text(json.dumps(split) + "\n", encoding="utf-8")

    for dataset_name in DATASET_NAMES:
        out_path = output_dir / f"FF++_{dataset_name}_c40_test_only.h5"
        with h5py.File(out_path, "w") as handle:
            if dataset_name == "original":
                handle.create_dataset(args.split_a, data=transform_frames(frames_a, dataset_name), compression="gzip")
                handle.create_dataset(args.split_b, data=transform_frames(frames_b, dataset_name), compression="gzip")
            else:
                handle.create_dataset(
                    f"{args.split_a}_{args.split_b}",
                    data=transform_frames(frames_a, dataset_name),
                    compression="gzip",
                )
                handle.create_dataset(
                    f"{args.split_b}_{args.split_a}",
                    data=transform_frames(frames_b, dataset_name),
                    compression="gzip",
                )


if __name__ == "__main__":
    main()
