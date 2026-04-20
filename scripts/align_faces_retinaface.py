#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class AlignmentRecord:
    input_path: str
    output_path: str
    detector: str
    landmarks: list[list[float]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align a face dataset with RetinaFace-class 5-point landmarks.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--size", type=int, default=112, help="Aligned square crop size.")
    parser.add_argument("--backend", choices=("auto", "retinaface", "insightface"), default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Optional max image count for smoke runs.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _iter_images(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in VALID_EXTS]


def _estimate_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / float(src.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    src_var = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = float(np.sum(singular_values) / max(src_var, 1e-8))
    translation = dst_mean - scale * (rotation @ src_mean)
    return np.concatenate([scale * rotation, translation.reshape(2, 1)], axis=1).astype(np.float32)


def _warp_bilinear(image: np.ndarray, transform: np.ndarray, size: int) -> np.ndarray:
    inverse = np.linalg.inv(np.vstack([transform, [0.0, 0.0, 1.0]]))[:2, :]
    yy, xx = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32), indexing="ij")
    source = np.stack([xx, yy, np.ones_like(xx)], axis=0).reshape(3, -1)
    mapped = inverse @ source
    x = mapped[0].reshape(size, size)
    y = mapped[1].reshape(size, size)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, image.shape[1] - 1)
    x1c = np.clip(x1, 0, image.shape[1] - 1)
    y0c = np.clip(y0, 0, image.shape[0] - 1)
    y1c = np.clip(y1, 0, image.shape[0] - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    warped = (
        wa[..., None] * image[y0c, x0c]
        + wb[..., None] * image[y0c, x1c]
        + wc[..., None] * image[y1c, x0c]
        + wd[..., None] * image[y1c, x1c]
    )
    return np.clip(warped, 0, 255).astype(np.uint8)


def _load_io_backend():
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for image loading/saving.") from exc
    return Image


def _build_detector(backend: str):
    errors: list[str] = []
    if backend in {"auto", "retinaface"}:
        try:
            from retinaface import RetinaFace  # type: ignore

            class RetinaFaceDetector:
                name = "retinaface"

                def detect(self, image_bgr: np.ndarray) -> np.ndarray:
                    faces = RetinaFace.detect_faces(image_bgr)
                    if not faces:
                        raise RuntimeError("No face detected.")
                    if isinstance(faces, dict):
                        candidates = list(faces.values())
                    else:
                        candidates = list(faces)
                    best = max(candidates, key=lambda item: float(item.get("score", 0.0)))
                    landmarks = best.get("landmarks")
                    if landmarks is None:
                        raise RuntimeError("RetinaFace did not return 5-point landmarks.")
                    order = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
                    return np.array([landmarks[key] for key in order], dtype=np.float32)

            return RetinaFaceDetector()
        except Exception as exc:
            errors.append(f"retinaface: {exc}")
            if backend == "retinaface":
                raise

    if backend in {"auto", "insightface"}:
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            class InsightFaceDetector:
                name = "insightface"

                def __init__(self) -> None:
                    self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                    self.app.prepare(ctx_id=0, det_size=(640, 640))

                def detect(self, image_bgr: np.ndarray) -> np.ndarray:
                    faces = self.app.get(image_bgr)
                    if not faces:
                        raise RuntimeError("No face detected.")
                    best = max(faces, key=lambda face: float(face.det_score))
                    return np.asarray(best.kps, dtype=np.float32)

            return InsightFaceDetector()
        except Exception as exc:
            errors.append(f"insightface: {exc}")
            if backend == "insightface":
                raise

    joined = "; ".join(errors) if errors else "no backend attempted"
    raise RuntimeError(f"Could not initialize a RetinaFace-class detector ({joined}).")


def _scaled_template(size: int) -> np.ndarray:
    return ARCFACE_TEMPLATE_112 * (float(size) / 112.0)


def main() -> None:
    args = _parse_args()
    Image = _load_io_backend()
    detector = _build_detector(args.backend)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _iter_images(input_dir)
    if args.limit is not None:
        image_paths = image_paths[: max(0, args.limit)]
    if not image_paths:
        raise SystemExit(f"No images found under {input_dir}")

    template = _scaled_template(args.size)
    records: list[dict[str, object]] = []
    for image_path in image_paths:
        rel_path = image_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            continue

        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image)
        bgr = rgb[:, :, ::-1]
        landmarks = detector.detect(bgr)
        transform = _estimate_similarity(landmarks, template)
        aligned = _warp_bilinear(rgb, transform, args.size)
        Image.fromarray(aligned).save(output_path)
        records.append(
            asdict(
                AlignmentRecord(
                    input_path=str(image_path),
                    output_path=str(output_path),
                    detector=detector.name,
                    landmarks=[[float(value) for value in point] for point in landmarks.tolist()],
                )
            )
        )

    manifest_path = output_dir / "alignment_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "backend": detector.name,
                "aligned_images": len(records),
                "records": records,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"[INFO] Aligned {len(records)} images into {output_dir}")


if __name__ == "__main__":
    main()
