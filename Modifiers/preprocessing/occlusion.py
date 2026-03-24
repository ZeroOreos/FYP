# FaceMaskOcclusion(...) | EyeBandOcclusion(...) | RandomBlockOcclusion(...); apply(image, rng=None) -> same-size occluded image

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import PreprocessingModifier


class FaceMaskOcclusion(PreprocessingModifier):
    name = "face_mask_occlusion"
    parameter_table = {
        1: 0.30,
        2: 0.40,
        3: 0.50,
        4: 0.60,
        5: 0.70,
    }

    def __init__(
        self,
        severity: int | None = None,
        coverage: float | None = None,
        color: tuple[int, int, int] = (210, 214, 220),
        seed: int | None = None,
    ) -> None:
        super().__init__(severity=severity, seed=seed)
        self.coverage = coverage
        self.color = color

    def _apply_pil(self, image: Image.Image, rng) -> Image.Image:
        coverage = float(self.resolve_parameter(self.coverage))
        array = np.asarray(image.convert("RGB")).copy()
        height, width = array.shape[:2]
        occ_height = max(1, round(height * coverage))
        top = height - occ_height
        array[top:, :] = np.array(self.color, dtype=np.uint8)
        return Image.fromarray(array)


class EyeBandOcclusion(PreprocessingModifier):
    name = "eye_band_occlusion"
    parameter_table = {
        1: 0.10,
        2: 0.14,
        3: 0.18,
        4: 0.22,
        5: 0.26,
    }

    def __init__(
        self,
        severity: int | None = None,
        band_height: float | None = None,
        color: tuple[int, int, int] = (24, 24, 24),
        seed: int | None = None,
    ) -> None:
        super().__init__(severity=severity, seed=seed)
        self.band_height = band_height
        self.color = color

    def _apply_pil(self, image: Image.Image, rng) -> Image.Image:
        band_height = float(self.resolve_parameter(self.band_height))
        array = np.asarray(image.convert("RGB")).copy()
        height = array.shape[0]
        stripe = max(1, round(height * band_height))
        center = round(height * 0.38)
        top = max(0, center - stripe // 2)
        bottom = min(height, top + stripe)
        array[top:bottom, :] = np.array(self.color, dtype=np.uint8)
        return Image.fromarray(array)


class RandomBlockOcclusion(PreprocessingModifier):
    name = "random_block_occlusion"
    parameter_table = {
        1: 0.05,
        2: 0.10,
        3: 0.15,
        4: 0.20,
        5: 0.25,
    }

    def __init__(
        self,
        severity: int | None = None,
        area_ratio: float | None = None,
        color: tuple[int, int, int] | None = None,
        aspect_ratio: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(severity=severity, seed=seed)
        self.area_ratio = area_ratio
        self.color = color
        self.aspect_ratio = aspect_ratio

    def _apply_pil(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        area_ratio = float(self.resolve_parameter(self.area_ratio))
        array = np.asarray(image.convert("RGB")).copy()
        height, width = array.shape[:2]
        block_area = max(1, round(height * width * area_ratio))
        block_width = max(1, min(width, round(np.sqrt(block_area * self.aspect_ratio))))
        block_height = max(1, min(height, round(block_area / block_width)))
        top = int(rng.integers(0, max(1, height - block_height + 1)))
        left = int(rng.integers(0, max(1, width - block_width + 1)))
        color = self.color
        if color is None:
            value = int(rng.integers(0, 256))
            color = (value, value, value)
        array[top : top + block_height, left : left + block_width] = np.array(color, dtype=np.uint8)
        return Image.fromarray(array)
