# RotationMisalignment(severity=None, degrees=None, seed=None); apply(image, rng=None) -> same-size rotated image

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import PreprocessingModifier


class RotationMisalignment(PreprocessingModifier):
    name = "rotation_misalignment"
    parameter_table = {
        1: 3,
        2: 6,
        3: 9,
        4: 12,
        5: 15,
    }

    def __init__(self, severity: int | None = None, degrees: float | None = None, seed: int | None = None) -> None:
        super().__init__(severity=severity, seed=seed)
        self.degrees = degrees

    def _apply_pil(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        degrees = float(self.resolve_parameter(self.degrees))
        signed_degrees = degrees if self.degrees is not None else float(rng.choice((-degrees, degrees)))
        return image.rotate(signed_degrees, resample=Image.Resampling.BILINEAR, expand=False)
