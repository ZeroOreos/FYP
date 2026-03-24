# BrightnessShift(severity=None, factor=None, seed=None) | ContrastShift(severity=None, factor=None, seed=None) | GammaShift(severity=None, gamma=None, seed=None); apply(image, rng=None) -> same-size illumination-shifted image

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import PreprocessingModifier


class BrightnessShift(PreprocessingModifier):
    name = "brightness_shift"
    parameter_table = {
        1: 0.85,
        2: 0.70,
        3: 0.55,
        4: 0.40,
        5: 0.25,
    }

    def __init__(self, severity: int | None = None, factor: float | None = None, seed: int | None = None) -> None:
        super().__init__(severity=severity, seed=seed)
        self.factor = factor

    def _apply_pil(self, image: Image.Image, rng) -> Image.Image:
        factor = float(self.resolve_parameter(self.factor))
        array = np.asarray(image).astype(np.float32) * factor
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


class ContrastShift(PreprocessingModifier):
    name = "contrast_shift"
    parameter_table = {
        1: 0.90,
        2: 0.75,
        3: 0.60,
        4: 0.45,
        5: 0.30,
    }

    def __init__(self, severity: int | None = None, factor: float | None = None, seed: int | None = None) -> None:
        super().__init__(severity=severity, seed=seed)
        self.factor = factor

    def _apply_pil(self, image: Image.Image, rng) -> Image.Image:
        factor = float(self.resolve_parameter(self.factor))
        array = np.asarray(image).astype(np.float32)
        mean = array.mean(axis=(0, 1), keepdims=True)
        adjusted = (array - mean) * factor + mean
        return Image.fromarray(np.clip(adjusted, 0, 255).astype(np.uint8))


class GammaShift(PreprocessingModifier):
    name = "gamma_shift"
    parameter_table = {
        1: 0.8,
        2: 0.65,
        3: 0.5,
        4: 1.4,
        5: 1.8,
    }

    def __init__(self, severity: int | None = None, gamma: float | None = None, seed: int | None = None) -> None:
        super().__init__(severity=severity, seed=seed)
        self.gamma = gamma

    def _apply_pil(self, image: Image.Image, rng) -> Image.Image:
        gamma = float(self.resolve_parameter(self.gamma))
        array = np.asarray(image).astype(np.float32) / 255.0
        adjusted = np.power(array, gamma)
        return Image.fromarray(np.clip(adjusted * 255.0, 0, 255).astype(np.uint8))
