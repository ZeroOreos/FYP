# PreprocessingModifier(severity=None, seed=None); apply(image: PIL.Image | np.ndarray, rng=None) -> PIL.Image | np.ndarray with same type and size

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image


class PreprocessingModifier(ABC):
    name: str = "modifier"
    severity_levels = (1, 2, 3, 4, 5)
    parameter_table: dict[int, Any] = {}

    def __init__(self, severity: int | None = None, seed: int | None = None) -> None:
        self.severity = severity
        self.seed = seed
        if severity is not None:
            self._validate_severity(severity)

    def __call__(self, image: Image.Image | np.ndarray, rng: Any = None) -> Image.Image | np.ndarray:
        return self.apply(image, rng=rng)

    def apply(self, image: Image.Image | np.ndarray, rng: Any = None) -> Image.Image | np.ndarray:
        image_type = type(image)
        pil_image = self._to_pil(image)
        out = self._apply_pil(pil_image, rng=self._resolve_rng(rng))
        return self._restore_type(out, image_type)

    @abstractmethod
    def _apply_pil(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        raise NotImplementedError

    def resolve_level(self) -> int:
        if self.severity is None:
            raise ValueError(f"{self.name} needs a severity or an explicit parameter override")
        self._validate_severity(self.severity)
        return self.severity

    def resolve_parameter(self, explicit: Any = None) -> Any:
        if explicit is not None:
            return explicit
        return self.parameter_table[self.resolve_level()]

    def _resolve_rng(self, rng: Any = None) -> np.random.Generator:
        if isinstance(rng, np.random.Generator):
            return rng
        if rng is not None:
            return np.random.default_rng(rng)
        return np.random.default_rng(self.seed)

    def _validate_severity(self, severity: int) -> None:
        if severity not in self.severity_levels:
            raise ValueError(f"severity must be one of {self.severity_levels}, got {severity}")

    @staticmethod
    def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.copy()
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return Image.fromarray(image)
        raise TypeError(f"unsupported image type: {type(image)!r}")

    @staticmethod
    def _restore_type(image: Image.Image, image_type: type[Any]) -> Image.Image | np.ndarray:
        if issubclass(image_type, Image.Image):
            return image
        if issubclass(image_type, np.ndarray):
            return np.asarray(image)
        return image
