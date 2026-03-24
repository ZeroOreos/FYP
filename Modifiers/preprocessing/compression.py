# JPEGCompression(severity=None, quality=None, seed=None); apply(image, rng=None) -> same-size JPEG-degraded image

from __future__ import annotations

from io import BytesIO

from PIL import Image

from .base import PreprocessingModifier


class JPEGCompression(PreprocessingModifier):
    name = "jpeg_compression"
    parameter_table = {
        1: 90,
        2: 70,
        3: 50,
        4: 30,
        5: 15,
    }

    def __init__(self, severity: int | None = None, quality: int | None = None, seed: int | None = None) -> None:
        super().__init__(severity=severity, seed=seed)
        self.quality = quality

    def _apply_pil(self, image: Image.Image, rng) -> Image.Image:
        quality = int(self.resolve_parameter(self.quality))
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=False)
        buffer.seek(0)
        return Image.open(buffer).copy()
