# ResolutionResampling(severity=None, scale=None, downsample_mode=..., upsample_mode=..., seed=None); apply(image, rng=None) -> same-size degraded image

from __future__ import annotations

from PIL import Image

from .base import PreprocessingModifier


class ResolutionResampling(PreprocessingModifier):
    name = "resolution_resampling"
    parameter_table = {
        1: 0.90,
        2: 0.75,
        3: 0.50,
        4: 0.35,
        5: 0.20,
    }

    def __init__(
        self,
        severity: int | None = None,
        scale: float | None = None,
        downsample_mode: int = Image.Resampling.BILINEAR,
        upsample_mode: int = Image.Resampling.BICUBIC,
        seed: int | None = None,
    ) -> None:
        super().__init__(severity=severity, seed=seed)
        self.scale = scale
        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode

    def _apply_pil(self, image: Image.Image, rng) -> Image.Image:
        scale = float(self.resolve_parameter(self.scale))
        width, height = image.size
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=self.downsample_mode,
        )
        return resized.resize((width, height), resample=self.upsample_mode)
