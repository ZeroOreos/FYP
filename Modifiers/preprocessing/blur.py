# GaussianBlur(...) | MotionBlur(...); apply(image, rng=None) -> same-size blurred image

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

from .base import PreprocessingModifier


class GaussianBlur(PreprocessingModifier):
    name = "gaussian_blur"
    parameter_table = {
        1: 0.5,
        2: 1.0,
        3: 2.0,
        4: 3.0,
        5: 4.0,
    }

    def __init__(self, severity: int | None = None, sigma: float | None = None, seed: int | None = None) -> None:
        super().__init__(severity=severity, seed=seed)
        self.sigma = sigma

    def _apply_pil(self, image: Image.Image, rng) -> Image.Image:
        sigma = float(self.resolve_parameter(self.sigma))
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))


class MotionBlur(PreprocessingModifier):
    name = "motion_blur"
    parameter_table = {
        1: 3,
        2: 5,
        3: 9,
        4: 15,
        5: 21,
    }

    def __init__(
        self,
        severity: int | None = None,
        length: int | None = None,
        angle: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(severity=severity, seed=seed)
        self.length = length
        self.angle = angle

    def _apply_pil(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        length = int(self.resolve_parameter(self.length))
        angle = float(self.angle if self.angle is not None else rng.uniform(0.0, 180.0))
        kernel = self._motion_kernel(length=length, angle=angle)
        array = np.asarray(image).astype(np.float32)
        blurred = self._convolve(array, kernel)
        return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))

    @staticmethod
    def _motion_kernel(length: int, angle: float) -> np.ndarray:
        length = max(3, int(length))
        if length % 2 == 0:
            length += 1

        center = length // 2
        kernel = np.zeros((length, length), dtype=np.float32)
        radians = math.radians(angle)
        dx = math.cos(radians)
        dy = math.sin(radians)

        for step in np.linspace(-center, center, num=length):
            x = int(round(center + step * dx))
            y = int(round(center + step * dy))
            kernel[y, x] = 1.0

        kernel_sum = kernel.sum()
        if kernel_sum == 0:
            kernel[center, center] = 1.0
            kernel_sum = 1.0
        return kernel / kernel_sum

    @staticmethod
    def _convolve(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        if array.ndim == 2:
            return MotionBlur._convolve_channel(array, kernel)

        channels = [MotionBlur._convolve_channel(array[..., idx], kernel) for idx in range(array.shape[2])]
        return np.stack(channels, axis=-1)

    @staticmethod
    def _convolve_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        height, width = channel.shape
        kh, kw = kernel.shape
        fft_shape = (height + kh - 1, width + kw - 1)
        image_fft = np.fft.rfft2(channel, s=fft_shape)
        kernel_fft = np.fft.rfft2(kernel, s=fft_shape)
        convolved = np.fft.irfft2(image_fft * kernel_fft, s=fft_shape)
        top = kh // 2
        left = kw // 2
        return convolved[top : top + height, left : left + width]
