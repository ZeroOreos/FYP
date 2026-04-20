#!/usr/bin/env python3
"""Shared runtime helpers for the training-first pipeline."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
TRAINING_RUNS_ROOT = PROJECT_ROOT / "TrainingRuns"
ATTACK_GENERATOR_ROOT = PROJECT_ROOT / "Modifiers" / "attack"
ATTACK_ASSETS_ROOT = PROJECT_ROOT / "Backends" / "assets" / "attack"
ATTACK_WORKDIR_ROOT = PROJECT_ROOT / "Backends" / "workdirs" / "attack"
RECOGNITION_ASSETS_ROOT = PROJECT_ROOT / "Backends" / "assets" / "recognition"
RECOGNITION_WORKDIR_ROOT = PROJECT_ROOT / "Backends" / "workdirs" / "recognition"

DEFAULT_TORCH_DEVICE = "auto"
TORCH_DEVICE_ENV = "FYP_TORCH_DEVICE"
MPS_HIGH_WATERMARK_ENV = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"


def resolve_torch_device(requested: str | None = None) -> str:
    import torch

    choice = (requested or os.environ.get(TORCH_DEVICE_ENV) or DEFAULT_TORCH_DEVICE).strip().lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            os.environ[MPS_HIGH_WATERMARK_ENV] = "0.0"
            return "mps"
        return "cpu"

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested. Not available in this PyTorch install.")
        return "cuda"
    if choice == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested. Not available on this machine.")
        os.environ[MPS_HIGH_WATERMARK_ENV] = "0.0"
        return "mps"
    if choice == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported torch device '{choice}'. Use auto, cuda, mps, or cpu.")
