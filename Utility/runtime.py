#!/usr/bin/env python3
# Shared runtime config for pipeline entry points.

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping, TypedDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
PAIRS_ROOT = DATASET_ROOT / "pairs"
RESULTS_ROOT = PROJECT_ROOT / "Results"
ATTACK_GENERATOR_ROOT = PROJECT_ROOT / "Modifiers" / "attack"

THROTTLE_BATCH_SIZE = 32
MODEL_THROTTLE_DELAYS = {
    "InsightFace": 0.02,
    "FaceNet": 0.5,
}
ALLOW_MISSING_PAIRS = True
DEFAULT_TORCH_DEVICE = "auto"
DEFAULT_ONNX_PROVIDER = "auto"
TORCH_DEVICE_ENV = "FYP_TORCH_DEVICE"
ONNX_PROVIDER_ENV = "FYP_ONNX_PROVIDER"
ATTACK_METHODS = (
    "advfacegan",
    "faceshifter",
    "fomm",
    "mipgan",
    "mordiff",
    "reface",
    "simswap",
)
ATTACK_GENERATOR_SCRIPTS = {
    method: ATTACK_GENERATOR_ROOT / method / "generate.py"
    for method in ATTACK_METHODS
}


class ModelSpec(TypedDict):
    name: str
    generate_script: Path


MODELS: list[ModelSpec] = [
    {
        "name": "InsightFace",
        "generate_script": PROJECT_ROOT / "Models" / "InsightFace" / "generate.py",
    },
    {
        "name": "FaceNet",
        "generate_script": PROJECT_ROOT / "Models" / "FaceNet" / "generate.py",
    },
]
VERIFY_SCRIPT = PROJECT_ROOT / "Recognition" / "evaluate.py"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_input_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")


def run_subprocess(cmd: list[str], stage_name: str, extra_env: Mapping[str, str] | None = None) -> None:
    print(f"\n[RUN] {stage_name}")
    print("[CMD]", " ".join(cmd))
    env = None
    if extra_env:
        env = {**os.environ, **extra_env}
    subprocess.run(cmd, check=True, env=env)


def validate_model_registry(models: list[ModelSpec]) -> None:
    if not models:
        raise ValueError("MODELS is empty.")

    names = [model["name"] for model in models]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate model names: {names}")

    for model in models:
        if not model["generate_script"].exists():
            raise FileNotFoundError(f"{model['name']} generate.py not found")
    if not VERIFY_SCRIPT.exists():
        raise FileNotFoundError(f"Shared verify.py not found: {VERIFY_SCRIPT}")


def resolve_attack_generator_script(attack_method: str, requested: Path | None = None) -> Path:
    if requested is not None:
        resolved = requested.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Attack generator script not found: {resolved}")
        return resolved

    if attack_method not in ATTACK_GENERATOR_SCRIPTS:
        raise ValueError(f"Unknown attack method '{attack_method}'.")

    resolved = ATTACK_GENERATOR_SCRIPTS[attack_method]
    if not resolved.exists():
        raise FileNotFoundError(
            f"Built-in generator missing for '{attack_method}': {resolved}"
        )
    return resolved


def runtime_env_overrides(
    *,
    torch_device: str | None = None,
    onnx_provider: str | None = None,
) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if torch_device:
        overrides[TORCH_DEVICE_ENV] = torch_device
    if onnx_provider:
        overrides[ONNX_PROVIDER_ENV] = onnx_provider
    return overrides


def resolve_torch_device(requested: str | None = None) -> str:
    import torch

    choice = (requested or os.environ.get(TORCH_DEVICE_ENV) or DEFAULT_TORCH_DEVICE).strip().lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
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
        return "mps"
    if choice == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported torch device '{choice}'. Use auto, cuda, mps, or cpu.")


def resolve_onnx_providers(requested: str | None = None) -> list[str]:
    import onnxruntime as ort

    choice = (requested or os.environ.get(ONNX_PROVIDER_ENV) or DEFAULT_ONNX_PROVIDER).strip().lower()
    available = set(ort.get_available_providers())

    def require(provider: str) -> list[str]:
        if provider not in available:
            raise RuntimeError(
                f"{provider} requested. Not available. Have: {sorted(available)}"
            )
        fallback = [name for name in ("CPUExecutionProvider",) if name in available and name != provider]
        return [provider, *fallback]

    if choice == "auto":
        for provider in ("CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"):
            if provider in available:
                return [provider, *(["CPUExecutionProvider"] if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available else [])]
        raise RuntimeError("No supported ONNX Runtime provider found.")
    if choice == "coreml":
        return require("CoreMLExecutionProvider")
    if choice == "cuda":
        return require("CUDAExecutionProvider")
    if choice == "cpu":
        return require("CPUExecutionProvider")
    raise ValueError(f"Unsupported ONNX provider '{choice}'. Use auto, coreml, cuda, or cpu.")
