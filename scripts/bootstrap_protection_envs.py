#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKDIR_ROOT = PROJECT_ROOT / "Backends" / "workdirs" / "protection"


@dataclass(frozen=True)
class EnvSpec:
    slug: str
    packages: tuple[str, ...]
    base_python_bin: str | None = None

    @property
    def workdir(self) -> Path:
        return WORKDIR_ROOT / self.slug

    @property
    def venv(self) -> Path:
        return self.workdir / "venv"

    @property
    def python_bin(self) -> Path:
        return self.venv / "bin" / "python"

    @property
    def pip_bin(self) -> Path:
        return self.venv / "bin" / "pip"

    @property
    def base_python(self) -> str:
        if self.base_python_bin:
            return self.base_python_bin
        return sys.executable


SPECS = {
    "fawkes": EnvSpec(
        "fawkes",
        (
            "numpy",
            "pillow",
            "bleach",
            "mtcnn",
            "tensorflow",
            "keras",
        ),
        base_python_bin=shutil.which("python3.11"),
    ),
    "faceshield": EnvSpec(
        "faceshield",
        (
            "torch",
            "torchvision",
            "torchaudio",
            "accelerate",
            "diffusers",
            "transformers",
            "insightface",
            "opencv-python",
            "opencv-python-headless",
            "pillow",
            "numpy",
            "scipy",
            "scikit-image",
            "scikit-learn",
            "omegaconf",
            "albumentations",
            "matplotlib",
            "prettytable",
            "pytorch-lightning",
            "torch-mtcnn",
            "onnx",
            "onnxruntime",
            "tqdm",
        ),
    ),
    "falco": EnvSpec(
        "falco",
        (
            "torch",
            "torchvision",
            "numpy",
            "scikit-learn",
            "scikit-image",
            "Pillow",
            "matplotlib",
            "tqdm",
            "opencv-python-headless",
            "urllib3",
            "requests",
            "scipy",
            "ninja",
            "ftfy",
            "regex",
            "git+https://github.com/openai/CLIP.git",
        ),
        base_python_bin=shutil.which("python3.11"),
    ),
    "deepprivacy2": EnvSpec(
        "deepprivacy2",
        (
            "torch",
            "torchvision",
            "numpy",
            "cython",
            "matplotlib",
            "tqdm",
            "moviepy",
            "tensorboard",
            "opencv-python",
            "click",
            "black",
            "torch_fidelity==0.3.0",
            "ninja",
            "pyspng",
            "wandb",
            "termcolor",
            "fast_pytorch_kmeans",
            "einops",
            "einops-exts",
            "regex",
            "resize_right==0.0.2",
            "setuptools<82",
            "pillow",
            "scipy",
            "scikit-image",
            "imageio",
            "timm",
            "face_detection@git+https://github.com/hukkelas/DSFD-Pytorch-Inference",
            "tops@git+https://github.com/hukkelas/torch_ops.git",
            "motpy@git+https://github.com/wmuron/motpy@c77f85d27e371c0a298e9a88ca99292d9b9cbe6b",
            "clip@git+https://github.com/openai/CLIP.git@b46f5ac7587d2e1862f8b7b1573179d80dcdd620",
        ),
        base_python_bin=shutil.which("python3.11"),
    ),
    "facial_attributes_obfuscation": EnvSpec(
        "facial_attributes_obfuscation",
        (
            "torch",
            "torchvision",
            "numpy",
            "pillow",
            "opencv-python",
            "scipy",
            "scikit-image",
            "tqdm",
            "matplotlib",
            "pandas",
            "scikit-learn",
            "pyyaml",
            "easydict",
            "munch",
            "pytorch-lightning",
        ),
        base_python_bin=shutil.which("python3.11"),
    ),
    "unidefense": EnvSpec(
        "unidefense",
        (
            "torch",
            "torchvision",
            "albumentations",
            "matplotlib",
            "timm",
            "wandb",
            "scipy",
            "scikit-learn",
            "pyyaml",
            "opencv-python",
            "lmdb",
            "tqdm",
        ),
        base_python_bin=shutil.which("python3.11"),
    ),
    "m2f2_det": EnvSpec(
        "m2f2_det",
        (
            "torch",
            "torchvision",
            "numpy",
            "scikit-learn",
            "scikit-image",
            "pandas",
            "tqdm",
            "efficientnet-pytorch",
            "h5py",
            "tokenizers",
            "sentencepiece",
            "accelerate",
            "transformers",
            "huggingface_hub",
            "einops",
            "einops-exts",
            "timm",
            "jsonlines",
            "ftfy",
            "regex",
            "munch",
            "opencv-python",
        ),
        base_python_bin=shutil.which("python3.11"),
    ),
}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def install_packages(pip_bin: Path, packages: tuple[str, ...]) -> None:
    for package in packages:
        cmd = [str(pip_bin), "install"]
        if "git+" in package:
            cmd.append("--no-build-isolation")
        cmd.append(package)
        run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-backend protection venvs with best-effort installs.")
    parser.add_argument("--models", nargs="*", default=sorted(SPECS))
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    for slug in args.models:
        spec = SPECS[slug]
        spec.workdir.mkdir(parents=True, exist_ok=True)
        if not spec.python_bin.exists():
            run([spec.base_python, "-m", "venv", str(spec.venv)])
        report = {
            "python": str(spec.python_bin),
            "base_python": spec.base_python,
            "packages": list(spec.packages),
            "status": "created",
        }
        if not args.skip_install:
            try:
                run([str(spec.pip_bin), "install", "--upgrade", "pip", "setuptools<82", "wheel"])
                install_packages(spec.pip_bin, spec.packages)
                report["status"] = "installed"
            except subprocess.CalledProcessError as exc:
                report["status"] = "install_failed"
                report["returncode"] = exc.returncode
        (spec.workdir / "env_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
