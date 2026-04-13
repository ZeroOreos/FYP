#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_ROOT = PROJECT_ROOT / "Backends" / "assets" / "protection"
SOURCE_ROOT = PROJECT_ROOT / "Backends" / "sources" / "protection"
WORKDIR_ROOT = PROJECT_ROOT / "Backends" / "workdirs" / "protection"


@dataclass(frozen=True)
class DownloadSpec:
    url: str
    dest: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, dest: Path) -> None:
    ensure_dir(dest.parent)
    if dest.exists():
        return
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=merged_env)


def ensure_gdown(python_bin: Path) -> None:
    run([str(python_bin), "-m", "pip", "install", "gdown"])


def ensure_huggingface_hub(python_bin: Path) -> None:
    run([str(python_bin), "-m", "pip", "install", "huggingface_hub"])


def gdown_file(python_bin: Path, file_id: str, dest: Path) -> None:
    ensure_dir(dest.parent)
    if dest.exists():
        return
    run([str(python_bin), "-m", "gdown", "--id", file_id, "-O", str(dest)])


def gdown_folder(python_bin: Path, folder_url: str, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    marker = dest_dir / ".gdown_folder_done"
    if marker.exists():
        return
    run([str(python_bin), "-m", "gdown", "--folder", folder_url, "-O", str(dest_dir)])
    marker.write_text("done\n", encoding="utf-8")


def huggingface_snapshot(
    python_bin: Path,
    repo_id: str,
    dest_dir: Path,
    *,
    allow_patterns: list[str] | None = None,
) -> None:
    ensure_dir(dest_dir)
    marker = dest_dir / ".hf_snapshot_done"
    if marker.exists():
        return
    run(
        [
            str(python_bin),
            "-c",
            (
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id={repo_id!r}, local_dir={str(dest_dir)!r}, "
                f"local_dir_use_symlinks=False, allow_patterns={allow_patterns!r})"
            ),
        ],
        env={"HF_HUB_DISABLE_PROGRESS_BARS": "1"},
    )
    marker.write_text("done\n", encoding="utf-8")


def import_falco() -> dict[str, object]:
    repo_dir = SOURCE_ROOT / "FALCO_upstream"
    asset_dir = ASSET_ROOT / "falco"
    python_bin = WORKDIR_ROOT / "falco" / "venv" / "bin" / "python"
    run(
        [str(python_bin), str((repo_dir / "download_models.py").resolve())],
        cwd=asset_dir,
        env={"PYTHONPATH": str(repo_dir)},
    )
    return {
        "status": "imported",
        "dest": str(asset_dir / "models" / "pretrained"),
    }


def import_deepprivacy2() -> dict[str, object]:
    repo_dir = SOURCE_ROOT / "DeepPrivacy2_upstream"
    asset_dir = ASSET_ROOT / "deepprivacy2"
    checkpoint_dir = asset_dir / "torch_home" / "hub" / "checkpoints"
    config_targets = [
        (
            repo_dir / "configs" / "anonymizers" / "face.py",
            asset_dir / "configs" / "anonymizers" / "face.py",
        ),
        (
            repo_dir / "configs" / "fdf" / "stylegan.py",
            asset_dir / "configs" / "fdf" / "stylegan.py",
        ),
    ]
    copied: list[str] = []
    for src, dest in config_targets:
        ensure_dir(dest.parent)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(str(dest))
    downloads = [
        DownloadSpec(
            "https://huggingface.co/spaces/haakohu/deep_privacy2_face/resolve/main/torch_home/hub/checkpoints/WIDERFace_DSFD_RES152.pth",
            checkpoint_dir / "61be4ec7-8c11-4a4a-a9f4-827144e4ab4f0c2764c1-80a0-4083-bbfa-68419f889b80e4692358-979b-458e-97da-c1a1660b3314",
        ),
        DownloadSpec(
            "https://huggingface.co/spaces/haakohu/deep_privacy2_face/resolve/main/torch_home/hub/checkpoints/89660f04-5c11-4dbf-adac-cbe2f11b0aeea25cbf78-7558-475a-b3c7-03f5c10b7934646b0720-ca0a-4d53-aded-daddbfa45c9e",
            checkpoint_dir / "89660f04-5c11-4dbf-adac-cbe2f11b0aeea25cbf78-7558-475a-b3c7-03f5c10b7934646b0720-ca0a-4d53-aded-daddbfa45c9e",
        ),
        DownloadSpec(
            "https://huggingface.co/spaces/haakohu/deep_privacy2_face/resolve/main/torch_home/hub/checkpoints/WIDERFace_DSFD_RES152.pth",
            checkpoint_dir / "WIDERFace_DSFD_RES152.pth",
        ),
    ]
    for spec in downloads:
        download_file(spec.url, spec.dest)
    return {
        "status": "imported",
        "files": copied + [str(spec.dest) for spec in downloads],
        "note": "DeepPrivacy2 face assets imported from the author's Hugging Face Space mirror because the original loke host returned HTTP 502.",
    }


def import_unidefense() -> dict[str, object]:
    asset_dir = ASSET_ROOT / "unidefense" / "ckpt"
    downloads = [
        DownloadSpec(
            "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth",
            asset_dir / "adv-efficientnet-b4-44fb3a87.pth",
        ),
        DownloadSpec(
            "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            asset_dir / "resnet18-5c106cde.pth",
        ),
        DownloadSpec(
            "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth",
            asset_dir / "resnet50_a1_0-14fe96d1.pth",
        ),
    ]
    for spec in downloads:
        download_file(spec.url, spec.dest)
    return {
        "status": "imported",
        "files": [str(spec.dest) for spec in downloads],
    }


def import_m2f2() -> dict[str, object]:
    python_bin = WORKDIR_ROOT / "m2f2_det" / "venv" / "bin" / "python"
    asset_dir = ASSET_ROOT / "m2f2_det"
    ensure_gdown(python_bin)
    ensure_huggingface_hub(python_bin)
    gdown_file(python_bin, "19oEpKB96xJVSrwkLV0ewje-W2dfBAR58", asset_dir / "utils" / "weights" / "vision_tower.pth")
    gdown_file(python_bin, "1X1ZUZkCwqg9mrsqoOS0EoO3v5WABNBAw", asset_dir / "checkpoints" / "stage_1" / "current_model_180.pth")
    clip_dir = asset_dir / "huggingface" / "openai--clip-vit-large-patch14-336"
    huggingface_snapshot(
        python_bin,
        "openai/clip-vit-large-patch14-336",
        clip_dir,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ],
    )
    return {
        "status": "imported",
        "files": [
            str(asset_dir / "utils" / "weights" / "vision_tower.pth"),
            str(asset_dir / "checkpoints" / "stage_1" / "current_model_180.pth"),
            str(clip_dir),
        ],
    }


def import_faceshield() -> dict[str, object]:
    python_bin = WORKDIR_ROOT / "faceshield" / "venv" / "bin" / "python"
    asset_dir = ASSET_ROOT / "faceshield" / "models"
    ensure_gdown(python_bin)
    gdown_folder(python_bin, "https://drive.google.com/drive/folders/1lmKkNUsoebszm3W5xhnw1ybKVwozMYwO?usp=drive_link", asset_dir)
    return {
        "status": "imported",
        "dest": str(asset_dir),
    }


def import_fao() -> dict[str, object]:
    python_bin = WORKDIR_ROOT / "facial_attributes_obfuscation" / "venv" / "bin" / "python"
    asset_dir = ASSET_ROOT / "facial_attributes_obfuscation" / "pretrained"
    ensure_gdown(python_bin)
    gdown_folder(python_bin, "https://drive.google.com/drive/folders/1upsYYgIzyzRuNEGCc7uPQ1AZ6NcHiPbD?usp=sharing", asset_dir)
    gdown_file(python_bin, "1gy9OJlVfBulWkIEnZhGpOLu084RgHw39", asset_dir / "resnet50_scratch_weight.pkl")
    return {
        "status": "partial",
        "files": [
            str(asset_dir / "resnet50_scratch_weight.pkl"),
        ],
        "note": "Google Drive assets imported; Baidu-hosted ArcFace and FaceParser weights still need manual retrieval.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import protection checkpoints into repo-owned asset roots.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["deepprivacy2", "unidefense", "m2f2_det", "faceshield", "facial_attributes_obfuscation", "falco"],
    )
    args = parser.parse_args()

    handlers = {
        "falco": import_falco,
        "deepprivacy2": import_deepprivacy2,
        "unidefense": import_unidefense,
        "m2f2_det": import_m2f2,
        "faceshield": import_faceshield,
        "facial_attributes_obfuscation": import_fao,
    }

    report: dict[str, object] = {}
    for model in args.models:
        try:
            report[model] = handlers[model]()
        except Exception as exc:
            report[model] = {"status": "failed", "error": str(exc)}

    report_path = ASSET_ROOT / "_import_report.json"
    ensure_dir(report_path.parent)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
