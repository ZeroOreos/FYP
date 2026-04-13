#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROTECTION_SOURCE_ROOT = PROJECT_ROOT / "Backends" / "sources" / "protection"
PROTECTION_ASSET_ROOT = PROJECT_ROOT / "Backends" / "assets" / "protection"
PROTECTION_WORKDIR_ROOT = PROJECT_ROOT / "Backends" / "workdirs" / "protection"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_images(dataset_dir: Path) -> list[tuple[Path, Path]]:
    dataset_dir = dataset_dir.resolve()
    items: list[tuple[Path, Path]] = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            items.append((path.resolve(), path.relative_to(dataset_dir)))
    if not items:
        raise RuntimeError(f"No valid images found in dataset: {dataset_dir}")
    return items


def group_images_by_parent(items: Iterable[tuple[Path, Path]]) -> dict[Path, list[tuple[Path, Path]]]:
    groups: dict[Path, list[tuple[Path, Path]]] = {}
    for src_path, rel_path in items:
        groups.setdefault(rel_path.parent, []).append((src_path, rel_path))
    return groups


def stage_group_copy(items: Iterable[tuple[Path, Path]], stage_dir: Path) -> dict[str, Path]:
    ensure_dir(stage_dir)
    mapping: dict[str, Path] = {}
    for src_path, rel_path in items:
        staged_name = rel_path.name
        staged_path = stage_dir / staged_name
        shutil.copy2(src_path, staged_path)
        mapping[staged_name] = rel_path
    return mapping


def copy_stage_outputs(stage_dir: Path, output_dir: Path, mapping: dict[str, Path], *, suffix: str = "") -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for staged_name, rel_path in mapping.items():
        base = Path(staged_name).stem
        candidates = sorted(stage_dir.glob(f"{base}{suffix}.*"))
        if not candidates:
            candidates = sorted(stage_dir.glob(f"{base}*"))
        if not candidates:
            records.append({"input": str(rel_path), "status": "missing"})
            continue
        source = candidates[0]
        target = ensure_dir(output_dir / rel_path.parent) / rel_path.name
        shutil.copy2(source, target)
        records.append({"input": str(rel_path), "status": "ok", "output": str(target)})
    return records


def resolve_device(requested: str | None = None) -> str:
    choice = (requested or "auto").strip().lower()
    if choice in {"cuda", "mps", "cpu"}:
        return choice
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env_updates: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        check=True,
        capture_output=capture_output,
    )


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

