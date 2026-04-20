#!/usr/bin/env python3
# Shared helpers for attack generator scripts.

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageFilter


ArrayTransform = Callable[[np.ndarray, np.ndarray, Any], np.ndarray]


@dataclass
class PairRecord:
    index: int
    victim_identity: str
    attacker_identity: str
    victim_image: str
    attacker_image: str
    output_image: str
    status: str
    message: str
    source_similarity: float | None = None
    target_similarity: float | None = None
    perturbation_linf: float | None = None


def bootstrap_project_root(module_file: str | Path, parents: int = 3) -> Path:
    project_root = Path(module_file).resolve().parents[parents]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKENDS_ROOT = PROJECT_ROOT / "Backends"
BACKEND_SOURCES_ROOT = BACKENDS_ROOT / "sources" / "attack"
BACKEND_ASSETS_ROOT = BACKENDS_ROOT / "assets" / "attack"
BACKEND_WORKDIRS_ROOT = BACKENDS_ROOT / "workdirs" / "attack"


@dataclass(frozen=True)
class ExternalBackendDefaults:
    method_slug: str
    upstream_dir_name: str
    default_entry_script: str | None = None
    default_checkpoint: str | None = None
    default_config: str | None = None

    @property
    def repo_dir(self) -> Path:
        return BACKEND_SOURCES_ROOT / self.upstream_dir_name

    @property
    def checkpoint_dir(self) -> Path:
        return BACKEND_ASSETS_ROOT / self.method_slug

    @property
    def work_dir(self) -> Path:
        return BACKEND_WORKDIRS_ROOT / self.method_slug


def external_backend_defaults(
    method_slug: str,
    upstream_dir_name: str,
    *,
    default_entry_script: str | None = None,
    default_checkpoint: str | None = None,
    default_config: str | None = None,
) -> ExternalBackendDefaults:
    return ExternalBackendDefaults(
        method_slug=method_slug,
        upstream_dir_name=upstream_dir_name,
        default_entry_script=default_entry_script,
        default_checkpoint=default_checkpoint,
        default_config=default_config,
    )


def build_base_arg_parser(
    description: str,
    *,
    default_image_size: int = 256,
    default_jpeg_quality: int = 95,
    include_seed: bool = False,
    include_overwrite: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("pair_input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("records_out", type=Path)
    if include_seed:
        parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=default_image_size)
    parser.add_argument("--jpeg-quality", type=int, default=default_jpeg_quality)
    if include_overwrite:
        parser.add_argument("--overwrite", action="store_true")
    return parser


def finalize_generator_args(
    args: argparse.Namespace,
    *,
    optional_path_fields: Sequence[str] = (),
) -> argparse.Namespace:
    path_fields = ("dataset_dir", "pair_input", "output_dir", "records_out", *optional_path_fields)
    for field in path_fields:
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, Path(value).resolve())
    return args


def require_existing_paths(*paths: Path | None) -> None:
    for path in paths:
        if path is not None and not path.exists():
            raise FileNotFoundError(path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def path_exists_or_none(path: Path | None) -> Path | None:
    if path is None or not path.exists():
        return None
    return path


def resolve_existing_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    resolved = value.resolve()
    return resolved if resolved.exists() else None


def resolve_external_backend_args(
    args: argparse.Namespace,
    *,
    defaults: ExternalBackendDefaults,
    repo_env: str,
    entry_env: str | None = None,
    checkpoint_env: str | None = None,
    config_env: str | None = None,
) -> argparse.Namespace:
    args = finalize_generator_args(
        args,
        optional_path_fields=(
            "repo_dir",
            "entry_script",
            "checkpoint",
            "config",
            "work_dir",
        ),
    )

    repo_dir = resolve_path_arg(getattr(args, "repo_dir", None), repo_env)
    if repo_dir is None:
        repo_dir = path_exists_or_none(defaults.repo_dir)
    args.repo_dir = repo_dir

    entry_script = resolve_path_arg(getattr(args, "entry_script", None), entry_env) if entry_env else getattr(args, "entry_script", None)
    if entry_script is None and defaults.default_entry_script and repo_dir is not None:
        entry_script = path_exists_or_none(repo_dir / defaults.default_entry_script)
    args.entry_script = resolve_existing_path(entry_script)

    checkpoint = resolve_path_arg(getattr(args, "checkpoint", None), checkpoint_env) if checkpoint_env else getattr(args, "checkpoint", None)
    if checkpoint is None and defaults.default_checkpoint:
        checkpoint = path_exists_or_none(defaults.checkpoint_dir / defaults.default_checkpoint)
    args.checkpoint = resolve_existing_path(checkpoint)

    config = resolve_path_arg(getattr(args, "config", None), config_env) if config_env else getattr(args, "config", None)
    if config is None and defaults.default_config and repo_dir is not None:
        config = path_exists_or_none(repo_dir / defaults.default_config)
    args.config = resolve_existing_path(config)

    work_dir = getattr(args, "work_dir", None)
    args.work_dir = work_dir if work_dir is not None else defaults.work_dir
    return args


def external_backend_extra_lines(
    *,
    repo_dir: Path | None,
    entry_script: Path | None,
    checkpoint: Path | None = None,
    config: Path | None = None,
    note: str | None = None,
) -> tuple[str, ...]:
    lines = [
        f"repo_dir: {repo_dir}" if repo_dir is not None else "repo_dir: missing",
        f"entry_script: {entry_script}" if entry_script is not None else "entry_script: missing",
    ]
    if checkpoint is not None:
        lines.append(f"checkpoint: {checkpoint}")
    if config is not None:
        lines.append(f"config: {config}")
    if note:
        lines.append(note)
    return tuple(lines)


def set_global_seed(seed: int, *, include_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if include_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_pair_rows(pair_input: Path) -> list[dict[str, str]]:
    data = np.load(pair_input, allow_pickle=True)
    required = ("victim_identity", "attacker_identity", "victim_image", "attacker_image")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"attack pair input missing required keys {missing}: {pair_input}")

    rows: list[dict[str, str]] = []
    for victim_identity, attacker_identity, victim_image, attacker_image in zip(
        data["victim_identity"].tolist(),
        data["attacker_identity"].tolist(),
        data["victim_image"].tolist(),
        data["attacker_image"].tolist(),
    ):
        rows.append({
            "victim_identity": str(victim_identity),
            "attacker_identity": str(attacker_identity),
            "victim_image": str(Path(victim_image).resolve()),
            "attacker_image": str(Path(attacker_image).resolve()),
        })
    return rows


def build_output_path(output_dir: Path, row: dict[str, str]) -> Path:
    return output_dir / row["victim_identity"] / Path(row["victim_image"]).name


def make_record(
    *,
    index: int,
    row: dict[str, str],
    output_path: Path,
    status: str,
    message: str,
    source_path: Path | None = None,
    target_path: Path | None = None,
    source_similarity: float | None = None,
    target_similarity: float | None = None,
    perturbation_linf: float | None = None,
) -> dict[str, Any]:
    source_path = source_path or Path(row["attacker_image"]).resolve()
    target_path = target_path or Path(row["victim_image"]).resolve()
    return asdict(PairRecord(
        index=index,
        victim_identity=row["victim_identity"],
        attacker_identity=row["attacker_identity"],
        victim_image=str(target_path),
        attacker_image=str(source_path),
        output_image=str(output_path),
        status=status,
        message=message,
        source_similarity=source_similarity,
        target_similarity=target_similarity,
        perturbation_linf=perturbation_linf,
    ))


def sanitize_args(args: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def write_summary(
    *,
    method_name: str,
    dataset_dir: Path,
    pair_input: Path,
    output_dir: Path,
    records_out: Path,
    args: Any,
    records: list[dict[str, Any]],
    extra_summary: dict[str, Any] | None = None,
) -> None:
    summary = {
        "generator": method_name,
        "dataset_dir": str(dataset_dir),
        "pair_input": str(pair_input),
        "output_dir": str(output_dir),
        "records_out": str(records_out),
        "config": sanitize_args(args),
        "num_pairs": len(records),
        "num_ok": sum(record["status"] == "ok" for record in records),
        "num_skipped": sum(record["status"] == "skipped" for record in records),
        "num_failed": sum(record["status"] == "failed" for record in records),
        "records": records,
    }
    if extra_summary:
        summary.update(extra_summary)
    records_out.parent.mkdir(parents=True, exist_ok=True)
    with open(records_out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def print_run_header(
    *,
    method_name: str,
    dataset_dir: Path,
    pair_input: Path,
    output_dir: Path,
    num_pairs: int,
    extra_lines: Iterable[str] = (),
) -> None:
    print(f"[INFO] generator: {method_name}")
    print(f"[INFO] dataset_dir: {dataset_dir}")
    print(f"[INFO] pair_input: {pair_input}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] pairs: {num_pairs}")
    for line in extra_lines:
        print(f"[INFO] {line}")


def resolve_path_arg(value: Path | None, env_name: str | None = None) -> Path | None:
    if value is not None:
        return value.resolve()
    if env_name is None:
        return None
    raw = os.environ.get(env_name)
    if not raw:
        return None
    return Path(raw).resolve()


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env_updates: dict[str, str] | None = None,
    stage_name: str | None = None,
) -> None:
    label = stage_name or Path(cmd[0]).name
    print(f"[RUN] {label}")
    print("[CMD]", " ".join(cmd))
    env = None
    if env_updates:
        env = {**os.environ, **env_updates}
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd is not None else None, env=env)


def prepare_pair_workdir(work_root: Path, index: int) -> Path:
    pair_dir = work_root / f"{index:05d}"
    if pair_dir.exists():
        shutil.rmtree(pair_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)
    return pair_dir


def copy_image_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def replace_tokens(value: str, mapping: dict[str, str]) -> str:
    result = value
    for key, replacement in mapping.items():
        result = result.replace(f"{{{key}}}", replacement)
    return result


def build_extra_args(extra_args: Sequence[str], mapping: dict[str, str]) -> list[str]:
    return [replace_tokens(value, mapping) for value in extra_args]


def find_first_existing_file(candidates: Sequence[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_latest_file(root: Path, patterns: Sequence[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(root.glob(pattern)))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def create_still_video_ffmpeg(image_path: Path, video_path: Path, *, seconds: float = 1.0, fps: int = 25) -> Path:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        str(seconds),
        "-r",
        str(fps),
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    run_command(cmd, stage_name="ffmpeg still->video")
    return video_path


def extract_first_frame_ffmpeg(video_path: Path, image_path: Path) -> Path:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(image_path),
    ]
    run_command(cmd, stage_name="ffmpeg video->frame")
    return image_path


def load_image_array(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_image_tensor(path: Path, image_size: int, device: Any) -> Any:
    import torch

    array = load_image_array(path, image_size)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def tensor_to_image_array(tensor: Any) -> np.ndarray:
    array = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(-1.0, 1.0).numpy()
    return ((array + 1.0) * 0.5).astype(np.float32)


def save_image_array(array: np.ndarray, path: Path, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(np.round(clipped * 255.0).astype(np.uint8), mode="RGB")
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, quality=max(1, min(100, int(jpeg_quality))))
    else:
        image.save(path)


def save_tensor_image(tensor: Any, path: Path, jpeg_quality: int) -> None:
    save_image_array(tensor_to_image_array(tensor), path, jpeg_quality)


def blur_array(array: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    image = image.filter(ImageFilter.GaussianBlur(radius=max(0.0, float(radius))))
    return np.asarray(image, dtype=np.float32) / 255.0


def shift_array(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    result = np.roll(array, shift=dy, axis=0)
    result = np.roll(result, shift=dx, axis=1)
    if dy > 0:
        result[:dy, :, :] = result[dy:dy + 1, :, :]
    elif dy < 0:
        result[dy:, :, :] = result[dy - 1:dy, :, :]
    if dx > 0:
        result[:, :dx, :] = result[:, dx:dx + 1, :]
    elif dx < 0:
        result[:, dx:, :] = result[:, dx - 1:dx, :]
    return result


def image_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_vec = a.astype(np.float64).reshape(-1)
    b_vec = b.astype(np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec)), 1e-12)
    return float(np.dot(a_vec, b_vec) / denom)


def image_linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def ellipse_mask(
    height: int,
    width: int,
    *,
    center_x: float = 0.5,
    center_y: float = 0.5,
    radius_x: float = 0.32,
    radius_y: float = 0.42,
    feather: float = 0.12,
) -> np.ndarray:
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    norm = ((xx - center_x) / max(radius_x, 1e-6)) ** 2 + ((yy - center_y) / max(radius_y, 1e-6)) ** 2
    edge_start = max(1.0 - feather, 1e-6)
    mask = np.clip((1.0 - norm) / edge_start, 0.0, 1.0)
    return mask[..., None]


def band_mask(
    height: int,
    width: int,
    *,
    y_center: float,
    band_height: float,
    feather: float = 0.08,
) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    distance = np.abs(y - y_center)
    band = np.clip((band_height * 0.5 - distance) / max(feather, 1e-6), 0.0, 1.0)
    return np.repeat(band[:, None, None], width, axis=1)


def match_color_statistics(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    mix: float = 0.8,
) -> np.ndarray:
    weights = np.clip(mask, 0.0, 1.0)
    denom = np.clip(np.sum(weights, axis=(0, 1), keepdims=True), 1e-6, None)
    source_mean = np.sum(source * weights, axis=(0, 1), keepdims=True) / denom
    target_mean = np.sum(target * weights, axis=(0, 1), keepdims=True) / denom
    source_var = np.sum(((source - source_mean) ** 2) * weights, axis=(0, 1), keepdims=True) / denom
    target_var = np.sum(((target - target_mean) ** 2) * weights, axis=(0, 1), keepdims=True) / denom
    aligned = (target - target_mean) * np.sqrt((source_var + 1e-6) / (target_var + 1e-6)) + source_mean
    return np.clip((1.0 - mix) * target + mix * aligned, 0.0, 1.0)


def run_generator(
    *,
    method_name: str,
    dataset_dir: Path,
    pair_input: Path,
    output_dir: Path,
    records_out: Path,
    args: Any,
    image_size: int,
    jpeg_quality: int,
    overwrite: bool,
    transform: ArrayTransform,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    require_existing_paths(dataset_dir, pair_input)
    rows = load_pair_rows(pair_input)
    output_dir.mkdir(parents=True, exist_ok=True)
    print_run_header(
        method_name=method_name,
        dataset_dir=dataset_dir,
        pair_input=pair_input,
        output_dir=output_dir,
        num_pairs=len(rows),
    )

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_path = Path(row["attacker_image"]).resolve()
        target_path = Path(row["victim_image"]).resolve()
        output_path = build_output_path(output_dir, row)
        print(f"[PAIR] {index + 1}/{len(rows)} -> {output_path}")

        if output_path.exists() and not overwrite:
            records.append(make_record(
                index=index,
                row=row,
                output_path=output_path,
                status="skipped",
                message="output already exists",
                source_path=source_path,
                target_path=target_path,
            ))
            continue

        try:
            source = load_image_array(source_path, image_size)
            target = load_image_array(target_path, image_size)
            output = np.clip(transform(source, target, args), 0.0, 1.0)
            save_image_array(output, output_path, jpeg_quality)
            records.append(make_record(
                index=index,
                row=row,
                output_path=output_path,
                status="ok",
                message="generated",
                source_path=source_path,
                target_path=target_path,
                source_similarity=image_cosine(output, source),
                target_similarity=image_cosine(output, target),
                perturbation_linf=image_linf(output, source),
            ))
        except Exception as exc:  # pragma: no cover - per-sample failures are part of runtime behavior
            records.append(make_record(
                index=index,
                row=row,
                output_path=output_path,
                status="failed",
                message=str(exc),
                source_path=source_path,
                target_path=target_path,
            ))
            print(f"[WARN] Pair failed {index}: {exc}")

    write_summary(
        method_name=method_name,
        dataset_dir=dataset_dir,
        pair_input=pair_input,
        output_dir=output_dir,
        records_out=records_out,
        args=args,
        records=records,
        extra_summary=extra_summary,
    )
    print(f"[INFO] records written: {records_out}")
