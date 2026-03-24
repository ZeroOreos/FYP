#!/usr/bin/env python3
# python3 Utility/modify.py <dataset_dir> --step <name[:k=v,...]> [--step ...] [--pair-input <atkpairs>] -> Dataset/<dataset>/<newest__...__base_root>; preserves structure


from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

from Modifiers.preprocessing import (
    BrightnessShift,
    ContrastShift,
    EyeBandOcclusion,
    FaceMaskOcclusion,
    GammaShift,
    GaussianBlur,
    JPEGCompression,
    MotionBlur,
    RandomBlockOcclusion,
    ResolutionResampling,
    RotationMisalignment,
)
from Utility.pathfinder import resolve_dataset_context


PROJECT_ROOT = Path(__file__).resolve().parent
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_INFLIGHT_MULTIPLIER = 4


def _fmt_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _fmt_ratio_pct(value: float) -> str:
    return f"{int(round(value * 100))}pct"


def _fmt_scale_token(scale: float) -> str:
    if scale <= 0:
        raise ValueError("scale must be > 0")
    factor = 1.0 / scale
    if abs(factor - round(factor)) < 1e-6:
        return f"x{int(round(factor))}"
    return f"scale{_fmt_number(scale)}"


@dataclass(frozen=True)
class ModifierSpec:
    cli_name: str
    factory: Callable[..., Any]
    token_builder: Callable[[dict[str, Any]], str]
    allowed_params: frozenset[str]


def _token_downsample(params: dict[str, Any]) -> str:
    scale = params.get("scale")
    if scale is None:
        severity = int(params["severity"])
        scale = ResolutionResampling.parameter_table[severity]
    return f"downsample_{_fmt_scale_token(float(scale))}"


def _token_gblur(params: dict[str, Any]) -> str:
    sigma = params.get("sigma")
    if sigma is None:
        sigma = GaussianBlur.parameter_table[int(params["severity"])]
    return f"blur_sigma{_fmt_number(float(sigma))}"


def _token_mblur(params: dict[str, Any]) -> str:
    length = params.get("length")
    if length is None:
        length = MotionBlur.parameter_table[int(params["severity"])]
    return f"motion_len{_fmt_number(float(length))}"


def _token_jpeg(params: dict[str, Any]) -> str:
    quality = params.get("quality")
    if quality is None:
        quality = JPEGCompression.parameter_table[int(params["severity"])]
    return f"jpeg_q{int(quality)}"


def _token_brightness(params: dict[str, Any]) -> str:
    factor = params.get("factor")
    if factor is None:
        factor = BrightnessShift.parameter_table[int(params["severity"])]
    return f"illum_brightness{_fmt_number(float(factor))}"


def _token_contrast(params: dict[str, Any]) -> str:
    factor = params.get("factor")
    if factor is None:
        factor = ContrastShift.parameter_table[int(params["severity"])]
    return f"illum_contrast{_fmt_number(float(factor))}"


def _token_gamma(params: dict[str, Any]) -> str:
    gamma = params.get("gamma")
    if gamma is None:
        gamma = GammaShift.parameter_table[int(params["severity"])]
    return f"illum_gamma{_fmt_number(float(gamma))}"


def _token_rotation(params: dict[str, Any]) -> str:
    degrees = params.get("degrees")
    if degrees is None:
        degrees = RotationMisalignment.parameter_table[int(params["severity"])]
    return f"rotation_deg{_fmt_number(float(degrees))}"


def _token_mask(params: dict[str, Any]) -> str:
    coverage = params.get("coverage")
    if coverage is None:
        coverage = FaceMaskOcclusion.parameter_table[int(params["severity"])]
    return f"occlusion_mask{_fmt_ratio_pct(float(coverage))}"


def _token_eye_band(params: dict[str, Any]) -> str:
    band_height = params.get("band_height")
    if band_height is None:
        band_height = EyeBandOcclusion.parameter_table[int(params["severity"])]
    return f"occlusion_eyeband{_fmt_ratio_pct(float(band_height))}"


def _token_block(params: dict[str, Any]) -> str:
    area_ratio = params.get("area_ratio")
    if area_ratio is None:
        area_ratio = RandomBlockOcclusion.parameter_table[int(params["severity"])]
    return f"occlusion_block{_fmt_ratio_pct(float(area_ratio))}"


REGISTRY: dict[str, ModifierSpec] = {
    "downsample": ModifierSpec("downsample", ResolutionResampling, _token_downsample, frozenset({"severity", "scale"})),
    "gaussian_blur": ModifierSpec("gaussian_blur", GaussianBlur, _token_gblur, frozenset({"severity", "sigma"})),
    "blur": ModifierSpec("blur", GaussianBlur, _token_gblur, frozenset({"severity", "sigma"})),
    "motion_blur": ModifierSpec("motion_blur", MotionBlur, _token_mblur, frozenset({"severity", "length", "angle"})),
    "jpeg": ModifierSpec("jpeg", JPEGCompression, _token_jpeg, frozenset({"severity", "quality"})),
    "brightness": ModifierSpec("brightness", BrightnessShift, _token_brightness, frozenset({"severity", "factor"})),
    "contrast": ModifierSpec("contrast", ContrastShift, _token_contrast, frozenset({"severity", "factor"})),
    "gamma": ModifierSpec("gamma", GammaShift, _token_gamma, frozenset({"severity", "gamma"})),
    "rotation": ModifierSpec("rotation", RotationMisalignment, _token_rotation, frozenset({"severity", "degrees"})),
    "mask": ModifierSpec("mask", FaceMaskOcclusion, _token_mask, frozenset({"severity", "coverage"})),
    "eye_band": ModifierSpec("eye_band", EyeBandOcclusion, _token_eye_band, frozenset({"severity", "band_height"})),
    "block": ModifierSpec("block", RandomBlockOcclusion, _token_block, frozenset({"severity", "area_ratio", "aspect_ratio"})),
}


@dataclass(frozen=True)
class StepConfig:
    spec_name: str
    params: dict[str, Any]
    token: str


@dataclass(frozen=True)
class Job:
    src_path: Path
    dst_path: Path
    relative_path: str


def parse_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in lowered for char in (".", "e")):
            return float(lowered)
        return int(lowered)
    except ValueError:
        return value


def parse_step_spec(raw: str) -> StepConfig:
    if ":" in raw:
        name, raw_params = raw.split(":", 1)
    else:
        name, raw_params = raw, ""

    name = name.strip()
    if name not in REGISTRY:
        raise ValueError(f"unknown modifier '{name}'")

    spec = REGISTRY[name]
    params: dict[str, Any] = {}
    if raw_params.strip():
        for entry in raw_params.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(f"invalid step parameter '{entry}' in '{raw}'")
            key, value = entry.split("=", 1)
            key = key.strip()
            if key not in spec.allowed_params:
                raise ValueError(f"modifier '{name}' does not accept '{key}'")
            params[key] = parse_scalar(value)

    if "severity" not in params and len(params) == 0:
        params["severity"] = 3

    token = spec.token_builder(params)
    return StepConfig(spec_name=name, params=params, token=token)


def build_modifier(step: StepConfig, base_seed: int) -> Any:
    spec = REGISTRY[step.spec_name]
    kwargs = dict(step.params)
    kwargs.setdefault("seed", base_seed)
    return spec.factory(**kwargs)


def derive_output_variant_name(dataset_dir: Path, steps: list[StepConfig]) -> str:
    context = resolve_dataset_context(dataset_dir)
    existing_tokens: list[str] = []
    if context.variant_name != context.base_root_name:
        if context.transform_chain != "clean":
            existing_tokens.extend(token for token in context.transform_chain.split("__") if token)
    new_tokens = [step.token for step in steps]
    tokens = list(reversed(new_tokens)) + existing_tokens
    if not tokens:
        raise ValueError("at least one modifier step is required")
    return f"{'__'.join(tokens)}_{context.base_root_name}"


def existing_transform_tokens(dataset_dir: Path) -> list[str]:
    context = resolve_dataset_context(dataset_dir)
    if context.variant_name == context.base_root_name or context.transform_chain == "clean":
        return []
    return [token for token in context.transform_chain.split("__") if token]


def validate_step_uniqueness(dataset_dir: Path, steps: list[StepConfig]) -> None:
    existing = set(existing_transform_tokens(dataset_dir))
    requested = [step.token for step in steps]
    duplicate_existing = [token for token in requested if token in existing]
    if duplicate_existing:
        raise ValueError(
            "requested transform already exists in dataset chain: "
            + ", ".join(duplicate_existing)
        )

    seen: set[str] = set()
    duplicate_requested: list[str] = []
    for token in requested:
        if token in seen and token not in duplicate_requested:
            duplicate_requested.append(token)
        seen.add(token)
    if duplicate_requested:
        raise ValueError(
            "requested transform is duplicated in this run: "
            + ", ".join(duplicate_requested)
        )


def derive_output_dataset_dir(dataset_dir: Path, steps: list[StepConfig]) -> Path:
    context = resolve_dataset_context(dataset_dir)
    output_variant_name = derive_output_variant_name(dataset_dir, steps)
    return context.dataset_root.joinpath(*context.dataset_relative_parts[:-1], output_variant_name)


def collect_jobs(dataset_dir: Path, output_dir: Path) -> list[Job]:
    jobs: list[Job] = []
    for src_path in sorted(dataset_dir.rglob("*")):
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in VALID_EXTS:
            continue
        relative = src_path.relative_to(dataset_dir)
        jobs.append(
            Job(
                src_path=src_path,
                dst_path=output_dir / relative,
                relative_path=relative.as_posix(),
            )
        )
    if not jobs:
        raise RuntimeError(f"no valid images found under {dataset_dir}")
    return jobs


def collect_jobs_from_pair_input(dataset_dir: Path, output_dir: Path, pair_input: Path) -> list[Job]:
    data = np.load(pair_input, allow_pickle=True)
    candidate_keys = ["victim_image", "target_image", "image_paths"]
    image_list = None
    for key in candidate_keys:
        if key in data.files:
            image_list = [str(value) for value in data[key].tolist()]
            break
    if image_list is None:
        raise ValueError(f"pair input does not contain a supported image path key: {pair_input}")

    jobs: list[Job] = []
    seen: set[str] = set()
    for image_str in image_list:
        src_path = Path(image_str)
        if not src_path.is_absolute():
            src_path = (dataset_dir / image_str).resolve()
        if not src_path.exists():
            continue
        relative = src_path.relative_to(dataset_dir.resolve()).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        jobs.append(
            Job(
                src_path=src_path,
                dst_path=output_dir / relative,
                relative_path=relative,
            )
        )
    if not jobs:
        raise RuntimeError(f"no valid images resolved from pair input: {pair_input}")
    return jobs


def stable_image_seed(global_seed: int, relative_path: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{relative_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, quality=95)
    else:
        image.save(path)


def process_job(job: Job, steps: list[StepConfig], seed: int, overwrite: bool) -> tuple[str, Optional[str]]:
    if job.dst_path.exists() and not overwrite:
        return "skipped", None

    rng_seed = stable_image_seed(seed, job.relative_path)
    rng = np.random.default_rng(rng_seed)
    modifiers = [build_modifier(step, base_seed=rng_seed) for step in steps]

    try:
        with Image.open(job.src_path) as image:
            image.load()
            out = image.copy()
    except (UnidentifiedImageError, OSError) as exc:
        return "failed", f"{job.relative_path}: {exc}"

    for modifier in modifiers:
        out = modifier.apply(out, rng=rng)

    try:
        save_image(out, job.dst_path)
    except OSError as exc:
        return "failed", f"{job.relative_path}: {exc}"

    return "written", None


def write_transform_metadata(dataset_dir: Path, output_dir: Path, steps: list[StepConfig], seed: int, num_workers: int) -> None:
    metadata = build_transform_metadata(dataset_dir, output_dir, steps, seed, num_workers)
    with open(output_dir / "transform.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def build_transform_metadata(dataset_dir: Path, output_dir: Path, steps: list[StepConfig], seed: int, num_workers: int) -> dict[str, Any]:
    context = resolve_dataset_context(dataset_dir)
    return {
        "input_dataset_dir": str(dataset_dir.resolve()),
        "output_dataset_dir": str(output_dir.resolve()),
        "base_dataset": context.base_dataset_name,
        "referenced_base_root": context.base_root_name,
        "variant_name": output_dir.name,
        "variant_type": "single" if len(steps) == 1 and context.variant_type == "clean" else "hybrid",
        "pipeline": [
            {
                "modifier": step.spec_name,
                "token": step.token,
                "params": step.params,
            }
            for step in steps
        ],
        "seed": seed,
        "num_workers": num_workers,
    }


def metadata_matches_request(metadata_path: Path, expected: dict[str, Any]) -> bool:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        existing = json.load(handle)
    return existing == expected


def materialize_dataset(
    dataset_dir: Path,
    output_dir: Path,
    steps: list[StepConfig],
    seed: int,
    num_workers: int,
    overwrite: bool,
    pair_input: Optional[Path],
) -> None:
    jobs = collect_jobs_from_pair_input(dataset_dir, output_dir, pair_input) if pair_input else collect_jobs(dataset_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    failures: list[str] = []

    max_inflight = max(num_workers, num_workers * MAX_INFLIGHT_MULTIPLIER)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        pending: dict[concurrent.futures.Future[tuple[str, Optional[str]]], Job] = {}
        job_iter = iter(jobs)
        processed = 0

        def submit_until_full() -> None:
            while len(pending) < max_inflight:
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                future = executor.submit(process_job, job, steps, seed, overwrite)
                pending[future] = job

        submit_until_full()

        while pending:
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                job = pending.pop(future)
                try:
                    status, error = future.result()
                except Exception as exc:
                    status, error = "failed", f"{job.relative_path}: {exc}"

                processed += 1
                if status == "written":
                    written += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failures.append(error or "unknown failure")

                if processed % 250 == 0 or processed == len(jobs):
                    print(f"[INFO] processed {processed}/{len(jobs)} images")

            submit_until_full()

    if failures:
        failures_path = output_dir / "failures.json"
        with open(failures_path, "w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=2)
        print(f"[WARN] {len(failures)} files failed; details in {failures_path}")

    write_transform_metadata(dataset_dir, output_dir, steps, seed, num_workers)
    print(f"[INFO] written: {written}")
    print(f"[INFO] skipped: {skipped}")
    print(f"[INFO] output: {output_dir}")


def validate_input_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not dataset_dir.is_dir():
        raise NotADirectoryError(dataset_dir)
    resolve_dataset_context(dataset_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a named modifier dataset variant.")
    parser.add_argument("dataset_dir", type=str, help="Path to Dataset/<dataset>/<variant_root>")
    parser.add_argument(
        "--step",
        action="append",
        required=True,
        help="Modifier step like blur:severity=3 or jpeg:quality=30. Repeat to build hybrids.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--pair-input", type=str, default=None, help="Optional atkpairs npz to restrict materialization to referenced images.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing output variant root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    validate_input_dataset(dataset_dir)

    steps = [parse_step_spec(raw_step) for raw_step in args.step]
    validate_step_uniqueness(dataset_dir, steps)
    output_dir = derive_output_dataset_dir(dataset_dir, steps)
    pair_input = Path(args.pair_input).resolve() if args.pair_input else None

    print(f"[INFO] input: {dataset_dir}")
    print(f"[INFO] steps: {[step.token for step in steps]}")
    print(f"[INFO] output: {output_dir}")

    if output_dir.exists() and not args.force:
        metadata_path = output_dir / "transform.json"
        if not metadata_path.exists():
            raise RuntimeError(f"existing output is missing transform metadata: {metadata_path}")
        expected_metadata = build_transform_metadata(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            steps=steps,
            seed=args.seed,
            num_workers=max(1, args.num_workers),
        )
        if metadata_matches_request(metadata_path, expected_metadata):
            print(f"[SKIP] output variant already matches request: {output_dir}")
            return
        raise RuntimeError(
            "existing output variant metadata does not match the current request; "
            "refuse to reuse or overwrite without --force"
        )

    if output_dir.exists() and args.force and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    materialize_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        steps=steps,
        seed=args.seed,
        num_workers=max(1, args.num_workers),
        overwrite=args.overwrite,
        pair_input=pair_input,
    )


if __name__ == "__main__":
    main()
