#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

import numpy as np

from Modifiers.attack.shared import band_mask, blur_array, build_base_arg_parser, ellipse_mask, match_color_statistics
from Modifiers.attack.shared import external_backend_defaults, finalize_generator_args, run_generator


DEFAULTS = external_backend_defaults(
    "adv_makeup",
    "surrogate/Adv-Makeup_upstream",
)

DEFAULT_IMAGE_SIZE = 256
DEFAULT_COLOR_MATCH = 0.92
DEFAULT_SHADOW_BLEND = 0.72
DEFAULT_FEATHER = 0.14
DEFAULT_SHIMMER = 0.08


def parse_args() -> Any:
    parser = build_base_arg_parser(
        "Generate Adv-Makeup-style probes with a local orbital-makeup approximation.",
        default_image_size=DEFAULT_IMAGE_SIZE,
    )
    parser.add_argument("--color-match", type=float, default=DEFAULT_COLOR_MATCH)
    parser.add_argument("--shadow-blend", type=float, default=DEFAULT_SHADOW_BLEND)
    parser.add_argument("--feather", type=float, default=DEFAULT_FEATHER)
    parser.add_argument("--shimmer", type=float, default=DEFAULT_SHIMMER)
    parser.add_argument("--repo-dir", default=DEFAULTS.repo_dir)
    return finalize_generator_args(parser.parse_args(), optional_path_fields=("repo_dir",))


def _eye_shadow_mask(height: int, width: int, feather: float) -> np.ndarray:
    left_eye = ellipse_mask(
        height,
        width,
        center_x=0.34,
        center_y=0.40,
        radius_x=0.14,
        radius_y=0.08,
        feather=feather,
    )
    right_eye = ellipse_mask(
        height,
        width,
        center_x=0.66,
        center_y=0.40,
        radius_x=0.14,
        radius_y=0.08,
        feather=feather,
    )
    brow_band = band_mask(height, width, y_center=0.34, band_height=0.12, feather=max(feather * 0.6, 1e-3))
    return np.clip(np.maximum(np.maximum(left_eye, right_eye), 0.55 * brow_band), 0.0, 1.0)


def adv_makeup_transform(source: np.ndarray, target: np.ndarray, args: Any) -> np.ndarray:
    height, width = source.shape[:2]
    mask = _eye_shadow_mask(height, width, float(args.feather))

    target_colorized = match_color_statistics(source, target, mask, mix=float(args.color_match))
    source_colorized = match_color_statistics(target, source, mask, mix=min(float(args.color_match) * 0.7, 1.0))

    makeup_base = (
        float(args.shadow_blend) * target_colorized
        + (1.0 - float(args.shadow_blend)) * source_colorized
    )
    makeup_base = blur_array(makeup_base, radius=1.1)

    shimmer = np.clip(target.mean(axis=2, keepdims=True) * float(args.shimmer), 0.0, 0.2)
    result = source * (1.0 - mask) + np.clip(makeup_base + shimmer, 0.0, 1.0) * mask
    return np.clip(result, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    run_generator(
        method_name="adv_makeup_wrapper",
        dataset_dir=args.dataset_dir,
        pair_input=args.pair_input,
        output_dir=args.output_dir,
        records_out=args.records_out,
        args=args,
        image_size=int(args.image_size),
        jpeg_quality=int(args.jpeg_quality),
        overwrite=bool(args.overwrite),
        transform=adv_makeup_transform,
        extra_summary={
            "backend_type": "local_approximation",
            "fidelity_note": "Orbital-region makeup transfer approximation aligned to the Adv-Makeup slot.",
            "native_repo": str(DEFAULTS.repo_dir),
            "native_release_gate": "Full upstream reproduction still needs the original Adv-Makeup checkpoints and dataset-specific test pipeline.",
        },
    )


if __name__ == "__main__":
    main()
