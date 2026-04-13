#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

import numpy as np

from Modifiers.attack.shared import band_mask, build_base_arg_parser, ellipse_mask, match_color_statistics
from Modifiers.attack.shared import finalize_generator_args, run_generator


DEFAULT_IMAGE_SIZE = 256
DEFAULT_FACE_BLEND = 0.5
DEFAULT_COLOR_MATCH = 0.8
DEFAULT_FACE_FEATHER = 0.16
DEFAULT_JAW_SOFTEN = 0.18


def parse_args() -> Any:
    parser = build_base_arg_parser(
        "Generate DiM-style morph probes with a local approximation path.",
        default_image_size=DEFAULT_IMAGE_SIZE,
    )
    parser.add_argument("--face-blend", type=float, default=DEFAULT_FACE_BLEND)
    parser.add_argument("--color-match", type=float, default=DEFAULT_COLOR_MATCH)
    parser.add_argument("--face-feather", type=float, default=DEFAULT_FACE_FEATHER)
    parser.add_argument("--jaw-soften", type=float, default=DEFAULT_JAW_SOFTEN)
    return finalize_generator_args(parser.parse_args())


def dim_transform(source: np.ndarray, target: np.ndarray, args: Any) -> np.ndarray:
    height, width = source.shape[:2]
    face_mask = ellipse_mask(height, width, feather=float(args.face_feather))
    jaw_mask = band_mask(height, width, y_center=0.68, band_height=0.34, feather=max(float(args.jaw_soften), 1e-3))
    blend_mask = np.clip(np.maximum(face_mask, 0.65 * jaw_mask), 0.0, 1.0)

    source_to_target = match_color_statistics(target, source, blend_mask, mix=float(args.color_match))
    target_to_source = match_color_statistics(source, target, blend_mask, mix=float(args.color_match))
    morphed_face = (
        (1.0 - float(args.face_blend)) * source_to_target
        + float(args.face_blend) * target_to_source
    )
    base_canvas = 0.5 * source + 0.5 * target
    return np.clip(base_canvas * (1.0 - blend_mask) + morphed_face * blend_mask, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    run_generator(
        method_name="dim_wrapper",
        dataset_dir=args.dataset_dir,
        pair_input=args.pair_input,
        output_dir=args.output_dir,
        records_out=args.records_out,
        args=args,
        image_size=int(args.image_size),
        jpeg_quality=int(args.jpeg_quality),
        overwrite=bool(args.overwrite),
        transform=dim_transform,
        extra_summary={
            "backend_type": "local_approximation",
            "fidelity_note": "Local DiM-style morph approximation for smoke tests and pipeline integration.",
            "native_repo": "Backends/sources/attack/DiM_native_upstream",
            "native_release_gate": "run_dim.py is still encrypted upstream and requires the CITeR passphrase-release process",
        },
    )


if __name__ == "__main__":
    main()
