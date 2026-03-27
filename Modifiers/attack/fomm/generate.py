#!/usr/bin/env python3
# python3 generate.py <dataset_dir> <pair_input> <output_dir> <records_out> -> FOMM-style reenactment probes

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Modifiers.attack.shared import band_mask, build_base_arg_parser, ellipse_mask, finalize_generator_args
from Modifiers.attack.shared import match_color_statistics, run_generator, shift_array


def parse_args() -> argparse.Namespace:
    parser = build_base_arg_parser("Generate FOMM-style attack probes.")
    parser.add_argument("--expression-weight", type=float, default=0.38)
    parser.add_argument("--pose-shift-scale", type=float, default=0.06)
    return finalize_generator_args(parser.parse_args())


def transform(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    height, width = source.shape[:2]
    dx = int(round((target[:, :, 0].mean() - source[:, :, 0].mean()) * width * float(args.pose_shift_scale)))
    dy = int(round((target[:, :, 1].mean() - source[:, :, 1].mean()) * height * float(args.pose_shift_scale)))
    shifted_target = shift_array(target, dx=dx, dy=dy)

    face_mask = ellipse_mask(height, width, radius_x=0.28, radius_y=0.36, feather=0.2)
    mouth_mask = band_mask(height, width, y_center=0.7, band_height=0.18, feather=0.08)
    eye_mask = band_mask(height, width, y_center=0.42, band_height=0.16, feather=0.08)
    feature_mask = np.clip(np.maximum(face_mask, np.maximum(mouth_mask, eye_mask)), 0.0, 1.0)

    reenacted = match_color_statistics(source, shifted_target, feature_mask, mix=0.6)
    return source * (1.0 - feature_mask) + (
        (1.0 - float(args.expression_weight)) * source + float(args.expression_weight) * reenacted
    ) * feature_mask


def main() -> None:
    args = parse_args()
    run_generator(
        method_name="fomm_proxy",
        dataset_dir=args.dataset_dir,
        pair_input=args.pair_input,
        output_dir=args.output_dir,
        records_out=args.records_out,
        args=args,
        image_size=int(args.image_size),
        jpeg_quality=int(args.jpeg_quality),
        overwrite=bool(args.overwrite),
        transform=transform,
    )


if __name__ == "__main__":
    main()
