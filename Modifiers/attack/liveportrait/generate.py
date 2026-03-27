#!/usr/bin/env python3
# python3 generate.py <dataset_dir> <pair_input> <output_dir> <records_out> -> LivePortrait-style reenactment probes

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Modifiers.attack.shared import band_mask, blur_array, build_base_arg_parser, ellipse_mask
from Modifiers.attack.shared import finalize_generator_args, match_color_statistics, run_generator, shift_array


def parse_args() -> argparse.Namespace:
    parser = build_base_arg_parser("Generate LivePortrait-style attack probes.")
    parser.add_argument("--expression-weight", type=float, default=0.5)
    parser.add_argument("--context-weight", type=float, default=0.18)
    parser.add_argument("--pose-shift-scale", type=float, default=0.09)
    return finalize_generator_args(parser.parse_args())


def transform(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    height, width = source.shape[:2]
    dx = int(round((target[:, :, 2].mean() - source[:, :, 2].mean()) * width * float(args.pose_shift_scale)))
    dy = int(round((target.mean() - source.mean()) * height * float(args.pose_shift_scale)))
    shifted_target = blur_array(shift_array(target, dx=dx, dy=dy), radius=0.7)

    face_mask = ellipse_mask(height, width, radius_x=0.32, radius_y=0.42, feather=0.22)
    mouth_mask = band_mask(height, width, y_center=0.72, band_height=0.18, feather=0.08)
    eye_mask = band_mask(height, width, y_center=0.44, band_height=0.18, feather=0.08)
    expression_mask = np.clip(np.maximum(face_mask, np.maximum(mouth_mask, eye_mask)), 0.0, 1.0)
    context_mask = ellipse_mask(height, width, radius_x=0.4, radius_y=0.5, feather=0.28)

    reenacted = match_color_statistics(source, shifted_target, expression_mask, mix=0.72)
    context = (1.0 - float(args.context_weight)) * source + float(args.context_weight) * shifted_target
    output = source * (1.0 - context_mask) + context * context_mask
    expressive = (1.0 - float(args.expression_weight)) * source + float(args.expression_weight) * reenacted
    return output * (1.0 - expression_mask) + expressive * expression_mask


def main() -> None:
    args = parse_args()
    run_generator(
        method_name="liveportrait_proxy",
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
