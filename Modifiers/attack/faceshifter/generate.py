#!/usr/bin/env python3
# python3 generate.py <dataset_dir> <pair_input> <output_dir> <records_out> -> FaceShifter-style swap probes

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Modifiers.attack.shared import blur_array, build_base_arg_parser, ellipse_mask, finalize_generator_args
from Modifiers.attack.shared import match_color_statistics, run_generator


def parse_args() -> argparse.Namespace:
    parser = build_base_arg_parser("Generate FaceShifter-style attack probes.")
    parser.add_argument("--primary-weight", type=float, default=0.82)
    parser.add_argument("--context-weight", type=float, default=0.24)
    return finalize_generator_args(parser.parse_args())


def transform(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    primary_mask = ellipse_mask(source.shape[0], source.shape[1], radius_x=0.34, radius_y=0.45, feather=0.22)
    context_mask = ellipse_mask(source.shape[0], source.shape[1], radius_x=0.42, radius_y=0.5, feather=0.24)
    aligned_target = match_color_statistics(source, target, primary_mask, mix=0.9)
    refined_target = blur_array(aligned_target, radius=0.6)
    primary_face = (1.0 - float(args.primary_weight)) * source + float(args.primary_weight) * refined_target
    context_face = (1.0 - float(args.context_weight)) * source + float(args.context_weight) * refined_target
    output = source * (1.0 - context_mask) + context_face * context_mask
    return output * (1.0 - primary_mask) + primary_face * primary_mask


def main() -> None:
    args = parse_args()
    run_generator(
        method_name="faceshifter_proxy",
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
