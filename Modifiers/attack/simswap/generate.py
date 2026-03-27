#!/usr/bin/env python3
# python3 generate.py <dataset_dir> <pair_input> <output_dir> <records_out> -> SimSwap-style swap probes

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Modifiers.attack.shared import build_base_arg_parser, ellipse_mask, finalize_generator_args
from Modifiers.attack.shared import match_color_statistics, run_generator


def parse_args() -> argparse.Namespace:
    parser = build_base_arg_parser("Generate SimSwap-style attack probes.")
    parser.add_argument("--face-weight", type=float, default=0.72)
    parser.add_argument("--color-match", type=float, default=0.85)
    parser.add_argument("--mask-radius-x", type=float, default=0.31)
    parser.add_argument("--mask-radius-y", type=float, default=0.41)
    return finalize_generator_args(parser.parse_args())


def transform(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    mask = ellipse_mask(
        source.shape[0],
        source.shape[1],
        radius_x=float(args.mask_radius_x),
        radius_y=float(args.mask_radius_y),
        feather=0.18,
    )
    aligned_target = match_color_statistics(source, target, mask, mix=float(args.color_match))
    swapped_face = (1.0 - float(args.face_weight)) * source + float(args.face_weight) * aligned_target
    return source * (1.0 - mask) + swapped_face * mask


def main() -> None:
    args = parse_args()
    run_generator(
        method_name="simswap_proxy",
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
