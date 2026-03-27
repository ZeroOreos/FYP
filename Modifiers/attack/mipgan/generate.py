#!/usr/bin/env python3
# python3 generate.py <dataset_dir> <pair_input> <output_dir> <records_out> -> MIPGAN-style morph probes

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
    parser = build_base_arg_parser("Generate MIPGAN-style attack probes.")
    parser.add_argument("--global-target-weight", type=float, default=0.46)
    parser.add_argument("--face-target-weight", type=float, default=0.52)
    return finalize_generator_args(parser.parse_args())


def transform(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    global_mix = (1.0 - float(args.global_target_weight)) * source + float(args.global_target_weight) * target
    global_mix = blur_array(global_mix, radius=0.8)
    face_mask = ellipse_mask(source.shape[0], source.shape[1], radius_x=0.33, radius_y=0.43, feather=0.22)
    aligned_target = match_color_statistics(source, target, face_mask, mix=0.7)
    face_mix = (1.0 - float(args.face_target_weight)) * source + float(args.face_target_weight) * aligned_target
    return global_mix * (1.0 - face_mask) + face_mix * face_mask


def main() -> None:
    args = parse_args()
    run_generator(
        method_name="mipgan_proxy",
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
