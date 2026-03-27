#!/usr/bin/env python3
# python3 generate.py <dataset_dir> <pair_input> <output_dir> <records_out> -> MorDiff-style morph probes

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Modifiers.attack.shared import blur_array, build_base_arg_parser, ellipse_mask, finalize_generator_args
from Modifiers.attack.shared import run_generator


def parse_args() -> argparse.Namespace:
    parser = build_base_arg_parser("Generate MorDiff-style attack probes.")
    parser.add_argument("--low-freq-target-weight", type=float, default=0.5)
    parser.add_argument("--high-freq-source-weight", type=float, default=0.6)
    return finalize_generator_args(parser.parse_args())


def transform(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    low_source = blur_array(source, radius=2.4)
    low_target = blur_array(target, radius=2.4)
    source_detail = source - low_source
    target_detail = target - low_target

    low_mix = (1.0 - float(args.low_freq_target_weight)) * low_source + float(args.low_freq_target_weight) * low_target
    detail_mix = float(args.high_freq_source_weight) * source_detail + (1.0 - float(args.high_freq_source_weight)) * target_detail
    full_mix = np.clip(low_mix + detail_mix, 0.0, 1.0)

    face_mask = ellipse_mask(source.shape[0], source.shape[1], radius_x=0.34, radius_y=0.44, feather=0.24)
    background_mix = (0.8 * source) + (0.2 * low_mix)
    return background_mix * (1.0 - face_mask) + full_mix * face_mask


def main() -> None:
    args = parse_args()
    run_generator(
        method_name="mordiff_proxy",
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
