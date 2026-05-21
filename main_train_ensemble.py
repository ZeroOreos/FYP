#!/usr/bin/env python3
# Example: python main_train_ensemble.py --config Training/generated/paper_ladder/full-ensemble.json
"""Ensemble face-recognition training entry point.

Primary attackers are generated natively during training.
Surrogate attackers should be materialized into cached dataset roots and referenced in config.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Training.config import EnsembleTrainingConfig


def _sanitize_cuda_alloc_conf_env() -> None:
    if os.environ.get("FYP_KEEP_EXPANDABLE_SEGMENTS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    raw = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if not raw:
        return
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    filtered = [part for part in parts if not part.lower().startswith("expandable_segments:")]
    if filtered == parts:
        return
    if filtered:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(filtered)
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ensemble face-recognition training.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--train-dir", type=Path, default=None, help="Training dataset root.")
    parser.add_argument("--val-dir", type=Path, default=None, help="Validation dataset root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Run output directory.")
    parser.add_argument(
        "--target-model",
        choices=("arcface", "cosface", "curricularface", "joint_pool"),
        default=None,
        help="Trainable margin-loss recognizer to run.",
    )
    parser.add_argument(
        "--recognizers-mode",
        choices=("serial_target", "joint_train"),
        default=None,
        help="Use serial_target for the mainline paper path; reserve joint_train for ablation/ceiling runs.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--target-backbone",
        choices=("resnet18", "resnet50", "iresnet100"),
        default=None,
        help="Recognizer backbone to use. Keep resnet18 for cheap implementation tests; use resnet50 or iresnet100 for heavier runs.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=None)
    parser.add_argument("--pairing-strategy", default=None, help="Override hard-pair mining strategy.")
    parser.add_argument("--hard-pair-fraction", type=float, default=None, help="Override hard-pair mining fraction.")
    parser.add_argument("--hard-pair-weight", type=float, default=None, help="Override additional hard-sample loss weight.")
    parser.add_argument("--sub-centers", type=int, default=None, help="ArcFace sub-center count. Use 1 to disable.")
    parser.add_argument("--disable-curriculum", action="store_true", help="Disable clean/adversarial curriculum scheduling.")
    parser.add_argument("--cached-attack-root", action="append", default=None)
    parser.add_argument(
        "--ceiling-placeholder",
        action="store_true",
        help="Use the WF42M / ArcFace ceiling config with the local proxy backbone and embedding dimension.",
    )
    parser.add_argument("--clean-only", action="store_true", help="Skip ensemble attacks and run clean training.")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> "EnsembleTrainingConfig":
    from Training.config import default_mode_b_config, load_config, published_ceiling_placeholder_config

    if args.config is not None:
        config = load_config(args.config.resolve())
    else:
        if args.train_dir is None or args.val_dir is None or args.output_dir is None:
            raise SystemExit(
                "Either provide --config or provide --train-dir, --val-dir, and --output-dir."
            )
        builder = published_ceiling_placeholder_config if args.ceiling_placeholder else default_mode_b_config
        config = builder(
            train_dir=args.train_dir.resolve(),
            val_dir=args.val_dir.resolve(),
            output_dir=args.output_dir.resolve(),
        )

    if args.epochs is not None:
        config.epochs = int(args.epochs)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.target_model is not None:
        config.target_model = args.target_model
        if args.target_model == "joint_pool" and args.recognizers_mode is None:
            config.recognizers_mode = "joint_train"
    if args.recognizers_mode is not None:
        config.recognizers_mode = args.recognizers_mode
    if args.target_backbone is not None:
        config.target_backbone = args.target_backbone
    if args.device is not None:
        config.device = args.device
    if args.pairing_strategy is not None:
        config.pairing_strategy = args.pairing_strategy
    if args.hard_pair_fraction is not None:
        config.hard_pair_fraction = float(args.hard_pair_fraction)
    if args.hard_pair_weight is not None:
        config.hard_pair_weight = float(args.hard_pair_weight)
    if args.sub_centers is not None:
        config.sub_center_count = max(1, int(args.sub_centers))
    if args.disable_curriculum:
        config.curriculum_enabled = False
    if args.clean_only:
        config.clean_only = True
        config.primary_attackers = []
        config.surrogate_attackers = []
        config.surrogate_models = []
    if args.cached_attack_root:
        for policy in config.surrogate_attackers:
            if policy.kind == "cached":
                policy.cache_roots = [str(Path(item).resolve()) for item in args.cached_attack_root]
    return config


def main() -> None:
    _sanitize_cuda_alloc_conf_env()
    args = parse_args()
    config = build_config_from_args(args)
    from Training.engine import run_training

    resume_from = args.resume_from.resolve() if args.resume_from is not None else config.resolved_resume_from()
    summary = run_training(config, resume_from=resume_from)
    print(f"[INFO] Training complete: {summary['output_dir']}")


if __name__ == "__main__":
    main()
