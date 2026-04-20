#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "Training"

TEMPLATE_NAMES = [
    "joint_pool_webface4m_paper_clean_baseline.json",
    "joint_pool_webface4m_paper_main_defended.json",
    "joint_pool_webface4m_paper_ablation_no_transfer.json",
    "joint_pool_webface4m_paper_ablation_no_feature.json",
    "joint_pool_webface4m_paper_ablation_no_rotation.json",
]


def _embedding_dim(backbone: str) -> int:
    return 512 if backbone == "iresnet100" else 256


def _default_batch_size(backbone: str, device: str) -> int:
    if device in {"mps", "cpu"}:
        return {
            "resnet18": 64,
            "resnet50": 32,
            "iresnet100": 16,
        }[backbone]
    return {
        "resnet18": 128,
        "resnet50": 64,
        "iresnet100": 32,
    }[backbone]


def _default_num_workers(device: str) -> int:
    if device in {"mps", "cpu"}:
        return 0
    return 2


def _default_gradient_checkpointing(backbone: str, device: str) -> bool:
    if device in {"mps", "cpu"}:
        return False
    return backbone == "iresnet100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render final paper ladder configs with only backbone and dataset size varied.")
    parser.add_argument("--backbone", choices=("resnet18", "resnet50", "iresnet100"), default="iresnet100")
    parser.add_argument("--dataset-fraction", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None, help="Optional explicit override. Defaults by backbone.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "paper_ladder_configs",
        help="Directory to write rendered configs into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size if args.batch_size is not None else _default_batch_size(args.backbone, args.device)
    embedding_dim = _embedding_dim(args.backbone)
    fraction_tag = str(args.dataset_fraction).replace(".", "p")

    for name in TEMPLATE_NAMES:
        src = TRAINING_ROOT / name
        data = json.loads(src.read_text(encoding="utf-8"))
        data["target_backbone"] = args.backbone
        data["embedding_dim"] = embedding_dim
        data["dataset_fraction"] = float(args.dataset_fraction)
        data["dataset_subset_seed"] = int(args.seed)
        data["batch_size"] = int(batch_size)
        data["device"] = args.device
        data["num_workers"] = _default_num_workers(args.device)
        data["use_gradient_checkpointing"] = _default_gradient_checkpointing(args.backbone, args.device)

        output_run_dir = Path(data["output_dir"])
        data["output_dir"] = str(
            output_run_dir.parent / f"{output_run_dir.name}_{args.backbone}_{fraction_tag}"
        )

        dst = output_dir / name.replace(".json", f"_{args.backbone}_{fraction_tag}.json")
        dst.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(dst)


if __name__ == "__main__":
    main()
