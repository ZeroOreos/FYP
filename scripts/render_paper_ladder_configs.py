#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "Training"


@dataclass(frozen=True)
class LadderRunSpec:
    slug: str
    label: str
    role: str
    base_config: str


LADDER_SPECS = (
    LadderRunSpec(
        slug="full-ensemble",
        label="Full Ensemble",
        role="primary",
        base_config="arcface_webface4m_paper_full_ensemble.json",
    ),
    LadderRunSpec(
        slug="baseline-clean",
        label="Baseline Clean",
        role="baseline",
        base_config="arcface_webface4m_paper_clean_baseline.json",
    ),
    LadderRunSpec(
        slug="recognizer-ensemble-only",
        label="Recognizer Ensemble Only",
        role="ablation",
        base_config="arcface_webface4m_paper_recognizer_ensemble_only.json",
    ),
    LadderRunSpec(
        slug="attack-ensemble-only",
        label="Attack Ensemble Only",
        role="ablation",
        base_config="arcface_webface4m_paper_attack_ensemble_only.json",
    ),
    LadderRunSpec(
        slug="single-attack-only",
        label="Single Attack Only",
        role="ablation",
        base_config="arcface_webface4m_paper_single_attack_only.json",
    ),
    LadderRunSpec(
        slug="full-ensemble-test",
        label="Full Ensemble Test",
        role="integration",
        base_config="arcface_webface4m_full_ensemble_test.json",
    ),
)


PAPER_EVAL_ATTACKERS = [
    {
        "name": "pgd",
        "family": "primary_white_box",
        "kind": "online",
        "weight": 0.5,
        "enabled": True,
        "eps": 0.03137254901960784,
        "alpha": 0.00392156862745098,
        "steps": 6,
        "random_start": True,
        "restarts": 2,
        "surrogate_weights": {"target": 1.0},
        "cache_roots": [],
        "refresh_every_epochs": None,
    },
    {
        "name": "bpfa",
        "family": "primary_transfer",
        "kind": "online",
        "weight": 0.2,
        "enabled": True,
        "eps": 0.03137254901960784,
        "alpha": 0.00196078431372549,
        "steps": 8,
        "random_start": True,
        "restarts": 2,
        "surrogate_weights": {"target": 1.0},
        "cache_roots": [],
        "refresh_every_epochs": None,
    },
    {
        "name": "dfanet",
        "family": "primary_feature",
        "kind": "online",
        "weight": 0.3,
        "enabled": True,
        "eps": 0.03137254901960784,
        "alpha": 0.00392156862745098,
        "steps": 6,
        "random_start": True,
        "restarts": 2,
        "surrogate_weights": {"target": 1.0},
        "cache_roots": [],
        "refresh_every_epochs": None,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the final paper ladder configs and a manifest.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Training" / "generated" / "paper_ladder",
        help="Directory to write rendered configs into.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "Training" / "generated" / "paper_ladder_manifest.json",
        help="JSON manifest describing the selected paper runs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/wen/data/runs"),
        help="Run output root to embed into the rendered configs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Defaults to $FYP_WEBFACE4M_ROOT or <repo>/Dataset/WebFace4M.",
    )
    parser.add_argument("--dataset-fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _selected_specs() -> list[LadderRunSpec]:
    return list(LADDER_SPECS)


def _render_config(
    *,
    spec: LadderRunSpec,
    dataset_fraction: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    data_root: Path,
    output_root: Path,
) -> dict[str, object]:
    src = TRAINING_ROOT / spec.base_config
    data = json.loads(src.read_text(encoding="utf-8"))
    output_dir = output_root / spec.slug

    data["dataset_fraction"] = float(dataset_fraction)
    data["dataset_subset_seed"] = int(seed)
    data["seed"] = int(seed)
    data["resume_from"] = None
    data["train_dir"] = str((data_root / "manifests" / "train.jsonl").resolve())
    data["val_dir"] = str((data_root / "manifests" / "val.jsonl").resolve())
    pairs_root = data_root.parent / "pairs"
    for pairs_key in ("val_pairs_path", "test_pairs_path"):
        raw_pairs_path = data.get(pairs_key)
        if not isinstance(raw_pairs_path, str) or not raw_pairs_path.strip():
            continue
        normalized_pairs_path = raw_pairs_path.replace("\\", "/")
        marker = "/Dataset/pairs/"
        if marker not in normalized_pairs_path:
            continue
        pair_suffix = normalized_pairs_path.split(marker, 1)[1]
        data[pairs_key] = str((pairs_root / pair_suffix).resolve())
    data["target_backbone"] = "iresnet100"
    data["batch_size"] = int(batch_size)
    data["clean_warmup_batch_size"] = int(data.get("clean_warmup_batch_size") or 128)
    data["shallow_adv_batch_size"] = int(data.get("shallow_adv_batch_size") or 48)
    data["full_adv_batch_size"] = int(data.get("full_adv_batch_size") or 32)
    data["attack_chunk_size"] = 32
    if spec.role != "integration":
        data["full_robust_eval_every_epochs"] = 8
    data["checkpoint_every"] = 1
    data["gradient_accumulation_steps"] = 1
    data["num_workers"] = int(num_workers)
    data["use_distributed"] = True
    data["use_sync_batchnorm"] = False
    data["use_gradient_checkpointing"] = False
    data["use_torch_compile"] = False
    data["use_channels_last"] = True
    data["ddp_no_sync_accumulation"] = False
    data["runtime_profile"] = "paper_full" if spec.role != "integration" else "custom"
    data["device"] = "cuda"
    data["output_dir"] = str(output_dir)
    data["eval_attackers"] = [dict(policy) for policy in PAPER_EVAL_ATTACKERS]
    if str(data.get("target_model", "")).strip().lower() not in {"joint_pool", "jointpool", "pool"}:
        data.pop("joint_pool_member_weights", None)
    if str(data.get("target_model", "")).strip().lower() in {"joint_pool", "jointpool", "pool"}:
        data["recognizers_mode"] = "joint_train"
    return data


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = PROJECT_ROOT
    data_root = (args.data_root or Path(os.environ.get("FYP_WEBFACE4M_ROOT", repo_root / "Dataset" / "WebFace4M"))).resolve()

    manifest_runs: list[dict[str, object]] = []
    for spec in _selected_specs():
        data = _render_config(
            spec=spec,
            dataset_fraction=args.dataset_fraction,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            data_root=data_root,
            output_root=args.output_root,
        )
        config_path = output_dir / f"{spec.slug}.json"
        config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        manifest_runs.append(
            {
                "slug": spec.slug,
                "label": spec.label,
                "role": spec.role,
                "base_config": str((TRAINING_ROOT / spec.base_config).resolve()),
                "generated_config": str(config_path),
                "output_dir": data["output_dir"],
            }
        )
        print(config_path)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_fraction": float(args.dataset_fraction),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "runs": manifest_runs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
