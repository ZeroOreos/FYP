#!/usr/bin/env python3
# Example: python scripts/evaluate_paper_breakdown.py --help
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Training.config import AttackPolicy, load_config
from Training.dataset import build_configured_class_to_idx, build_eval_loader
from Training.engine import maybe_init_distributed, maybe_wrap_ddp
from Training.evaluate import evaluate_clean, evaluate_robust
from Training.recognizers import build_recognizer_ensemble, build_surrogates, build_target_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the paper robustness breakdown: clean/trained/extra attack rows "
            "with target and recognizer-ensemble metrics."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <config.output_dir>/checkpoints/latest.pth, then newest under $FYP_OUTPUT_ROOT.",
    )
    parser.add_argument("--eval-batches", type=int, default=0, help="0 means full validation loader.")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--skip-trained-attacks", action="store_true")
    parser.add_argument("--skip-extra-attacks", action="store_true")
    parser.add_argument("--extra-steps", type=int, default=12)
    parser.add_argument("--extra-restarts", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--attack-chunk-size", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_primary() -> bool:
    return _rank() == 0


def _latest_checkpoint_under(output_root: Path) -> Path | None:
    candidates = sorted(
        output_root.glob("*/checkpoints/latest.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _infer_checkpoint(config, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    resume_from = config.resolved_resume_from()
    if resume_from is not None and resume_from.exists():
        return resume_from.expanduser().resolve()
    candidate = config.resolved_output_dir() / "checkpoints" / "latest.pth"
    if candidate.exists():
        return candidate
    output_root = Path(os.environ.get("FYP_OUTPUT_ROOT", config.resolved_output_dir().parent)).expanduser().resolve()
    return _latest_checkpoint_under(output_root)


def _load_state(module: torch.nn.Module, state_dict: dict[str, torch.Tensor], *, allow_partial: bool) -> dict[str, object]:
    if not allow_partial:
        module.load_state_dict(state_dict)
        return {"loaded_tensors": len(state_dict), "skipped_tensors": 0, "partial_load": False}

    current_state = module.state_dict()
    compatible = {
        name: tensor
        for name, tensor in state_dict.items()
        if name in current_state and tuple(current_state[name].shape) == tuple(tensor.shape)
    }
    skipped = sorted(name for name in state_dict if name not in compatible)
    current_state.update(compatible)
    module.load_state_dict(current_state)
    return {
        "loaded_tensors": len(compatible),
        "skipped_tensors": len(skipped),
        "partial_load": True,
        "skipped_tensor_names": skipped[:20],
    }


def _load_checkpoint(
    *,
    model: torch.nn.Module,
    recognizer_ensemble: torch.nn.Module | None,
    checkpoint_path: Path,
    allow_partial: bool,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    target_model = getattr(model, "module", model)
    report: dict[str, object] = {
        "target": _load_state(target_model, checkpoint["model_state_dict"], allow_partial=allow_partial)
    }
    recognizer_state = checkpoint.get("recognizer_ensemble_state_dict")
    if recognizer_ensemble is not None:
        if recognizer_state is None:
            if not allow_partial:
                raise RuntimeError(
                    "Checkpoint has no recognizer_ensemble_state_dict. "
                    "Use a fresh paper checkpoint or pass --allow-partial-checkpoint only for debugging."
                )
            report["recognizer_ensemble"] = {
                "loaded_tensors": 0,
                "skipped_tensors": 0,
                "partial_load": True,
                "missing_from_checkpoint": True,
            }
        else:
            report["recognizer_ensemble"] = _load_state(
                recognizer_ensemble,
                recognizer_state,
                allow_partial=allow_partial,
            )
    return report


def _materialize_batches(loader, max_batches: int):
    if max_batches <= 0:
        return loader
    batches = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batches.append(batch)
    return batches


def _extra_attack_policies(config, *, steps: int, restarts: int) -> list[AttackPolicy]:
    enabled = config.enabled_eval_attackers()
    eps = max((float(policy.eps) for policy in enabled), default=8.0 / 255.0)
    alpha = eps / float(max(4, steps // 2))
    return [
        AttackPolicy(
            name="pgd_strong",
            family="stress_white_box",
            kind="online",
            weight=1.0,
            enabled=True,
            eps=eps,
            alpha=alpha,
            steps=max(1, int(steps)),
            random_start=True,
            restarts=max(1, int(restarts)),
            surrogate_weights={"target": 0.5, "recognizers": 0.5},
        ),
        AttackPolicy(
            name="cw_margin",
            family="stress_margin",
            kind="online",
            weight=1.0,
            enabled=True,
            eps=eps,
            alpha=alpha,
            steps=max(1, int(steps)),
            random_start=True,
            restarts=max(1, int(restarts)),
            surrogate_weights={"target": 1.0},
        ),
    ]


def _metric_at(metrics: dict[str, Any], path: str, default=None):
    value: Any = metrics
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _row_from_metrics(
    *,
    condition: str,
    condition_family: str,
    metrics: dict[str, Any],
    recognizer_names: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "condition": condition,
        "condition_family": condition_family,
        "target_accuracy": metrics.get("accuracy"),
        "target_loss": metrics.get("loss"),
        "attack_success_rate": metrics.get("attack_success_rate"),
        "samples": metrics.get("samples"),
        "target_ece": _metric_at(metrics, "calibration.ece"),
        "target_brier": _metric_at(metrics, "calibration.brier"),
    }
    recognizer_metrics = metrics.get("recognizer_metrics") or {}
    if isinstance(recognizer_metrics, dict):
        for name in recognizer_names:
            member = recognizer_metrics.get(name) or {}
            row[f"{name}_accuracy"] = member.get("accuracy") if isinstance(member, dict) else None
            row[f"{name}_loss"] = member.get("loss") if isinstance(member, dict) else None
    row["recognizer_ensemble_accuracy"] = _metric_at(metrics, "recognizer_ensemble.fused_accuracy")
    row["recognizer_ensemble_gain_vs_best_single"] = _metric_at(
        metrics,
        "recognizer_ensemble.gain_vs_best_single",
    )
    row["recognizer_ensemble_pairwise_error_corr"] = _metric_at(
        metrics,
        "recognizer_ensemble.pairwise_error_correlation_mean",
    )
    row["recognizer_ece"] = _metric_at(metrics, "recognizer_calibration.ece")
    row["recognizer_brier"] = _metric_at(metrics, "recognizer_calibration.brier")
    return row


def _format_float(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(rows: list[dict[str, object]], recognizer_names: list[str]) -> str:
    columns = ["condition", "target_accuracy", "attack_success_rate"]
    columns.extend(f"{name}_accuracy" for name in recognizer_names)
    columns.extend(["recognizer_ensemble_accuracy", "recognizer_ensemble_gain_vs_best_single"])
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_float(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def _write_outputs(
    *,
    report: dict[str, object],
    rows: list[dict[str, object]],
    recognizer_names: list[str],
    output_json: Path | None,
    output_csv: Path | None,
    output_md: Path | None,
) -> None:
    if output_json is not None:
        output_json = output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_csv is not None:
        output_csv = output_csv.expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if output_md is not None:
        output_md = output_md.expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown_table(rows, recognizer_names), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config.expanduser().resolve())
    distributed = maybe_init_distributed(config)
    device_name = config.device
    if device_name == "auto" and distributed and torch.cuda.is_available():
        device_name = "cuda"

    class_to_idx = build_configured_class_to_idx(
        config.resolved_train_dir(),
        config.resolved_val_dir(),
        dataset_fraction=config.dataset_fraction,
        dataset_subset_seed=config.dataset_subset_seed,
        dataset_min_images_per_identity=config.dataset_min_images_per_identity,
    )
    model, device = build_target_model(
        model_name=config.target_model,
        num_classes=len(class_to_idx),
        embedding_dim=config.embedding_dim,
        device_name=device_name,
        backbone_name=config.target_backbone,
        use_gradient_checkpointing=False,
        arcface_scale=config.arcface_scale,
        arcface_margin=config.arcface_margin,
        use_partial_fc=config.use_partial_fc,
        partial_fc_negative_sample_rate=config.partial_fc_negative_sample_rate,
        sub_center_count=config.sub_center_count,
        dropout_p=config.dropout_p,
        member_weights=config.joint_pool_member_weights,
    )
    model = maybe_wrap_ddp(model, device, config, distributed)
    recognizer_ensemble = build_recognizer_ensemble(
        recognizer_specs=config.enabled_recognizers(),
        num_classes=len(class_to_idx),
        embedding_dim=config.embedding_dim,
        device=device,
        arcface_scale=config.arcface_scale,
        arcface_margin=config.arcface_margin,
        use_partial_fc=config.use_partial_fc,
        partial_fc_negative_sample_rate=(
            config.partial_fc_negative_sample_rate
            if config.recognizer_partial_fc_negative_sample_rate is None
            else float(config.recognizer_partial_fc_negative_sample_rate)
        ),
        sub_center_count=config.sub_center_count,
        weight_strategy=config.recognizer_weight_strategy,
    )

    checkpoint_path = _infer_checkpoint(config, args.checkpoint)
    if checkpoint_path is None:
        raise SystemExit("No checkpoint found. Pass --checkpoint for the run you want to evaluate.")
    checkpoint_load = _load_checkpoint(
        model=model,
        recognizer_ensemble=recognizer_ensemble,
        checkpoint_path=checkpoint_path,
        allow_partial=bool(args.allow_partial_checkpoint),
    )
    if _is_primary():
        print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
        print(f"[INFO] Checkpoint load: {checkpoint_load}")

    _, val_loader = build_eval_loader(
        data_dir=config.resolved_val_dir(),
        class_to_idx=class_to_idx,
        image_size=config.image_size,
        batch_size=int(args.batch_size or config.batch_size),
        num_workers=config.num_workers,
        distributed=distributed,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
    )
    eval_loader = _materialize_batches(val_loader, int(args.eval_batches))
    surrogates = build_surrogates(config.surrogate_models, getattr(model, "module", model), device)
    recognizer_names = [policy.name.strip().lower() for policy in config.enabled_recognizers()]
    policies: list[AttackPolicy] = []
    if not args.skip_trained_attacks:
        policies.extend(config.enabled_eval_attackers())
    if not args.skip_extra_attacks:
        policies.extend(
            _extra_attack_policies(
                config,
                steps=int(args.extra_steps),
                restarts=int(args.extra_restarts),
            )
        )

    rows: list[dict[str, object]] = []
    clean_metrics = evaluate_clean(
        model,
        eval_loader,
        device,
        recognizer_ensemble=recognizer_ensemble,
        progress_desc="Paper breakdown clean" if _is_primary() else None,
    )
    rows.append(
        _row_from_metrics(
            condition="clean",
            condition_family="clean",
            metrics=clean_metrics,
            recognizer_names=recognizer_names,
        )
    )

    robust_by_condition: dict[str, dict[str, object]] = {}
    attack_chunk_size = args.attack_chunk_size if args.attack_chunk_size is not None else config.attack_chunk_size
    for policy in policies:
        metrics = evaluate_robust(
            model=model,
            loader=eval_loader,
            device=device,
            policy=policy,
            surrogates=surrogates,
            recognizer_ensemble=recognizer_ensemble,
            image_size=config.image_size,
            attack_chunk_size=attack_chunk_size,
            progress_desc=f"Paper breakdown {policy.name}" if _is_primary() else None,
        )
        robust_by_condition[policy.name] = metrics
        rows.append(
            _row_from_metrics(
                condition=policy.name,
                condition_family=policy.family,
                metrics=metrics,
                recognizer_names=recognizer_names,
            )
        )

    report: dict[str, object] = {
        "config": str(args.config.expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_load": checkpoint_load,
        "distributed": bool(distributed),
        "eval_batches": int(args.eval_batches),
        "class_count": int(len(class_to_idx)),
        "recognizers": recognizer_names,
        "conditions": [
            {"name": policy.name, "family": policy.family, "steps": policy.steps, "restarts": policy.restarts}
            for policy in policies
        ],
        "clean": clean_metrics,
        "robust_by_condition": robust_by_condition,
        "recognizer_attack_breakdown": rows,
        "paper_table": _markdown_table(rows, recognizer_names),
    }

    if _is_primary():
        print(report["paper_table"])
        _write_outputs(
            report=report,
            rows=rows,
            recognizer_names=recognizer_names,
            output_json=args.output_json,
            output_csv=args.output_csv,
            output_md=args.output_md,
        )


if __name__ == "__main__":
    main()
