#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Training.config import load_config
from Training.dataset import build_configured_class_to_idx, build_eval_loader
from Training.engine import maybe_init_distributed, maybe_wrap_ddp
from Training.evaluate import (
    choose_eval_policy,
    evaluate_clean,
    evaluate_robust,
    evaluate_robust_all,
    evaluate_verification_pairs,
)
from Training.recognizers import build_recognizer_ensemble, build_surrogates, build_target_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an evaluation preflight stack against any recent checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint. Defaults to newest latest.pth under $FYP_OUTPUT_ROOT.")
    parser.add_argument("--pairs", type=Path, default=None, help="Optional verification pair bundle. Defaults to config.val_pairs_path.")
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=2,
        help="Number of validation batches to use for clean/robust probes. Use 0 or less for the full validation loader.",
    )
    parser.add_argument("--allow-partial-checkpoint", action="store_true", default=True)
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-robust", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_primary() -> bool:
    return _rank() == 0


def _latest_checkpoint_under(output_root: Path) -> Path | None:
    candidates = sorted(output_root.glob("*/checkpoints/latest.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
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


def _load_checkpoint_for_probe(
    model,
    recognizer_ensemble,
    checkpoint_path: Path,
    *,
    allow_partial: bool,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    target_model = getattr(model, "module", model)
    state_dict = checkpoint["model_state_dict"]
    if not allow_partial:
        target_model.load_state_dict(state_dict)
        recognizer_state = checkpoint.get("recognizer_ensemble_state_dict")
        if recognizer_ensemble is not None and recognizer_state is not None:
            recognizer_ensemble.load_state_dict(recognizer_state)
        return {
            "loaded_tensors": len(state_dict),
            "skipped_tensors": 0,
            "partial_load": False,
            "recognizer_loaded": bool(recognizer_ensemble is not None and recognizer_state is not None),
        }
    current_state = target_model.state_dict()
    compatible = {
        name: tensor
        for name, tensor in state_dict.items()
        if name in current_state and tuple(current_state[name].shape) == tuple(tensor.shape)
    }
    skipped = sorted(name for name in state_dict if name not in compatible)
    current_state.update(compatible)
    target_model.load_state_dict(current_state)
    recognizer_loaded = False
    recognizer_skipped: list[str] = []
    recognizer_state = checkpoint.get("recognizer_ensemble_state_dict")
    if recognizer_ensemble is not None and recognizer_state is not None:
        current_recognizer_state = recognizer_ensemble.state_dict()
        compatible_recognizer = {
            name: tensor
            for name, tensor in recognizer_state.items()
            if name in current_recognizer_state and tuple(current_recognizer_state[name].shape) == tuple(tensor.shape)
        }
        recognizer_skipped = sorted(name for name in recognizer_state if name not in compatible_recognizer)
        current_recognizer_state.update(compatible_recognizer)
        recognizer_ensemble.load_state_dict(current_recognizer_state)
        recognizer_loaded = bool(compatible_recognizer)
    return {
        "loaded_tensors": len(compatible),
        "skipped_tensors": len(skipped),
        "partial_load": True,
        "skipped_tensor_names": skipped[:20],
        "recognizer_loaded": recognizer_loaded,
        "recognizer_skipped_tensors": len(recognizer_skipped),
        "recognizer_skipped_tensor_names": recognizer_skipped[:20],
    }


def _skipped_tensors(checkpoint_load: dict[str, object] | None) -> set[str]:
    if not isinstance(checkpoint_load, dict):
        return set()
    names = checkpoint_load.get("skipped_tensor_names")
    if not isinstance(names, list):
        return set()
    return {str(name) for name in names}


def _robust_classification_load_error(checkpoint_load: dict[str, object] | None) -> str | None:
    skipped = _skipped_tensors(checkpoint_load)
    if "margin.weight" in skipped:
        return (
            "target margin.weight was skipped during checkpoint load; robust classification metrics "
            "would use an untrained classifier head. Restore the original class mapping/manifest or "
            "rerun without partial checkpoint loading."
        )
    return None


def _materialize_batches(loader, max_batches: int):
    if max_batches <= 0:
        return loader
    batches = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batches.append(batch)
    return batches


def _pair_shape(pairs_path: Path) -> dict[str, object]:
    import numpy as np

    pair_data = dict(np.load(pairs_path, allow_pickle=True))
    img1_paths = [str(item) for item in pair_data["img1_paths"].tolist()]
    img2_paths = [str(item) for item in pair_data["img2_paths"].tolist()]
    unique_refs = set(img1_paths + img2_paths)
    shard_refs = sum(1 for item in unique_refs if "::" in item)
    image_refs = len(unique_refs) - shard_refs
    return {
        "pairs": len(img1_paths),
        "unique_refs": len(unique_refs),
        "image_refs": image_refs,
        "shard_refs": shard_refs,
    }


def _run_stage(name: str, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return {"ok": True, "elapsed_s": elapsed, "result": result}


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

    _, val_loader = build_eval_loader(
        data_dir=config.resolved_val_dir(),
        class_to_idx=class_to_idx,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        distributed=distributed,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
    )
    surrogates = build_surrogates(config.surrogate_models, getattr(model, "module", model), device)
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
    checkpoint_load = None
    if checkpoint_path is not None:
        checkpoint_load = _load_checkpoint_for_probe(
            model,
            recognizer_ensemble,
            checkpoint_path,
            allow_partial=bool(args.allow_partial_checkpoint),
        )
        if _is_primary():
            print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
            print(f"[INFO] Checkpoint load: {checkpoint_load}")
    eval_policy = choose_eval_policy(config.all_eval_attackers(), config.eval_attack_name)

    report: dict[str, object] = {
        "config": str(args.config.resolve()),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_load": checkpoint_load,
        "device": str(device),
        "distributed": bool(distributed),
        "eval_batches": int(args.eval_batches),
        "class_count": int(len(class_to_idx)),
        "stages": {},
    }

    stages = report["stages"]
    assert isinstance(stages, dict)
    clean_loader = _materialize_batches(val_loader, args.eval_batches)
    if not args.skip_clean:
        stages["clean"] = _run_stage(
            "clean",
            lambda: evaluate_clean(
                model,
                clean_loader,
                device,
                recognizer_ensemble=recognizer_ensemble,
                progress_desc="Probe clean" if _is_primary() else None,
            ),
        )

    attackers = config.all_eval_attackers()
    robust_load_error = _robust_classification_load_error(checkpoint_load)
    if not args.skip_robust and attackers and robust_load_error is None:
        robust_single_loader = _materialize_batches(val_loader, args.eval_batches)
        stages["robust_single"] = _run_stage(
            "robust_single",
            lambda: evaluate_robust(
                model=model,
                loader=robust_single_loader,
                device=device,
                policy=eval_policy,
                surrogates=surrogates,
                recognizer_ensemble=recognizer_ensemble,
                image_size=config.image_size,
                attack_chunk_size=config.attack_chunk_size,
                progress_desc="Probe robust single" if _is_primary() else None,
            ),
        )
        robust_all_loader = _materialize_batches(val_loader, args.eval_batches)
        stages["robust_all"] = _run_stage(
            "robust_all",
            lambda: evaluate_robust_all(
                model=model,
                loader=robust_all_loader,
                device=device,
                policies=attackers,
                surrogates=surrogates,
                recognizer_ensemble=recognizer_ensemble,
                image_size=config.image_size,
                attack_chunk_size=config.attack_chunk_size,
                progress_prefix="Probe robust all" if _is_primary() else None,
                distributed=distributed,
            ),
        )
    elif args.skip_robust:
        stages["robust"] = {"ok": True, "skipped": True, "reason": "skip_robust"}
    elif robust_load_error is not None:
        stages["robust"] = {
            "ok": False,
            "skipped": True,
            "reason": "invalid_partial_checkpoint_for_robust_classification",
            "error": robust_load_error,
        }
    else:
        stages["robust"] = {"ok": True, "skipped": True, "reason": "no_attackers"}

    if not args.skip_verification and config.val_pairs_path is not None:
        pairs_path = (args.pairs.expanduser().resolve() if args.pairs is not None else Path(config.val_pairs_path).resolve())
        if pairs_path.exists():
            report["verification_pairs"] = _pair_shape(pairs_path)
            stages["verification"] = _run_stage(
                "verification",
                lambda: evaluate_verification_pairs(
                    model=model,
                    pairs_path=pairs_path,
                    device=device,
                    progress_desc="Probe verification" if _is_primary() else None,
                ),
            )
        else:
            stages["verification"] = {"ok": False, "error": f"missing_pairs:{pairs_path}"}

    if _is_primary():
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.output_json is not None:
            args.output_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            args.output_json.expanduser().resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
