#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from Training.attacks import generate_attack_batch
from Training.config import AttackPolicy, load_config
from Training.dataset import build_class_to_idx, build_eval_loader
from Training.evaluate import _load_image_from_reference, evaluate_clean, evaluate_robust_all, verification_metrics_from_scores
from Training.recognizers import TargetSelfSurrogate, build_target_model


TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a comparison ladder report from completed training runs.")
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        required=True,
        help="Training run directory to include. Pass multiple times.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("TrainingRuns/comparison_ladder.json"),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("TrainingRuns/comparison_ladder.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Torch device to use for posthoc evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for posthoc evaluation passes.",
    )
    return parser.parse_args()


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pth"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir}")
    return checkpoints[-1]


def _load_history(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "history.json").read_text(encoding="utf-8"))


def _parse_training_seconds(log_path: Path) -> float | None:
    first_ts = None
    last_ts = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = TIMESTAMP_RE.match(line)
        if match is None:
            continue
        ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        if first_ts is None:
            first_ts = ts
        last_ts = ts
    if first_ts is None or last_ts is None:
        return None
    return float((last_ts - first_ts).total_seconds())


def _load_manifest_label_maps(*manifest_paths: Path) -> tuple[dict[str, int], dict[str, str]]:
    label_by_source: dict[str, int] = {}
    rel_path_by_source: dict[str, str] = {}
    for manifest_path in manifest_paths:
        if manifest_path is None or not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                source = f"{Path(item['shard_path']).resolve()}::{item['key']}.jpg"
                label_by_source[source] = int(item["label_idx"])
                rel_path_by_source[source] = str(item["rel_path"])
    return label_by_source, rel_path_by_source


def _load_model_for_run(run_dir: Path, device: torch.device):
    config = load_config(run_dir / "config.snapshot.json")
    class_to_idx = build_class_to_idx(Path(config.train_dir))
    model, _ = build_target_model(
        model_name=config.target_model,
        num_classes=len(class_to_idx),
        embedding_dim=config.embedding_dim,
        device_name=str(device),
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
    checkpoint = torch.load(_latest_checkpoint(run_dir), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return config, model


def _pair_bundle(pairs_path: Path) -> dict[str, object]:
    pair_data = dict(np.load(pairs_path, allow_pickle=True))
    img1_paths = [str(item) for item in pair_data["img1_paths"].tolist()]
    img2_paths = [str(item) for item in pair_data["img2_paths"].tolist()]
    labels = torch.as_tensor(pair_data["labels"].astype("int64"))
    return {
        "img1_paths": img1_paths,
        "img2_paths": img2_paths,
        "labels": labels,
    }


def _embed_paths(
    model,
    paths: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        tensors = []
        for path in batch_paths:
            image = _load_image_from_reference(path)
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append(tensor)
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            embeddings = model.forward_embeddings(batch).detach().cpu()
        for path, embedding in zip(batch_paths, embeddings):
            cache[path] = embedding
    return cache


def _compute_pair_scores(
    left_embeddings: dict[str, torch.Tensor],
    right_embeddings: dict[str, torch.Tensor],
    img1_paths: list[str],
    img2_paths: list[str],
) -> torch.Tensor:
    scores = []
    for path_a, path_b in zip(img1_paths, img2_paths):
        emb_a = torch.nn.functional.normalize(left_embeddings[path_a], dim=0)
        emb_b = torch.nn.functional.normalize(right_embeddings[path_b], dim=0)
        scores.append(torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0), dim=1).squeeze(0))
    return torch.stack(scores)


def _choose_operating_threshold(clean_metrics: dict[str, object]) -> float:
    threshold = clean_metrics.get("threshold@far=0.01")
    supported = clean_metrics.get("far_supported@0.01")
    if supported and threshold is not None:
        return float(threshold)
    return float(clean_metrics["best_threshold"])


def _pair_operating_metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    positive = labels == 1
    negative = labels == 0
    predictions = scores >= threshold
    tar = float((predictions[positive]).float().mean().item()) if positive.any() else 0.0
    far = float((predictions[negative]).float().mean().item()) if negative.any() else 0.0
    impersonation_success = far
    dodging_success = float((~predictions[positive]).float().mean().item()) if positive.any() else 0.0
    overall_asr = (
        float((predictions[negative]).sum().item()) + float((~predictions[positive]).sum().item())
    ) / float(max(1, labels.numel()))
    return {
        "operating_threshold": float(threshold),
        "tar": tar,
        "far": far,
        "impersonation_success_rate": impersonation_success,
        "dodging_success_rate": dodging_success,
        "attack_success_rate": overall_asr,
    }


def _attack_unique_paths(
    *,
    source_model,
    target_model,
    device: torch.device,
    batch_size: int,
    unique_paths: list[str],
    label_by_source: dict[str, int],
    rel_path_by_source: dict[str, str],
    policy: AttackPolicy,
) -> dict[str, torch.Tensor]:
    attacked: dict[str, torch.Tensor] = {}
    surrogate_map = {"target": TargetSelfSurrogate(source_model)}
    for start in range(0, len(unique_paths), batch_size):
        batch_paths = unique_paths[start:start + batch_size]
        tensors = []
        labels = []
        rel_paths = []
        for path in batch_paths:
            if path not in label_by_source:
                raise KeyError(f"Missing manifest label for pair image: {path}")
            image = _load_image_from_reference(path)
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(array).permute(2, 0, 1))
            labels.append(int(label_by_source[path]))
            rel_paths.append(rel_path_by_source.get(path, Path(path).name))
        batch = torch.stack(tensors).to(device)
        label_tensor = torch.as_tensor(labels, dtype=torch.int64, device=device)
        attack_result = generate_attack_batch(
            policy=policy,
            images=batch,
            labels=label_tensor,
            rel_paths=rel_paths,
            image_size=batch.shape[-1],
            target_model=source_model,
            surrogates=surrogate_map,
            device=device,
        )
        with torch.no_grad():
            embeddings = target_model.forward_embeddings(attack_result.images).detach().cpu()
        for path, embedding in zip(batch_paths, embeddings):
            attacked[path] = embedding
    return attacked


def _measure_inference(
    model,
    config,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | None]:
    class_to_idx = build_class_to_idx(Path(config.train_dir))
    _, loader = build_eval_loader(
        data_dir=Path(config.val_dir),
        class_to_idx=class_to_idx,
        image_size=config.image_size,
        batch_size=batch_size,
        num_workers=0,
    )
    batch = next(iter(loader), None)
    if batch is None:
        return {"inference_ms_per_image": None, "peak_memory_mb": None}
    images, _, _, _ = batch
    images = images.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model.predict_logits(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
    elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        peak_memory = float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
    else:
        peak_memory = None
    elapsed = time.perf_counter() - start
    return {
        "inference_ms_per_image": 1000.0 * elapsed / float(max(1, images.shape[0])),
        "peak_memory_mb": peak_memory,
    }


def _attack_budget_summary(policies: list[AttackPolicy]) -> list[dict[str, object]]:
    return [
        {
            "name": policy.name,
            "family": policy.family,
            "kind": policy.kind,
            "eps": policy.eps,
            "alpha": policy.alpha,
            "steps": policy.steps,
            "restarts": policy.restarts,
            "random_start": policy.random_start,
            "weight": policy.weight,
        }
        for policy in policies
        if policy.enabled and policy.weight > 0
    ]


def build_report(run_dirs: list[Path], output_json: Path, output_md: Path, device_name: str, batch_size: int) -> None:
    device = torch.device(device_name)
    run_infos = []
    loaded_models: dict[str, tuple[object, object]] = {}

    for run_dir in run_dirs:
        config, model = _load_model_for_run(run_dir, device)
        loaded_models[str(run_dir)] = (config, model)
        history = _load_history(run_dir)
        final_epoch = history[-1]
        label_by_source, rel_path_by_source = _load_manifest_label_maps(Path(config.train_dir), Path(config.val_dir))
        pair_bundle = _pair_bundle(Path(config.val_pairs_path))
        unique_paths = sorted(set(pair_bundle["img1_paths"]) | set(pair_bundle["img2_paths"]))
        clean_embeddings = _embed_paths(model, unique_paths, device, batch_size=batch_size)
        clean_scores = _compute_pair_scores(clean_embeddings, clean_embeddings, pair_bundle["img1_paths"], pair_bundle["img2_paths"])
        clean_verification = verification_metrics_from_scores(clean_scores, pair_bundle["labels"])
        operating_threshold = _choose_operating_threshold(clean_verification)
        clean_operating = _pair_operating_metrics(clean_scores, pair_bundle["labels"], operating_threshold)
        _, val_loader = build_eval_loader(
            data_dir=Path(config.val_dir),
            class_to_idx=build_class_to_idx(Path(config.train_dir)),
            image_size=config.image_size,
            batch_size=batch_size,
            num_workers=0,
        )
        clean_classification = evaluate_clean(model, val_loader, device)
        robust_classification = evaluate_robust_all(
            model=model,
            loader=val_loader,
            device=device,
            policies=config.primary_attackers,
            surrogates={"target": TargetSelfSurrogate(model)},
            image_size=config.image_size,
        )
        whitebox_verification: dict[str, dict[str, object]] = {}
        for policy in config.enabled_primary_attackers():
            attacked_embeddings = _attack_unique_paths(
                source_model=model,
                target_model=model,
                device=device,
                batch_size=batch_size,
                unique_paths=unique_paths,
                label_by_source=label_by_source,
                rel_path_by_source=rel_path_by_source,
                policy=policy,
            )
            adv_scores = _compute_pair_scores(attacked_embeddings, clean_embeddings, pair_bundle["img1_paths"], pair_bundle["img2_paths"])
            adv_metrics = verification_metrics_from_scores(adv_scores, pair_bundle["labels"])
            operating = _pair_operating_metrics(adv_scores, pair_bundle["labels"], operating_threshold)
            adv_metrics.update(
                {
                    "clean_tar": clean_operating["tar"],
                    "robust_tar": operating["tar"],
                    "clean_far": clean_operating["far"],
                    "robust_far": operating["far"],
                    "tar_drop": clean_operating["tar"] - operating["tar"],
                    "impersonation_success_rate": operating["impersonation_success_rate"],
                    "dodging_success_rate": operating["dodging_success_rate"],
                    "attack_success_rate": operating["attack_success_rate"],
                }
            )
            whitebox_verification[policy.name] = adv_metrics

        run_infos.append(
            {
                "run_dir": str(run_dir),
                "target_model": config.target_model,
                "target_backbone": config.target_backbone,
                "dataset_name": config.dataset_name,
                "device": config.device,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "training_time_seconds": _parse_training_seconds(run_dir / "train.log"),
                "attack_budgets": _attack_budget_summary(config.enabled_primary_attackers()),
                "clean_classification": clean_classification,
                "robust_classification": robust_classification,
                "clean_verification": {**clean_verification, **{f"clean_{k}": v for k, v in clean_operating.items() if k != "attack_success_rate"}},
                "whitebox_verification_by_attack": whitebox_verification,
                "efficiency": _measure_inference(model, config, device, batch_size),
                "history_final_epoch": final_epoch,
            }
        )

    transfer_matrix: list[dict[str, object]] = []
    for target_info in run_infos:
        target_run = Path(target_info["run_dir"])
        target_config, target_model = loaded_models[str(target_run)]
        label_by_source, rel_path_by_source = _load_manifest_label_maps(Path(target_config.train_dir), Path(target_config.val_dir))
        pair_bundle = _pair_bundle(Path(target_config.val_pairs_path))
        unique_paths = sorted(set(pair_bundle["img1_paths"]) | set(pair_bundle["img2_paths"]))
        clean_embeddings = _embed_paths(target_model, unique_paths, device, batch_size=batch_size)
        clean_scores = _compute_pair_scores(clean_embeddings, clean_embeddings, pair_bundle["img1_paths"], pair_bundle["img2_paths"])
        clean_metrics = verification_metrics_from_scores(clean_scores, pair_bundle["labels"])
        operating_threshold = _choose_operating_threshold(clean_metrics)
        clean_operating = _pair_operating_metrics(clean_scores, pair_bundle["labels"], operating_threshold)

        for source_info in run_infos:
            source_run = Path(source_info["run_dir"])
            if source_run == target_run:
                continue
            source_config, source_model = loaded_models[str(source_run)]
            for policy in source_config.enabled_primary_attackers():
                attacked_embeddings = _attack_unique_paths(
                    source_model=source_model,
                    target_model=target_model,
                    device=device,
                    batch_size=batch_size,
                    unique_paths=unique_paths,
                    label_by_source=label_by_source,
                    rel_path_by_source=rel_path_by_source,
                    policy=policy,
                )
                adv_scores = _compute_pair_scores(attacked_embeddings, clean_embeddings, pair_bundle["img1_paths"], pair_bundle["img2_paths"])
                operating = _pair_operating_metrics(adv_scores, pair_bundle["labels"], operating_threshold)
                transfer_matrix.append(
                    {
                        "target_model": target_config.target_model,
                        "source_model": source_config.target_model,
                        "attack_name": policy.name,
                        "attack_family": policy.family,
                        "clean_tar": clean_operating["tar"],
                        "transfer_tar": operating["tar"],
                        "transfer_tar_drop": clean_operating["tar"] - operating["tar"],
                        "transfer_far": operating["far"],
                        "transfer_impersonation_success_rate": operating["impersonation_success_rate"],
                        "transfer_dodging_success_rate": operating["dodging_success_rate"],
                        "transfer_attack_success_rate": operating["attack_success_rate"],
                    }
                )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": device_name,
        "runs": run_infos,
        "transfer_matrix": transfer_matrix,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_to_markdown(report), encoding="utf-8")


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Comparison Ladder",
        "",
        f"Generated: {report['generated_at']}",
        f"Device: `{report['device']}`",
        "",
        "## Runs",
        "",
        "| Model | Clean AUC | EER | TAR@1e-2 | Best Acc | Clean Cls Acc | Robust Cls Acc | Robust ASR | Train Time (s) | Inf ms/img | Peak Mem MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        clean_ver = run["clean_verification"]
        robust_cls = run["robust_classification"]["average"]
        eff = run["efficiency"]
        lines.append(
            "| "
            f"{run['target_model']} | "
            f"{clean_ver['roc_auc']:.4f} | "
            f"{clean_ver['eer']:.4f} | "
            f"{clean_ver['tar@far=0.01']:.4f} | "
            f"{clean_ver['best_accuracy']:.4f} | "
            f"{run['clean_classification']['accuracy']:.4f} | "
            f"{robust_cls['accuracy']:.4f} | "
            f"{robust_cls['attack_success_rate']:.4f} | "
            f"{(run['training_time_seconds'] or 0.0):.1f} | "
            f"{(eff['inference_ms_per_image'] or 0.0):.2f} | "
            f"{(eff['peak_memory_mb'] or 0.0):.1f} |"
        )
    lines.extend(["", "## White-Box Verification", ""])
    for run in report["runs"]:
        lines.append(f"### {run['target_model']}")
        lines.append("")
        lines.append("| Attack | Robust TAR | TAR Drop | Impersonation SR | Dodging SR | Overall ASR | Robust AUC |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for attack_name, metrics in run["whitebox_verification_by_attack"].items():
            lines.append(
                "| "
                f"{attack_name} | "
                f"{metrics['robust_tar']:.4f} | "
                f"{metrics['tar_drop']:.4f} | "
                f"{metrics['impersonation_success_rate']:.4f} | "
                f"{metrics['dodging_success_rate']:.4f} | "
                f"{metrics['attack_success_rate']:.4f} | "
                f"{metrics['roc_auc']:.4f} |"
            )
        lines.append("")
    lines.extend(["## Transfer Matrix", "", "| Target | Source | Attack | TAR Drop | Impersonation SR | Dodging SR | Overall ASR |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for item in report["transfer_matrix"]:
        lines.append(
            "| "
            f"{item['target_model']} | "
            f"{item['source_model']} | "
            f"{item['attack_name']} | "
            f"{item['transfer_tar_drop']:.4f} | "
            f"{item['transfer_impersonation_success_rate']:.4f} | "
            f"{item['transfer_dodging_success_rate']:.4f} | "
            f"{item['transfer_attack_success_rate']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dirs = [Path(item).resolve() for item in args.run_dirs]
    build_report(run_dirs, args.output_json.resolve(), args.output_md.resolve(), args.device, args.batch_size)
    print(f"[INFO] Wrote JSON report: {args.output_json.resolve()}")
    print(f"[INFO] Wrote Markdown report: {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
