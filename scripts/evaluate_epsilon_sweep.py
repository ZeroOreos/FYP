#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from Training.attacks import generate_attack_batch
from Training.config import AttackPolicy, load_config
from Training.dataset import build_class_to_idx
from Training.evaluate import _load_image_from_reference, verification_metrics_from_scores
from Training.recognizers import TargetSelfSurrogate, build_target_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight adversarial epsilon sweep for completed runs.")
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        required=True,
        help="Completed run directory to evaluate. Pass multiple times.",
    )
    parser.add_argument(
        "--attack",
        dest="attack_names",
        action="append",
        default=None,
        help="Attack name to sweep. Defaults to all enabled primary attackers from the run config.",
    )
    parser.add_argument(
        "--eps",
        dest="eps_values",
        action="append",
        type=float,
        default=None,
        help="Attack epsilon to evaluate in input space [0,1]. Pass multiple times. Defaults to 4/255, 8/255, 12/255, 16/255, 24/255.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )
    return parser.parse_args()


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pth"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir}")
    return checkpoints[-1]


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
    return {
        "img1_paths": [str(item) for item in pair_data["img1_paths"].tolist()],
        "img2_paths": [str(item) for item in pair_data["img2_paths"].tolist()],
        "labels": torch.as_tensor(pair_data["labels"].astype("int64")),
    }


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
                image_path = item.get("image_path")
                if image_path is not None and Path(str(image_path)).expanduser().resolve().is_file():
                    source = str(Path(str(image_path)).expanduser().resolve())
                else:
                    source = f"{Path(item['shard_path']).resolve()}::{item['key']}.jpg"
                label_by_source[source] = int(item["label_idx"])
                rel_path_by_source[source] = str(item["rel_path"])
    return label_by_source, rel_path_by_source


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
            tensors.append(torch.from_numpy(array).permute(2, 0, 1))
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


def _policy_with_eps(policy: AttackPolicy, eps: float) -> AttackPolicy:
    base_eps = max(1e-12, float(policy.eps))
    scaled_alpha = float(policy.alpha) * (float(eps) / base_eps)
    return replace(policy, eps=float(eps), alpha=scaled_alpha)


def main() -> None:
    args = parse_args()
    eps_values = args.eps_values or [4.0 / 255.0, 8.0 / 255.0, 12.0 / 255.0, 16.0 / 255.0, 24.0 / 255.0]
    device = torch.device(args.device)
    report: dict[str, object] = {
        "device": args.device,
        "batch_size": args.batch_size,
        "eps_values": eps_values,
        "runs": [],
    }

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).resolve()
        config, model = _load_model_for_run(run_dir, device)
        enabled_attackers = {policy.name: policy for policy in config.enabled_primary_attackers()}
        attack_names = args.attack_names or list(enabled_attackers)
        selected_policies = [enabled_attackers[name] for name in attack_names if name in enabled_attackers]
        if not selected_policies:
            raise RuntimeError(f"No requested enabled primary attackers found for {run_dir}")

        pair_bundle = _pair_bundle(Path(config.val_pairs_path).resolve())
        unique_paths = sorted(set(pair_bundle["img1_paths"]) | set(pair_bundle["img2_paths"]))
        label_by_source, rel_path_by_source = _load_manifest_label_maps(Path(config.train_dir), Path(config.val_dir))
        clean_embeddings = _embed_paths(model, unique_paths, device, batch_size=args.batch_size)
        clean_scores = _compute_pair_scores(clean_embeddings, clean_embeddings, pair_bundle["img1_paths"], pair_bundle["img2_paths"])
        clean_metrics = verification_metrics_from_scores(clean_scores, pair_bundle["labels"])
        operating_threshold = _choose_operating_threshold(clean_metrics)
        clean_operating = _pair_operating_metrics(clean_scores, pair_bundle["labels"], operating_threshold)

        run_result: dict[str, object] = {
            "run_dir": str(run_dir),
            "target_model": config.target_model,
            "target_backbone": config.target_backbone,
            "clean_verification": clean_metrics,
            "clean_operating": clean_operating,
            "sweeps": {},
        }

        for policy in selected_policies:
            sweep_rows = []
            for eps in eps_values:
                sweep_policy = _policy_with_eps(policy, eps)
                attacked_embeddings = _attack_unique_paths(
                    source_model=model,
                    target_model=model,
                    device=device,
                    batch_size=args.batch_size,
                    unique_paths=unique_paths,
                    label_by_source=label_by_source,
                    rel_path_by_source=rel_path_by_source,
                    policy=sweep_policy,
                )
                adv_scores = _compute_pair_scores(
                    attacked_embeddings,
                    clean_embeddings,
                    pair_bundle["img1_paths"],
                    pair_bundle["img2_paths"],
                )
                adv_metrics = verification_metrics_from_scores(adv_scores, pair_bundle["labels"])
                operating = _pair_operating_metrics(adv_scores, pair_bundle["labels"], operating_threshold)
                sweep_rows.append(
                    {
                        "eps": float(eps),
                        "alpha": float(sweep_policy.alpha),
                        "steps": int(sweep_policy.steps),
                        "restarts": int(sweep_policy.restarts),
                        "verification": adv_metrics,
                        "operating": operating,
                        "delta": {
                            "tar_drop_at_far_1e_4": float(clean_metrics.get("tar@far=0.0001", 0.0)) - float(adv_metrics.get("tar@far=0.0001", 0.0)),
                            "tar_drop_at_far_1e_5": float(clean_metrics.get("tar@far=1e-05", 0.0)) - float(adv_metrics.get("tar@far=1e-05", 0.0)),
                            "auc_drop": float(clean_metrics.get("roc_auc", 0.0)) - float(adv_metrics.get("roc_auc", 0.0)),
                        },
                    }
                )
            run_result["sweeps"][policy.name] = sweep_rows

        report["runs"].append(run_result)

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
