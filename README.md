# FYP

Face recognition robustness research focused on adversarial-plus-recognizer ensemble training for robust face embeddings.

## Entry Point

- `python3 main_train_ensemble.py --train-dir <train_dir> --val-dir <val_dir> --output-dir <run_dir>`

This repo is now training-first. The old recognition-only and attack-materialization top-level pipelines were removed during the pivot cleanup.

## Current Scope

- recognizer side of the ensemble: `ArcFace`, `CosFace`, `CurricularFace`
- current trainable target paths in code: `ArcFace`, `CosFace`, `CurricularFace` placeholders over `ResNet18`
- primary attacker set for the research plan: `PGD`, `BPFA`, `DFANet`
- surrogate attacker set for the research plan: `AdvFaceGAN`, `Adv-Makeup`, `Greedy-DiM`
- currently wired native primary attacks in code: `pgd`, `bpfa`, `dfanet`
- sampling strategies: `weighted_random`, `round_robin`, `family_round_robin`
- validation modes: clean-only, single-attack robust eval, all-enabled-attacks robust eval

## Layout

- `Dataset/`: clean data and local split outputs
- `Training/`: ensemble training stack
- `TrainingRuns/`: checkpoints and run summaries
- `Modifiers/attack/`: cached heavy-attack wrappers and generators
- `Backends/`: local envs, upstream sources, and checkpoint assets
- `Utility/`: small shared runtime helpers
- `notes/`: active project notes
- `scripts/prepare_training_split.py`: build local train/val splits from an identity-root dataset when needed, but active training should use `WebFace4M` manifests

## Minimal Setup

```bash
pip install torch torchvision
pip install facenet-pytorch pytorch-lightning "setuptools<81"
```

Retained backend inventory in the refactored layout:

- `Backends/sources/recognition/CosFace_upstream/`
- `Backends/sources/recognition/CurricularFace_upstream/`
- `Backends/sources/attack/surrogate/Adv-Makeup_upstream/`
- `Backends/sources/attack/surrogate/AdvFaceGAN_upstream/`
- `Backends/sources/attack/surrogate/Greedy-DiM_upstream/`

Primary attack native slots:

- `Backends/sources/attack/primary/BPFA_upstream/`
- `Backends/sources/attack/primary/DFANet_upstream/`

Surrogate wrappers:

- `Modifiers/attack/advfacegan/`
- `Modifiers/attack/adv_makeup/`
- `Modifiers/attack/dim/`

For research framing, do not confuse the recognizer side of the ensemble with the attacker side. The recognizer ensemble is `ArcFace`, `CosFace`, and `CurricularFace`. The attacker ensemble is organized separately into a primary attacker set and a surrogate attacker set.

## Quick Start

Run the verified `WebFace4M` subset clean config:

```bash
python3 main_train_ensemble.py \
  --config Training/arcface_webface4m_subset_lowepoch_mps.json
```

Run the verified ensemble smoke config:

```bash
python3 main_train_ensemble.py \
  --config Training/smoke_ensemble_config.json
```

Render the final paper ladder configs while varying only backbone and dataset fraction:

```bash
python3 scripts/render_paper_ladder_configs.py \
  --backbone resnet18 \
  --dataset-fraction 0.01
```

This writes canonical paper-reporting configs based on the full `WebFace4M` manifests, not the old subset presets.

## Shell Server Workflow

For a plain remote shell server, the repo now includes a minimal cloud-oriented path:

1. Bootstrap the dataset on the server instead of copying laptop-local manifests:

```bash
FYP_WEBFACE4M_ROOT=/shared/fyp/data/WebFace4M \
sh scripts/bootstrap_cloud_webface4m.sh
```

This downloads shards into `raw/` and regenerates manifest files with server-local absolute shard paths.

2. Render a server-local config and launch a run:

```bash
FYP_WEBFACE4M_ROOT=/shared/fyp/data/WebFace4M \
FYP_OUTPUT_ROOT=/shared/fyp/runs \
FYP_EXPORT_ROOT=/shared/fyp/exports \
sh scripts/launch_cloud_train.sh \
  Training/arcface_webface4m_resnet18_cloud_template.json \
  arcface-r18-server-smoke
```

If you want multi-GPU launch through `torchrun`, set:

```bash
USE_TORCHRUN=1 NPROC_PER_NODE=4 sh scripts/launch_cloud_train.sh \
  Training/arcface_webface4m_resnet18_cloud_template.json \
  arcface-r18-server-ddp
```

3. Pull back metrics and logs from the exported bundle under `FYP_EXPORT_ROOT`.

Each run export contains:

- `latest_metrics.json`: compact latest-epoch metrics for dashboards or quick checks
- `metrics.jsonl`: one JSON record per epoch
- `history.json`: full structured training history
- `summary.json`: final run summary
- `train.log`: human-readable logs
- `config.snapshot.json`: exact resolved config used for the run
- `repro_state.json`: runtime and package provenance

You can re-export an existing run manually:

```bash
python3 scripts/export_run_artifacts.py \
  --run-dir /shared/fyp/runs/arcface-r18-server-smoke \
  --export-root /shared/fyp/exports \
  --bundle
```

The generated `.tar.gz` bundle is the recommended artifact to `scp` or `rsync` back off the server.

## Verified Runs

- clean `WebFace4M` subset attempt:
  [TrainingRuns/webface4m_arcface_resnet18_subset_lowepoch_mps](/Users/jeromeharianto/Documents/School2/FYP/TrainingRuns/webface4m_arcface_resnet18_subset_lowepoch_mps)
- attacked ensemble smoke run:
  [TrainingRuns/webface4m_arcface_resnet18_ensemble_smoke](/Users/jeromeharianto/Documents/School2/FYP/TrainingRuns/webface4m_arcface_resnet18_ensemble_smoke)
