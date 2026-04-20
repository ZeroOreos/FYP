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
- `TrainingRuns/`: server-local checkpoints and run summaries
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

For a plain remote shell server, the repo now includes a minimal upload-first path:

1. Bootstrap the dataset on the server instead of copying laptop-local manifests:

```bash
FYP_WEBFACE4M_ROOT=/shared/fyp/data/WebFace4M \
sh scripts/bootstrap_cloud_webface4m.sh
```

This downloads shards into `raw/` and regenerates manifest files with server-local absolute shard paths.

If direct filesystem JPEG reads turn out to be faster on the server, keep the same manifest contract and opt into extraction:

```bash
EXTRACT_IMAGES=1 \
FYP_WEBFACE4M_ROOT=/shared/fyp/data/WebFace4M \
sh scripts/bootstrap_cloud_webface4m.sh
```

That extracts images under `images/`, rebuilds manifests with `image_path` entries, and the current loader will prefer extracted files while still falling back to shard reads when needed.

2. Render a server-local config and launch the serious run:

```bash
FYP_WEBFACE4M_ROOT=/shared/fyp/data/WebFace4M \
FYP_OUTPUT_ROOT=/shared/fyp/runs \
sh scripts/launch_cloud_train.sh \
  Training/arcface_webface4m_resnet18_cloud_template.json \
  arcface-r18-server-serious
```

If you want multi-GPU launch through `torchrun`, set:

```bash
USE_TORCHRUN=1 NPROC_PER_NODE=4 sh scripts/launch_cloud_train.sh \
  Training/arcface_webface4m_resnet18_cloud_template.json \
  arcface-r18-server-ddp-serious
```

The default cloud template now matches the repo's 24-epoch serious `ResNet18` schedule instead of the older 12-epoch smoke cadence.

Each run writes directly into the server run directory:

- `latest_metrics.json`: compact latest-epoch metrics for dashboards or quick checks
- `metrics.jsonl`: one JSON record per epoch
- `history.json`: full structured training history
- `summary.json`: final run summary
- `train.log`: human-readable logs
- `config.snapshot.json`: exact resolved config used for the run
- `repro_state.json`: runtime and package provenance

If you still need a compact export later, you can package an existing run manually:

```bash
python3 scripts/export_run_artifacts.py \
  --run-dir /shared/fyp/runs/arcface-r18-server-serious \
  --export-root /shared/fyp/exports \
  --bundle
```

## Verified Runs

Local run artifacts are intentionally ignored and should be purged before uploading this repo to a server.
