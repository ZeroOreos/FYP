# FYP

Face recognition robustness research focused on multi-recognizer plus multi-attacker training for robust face embeddings.

## Entry Point

- `python3 main_train_ensemble.py --train-dir <train_dir> --val-dir <val_dir> --output-dir <run_dir>`

This repo is training-first. Legacy recognition-only and attack-materialization top-level paths were removed during the final cleanup.

## Current Scope

- recognizer ensemble families for the paper path: `ArcFace`, `CosFace`, `CurricularFace`
- current trainable target paths in code: single-target `ArcFace` / `CosFace` / `CurricularFace`, plus the legacy composite `joint_pool` target path
- primary attacker set for the research plan: `PGD`, `BPFA`, `DFANet`
- surrogate attacker set for the research plan: `AdvFaceGAN`, `Adv-Makeup`, `Greedy-DiM`
- currently wired native primary attacks in code: `pgd`, `bpfa`, `dfanet`
- sampling strategies: `weighted_random`, `round_robin`, `family_round_robin`
- validation modes: clean-only, single-attack robust eval, all-enabled-attacks robust eval

## Layout

- `Dataset/`: dataset metadata, exact class mapping, and verification-pair metadata
- `Training/`: training, attack generation, evaluation, and config code
- `TrainingRuns/`: final experiment summaries, histories, metrics, config snapshots, and posthoc JSONs
- `Modifiers/attack/`: optional heavy-attack wrappers and generators
- `Backends/`: upstream-source and asset layout for optional external attack/recognition backends
- `Utility/`: small shared runtime helpers
- `scripts/`: import, rendering, evaluation, probing, and artifact utilities

Notes, raw datasets, extracted images, local backups, and model checkpoints are intentionally not tracked.

## Minimal Setup

```bash
python3.8 -m pip install torch==2.2.2 torchvision==0.17.2
pip install facenet-pytorch pytorch-lightning "setuptools<81"
```

With `uv`, the project metadata now includes the common runtime packages:

```bash
uv sync
```

If you want to try `torch.compile` on Linux `x86_64`, install the optional compile extra:

```bash
uv sync --extra compile
```

If Triton is missing, the training code now falls back to eager execution instead of aborting the run.

Backend inventory retained by the layout:

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

The three moving parts are separate:

- target model: the single defended model being optimized
- recognizer ensemble: auxiliary recognizer heads or models that provide additional training pressure
- attacker ensemble: rotating attack policies split into primary attackers and surrogate attackers

`joint_pool` remains available as a composite target architecture, but it is not the same thing as the external recognizer-ensemble path used for the paper design.

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

Render the canonical paper ladder plus the short integration config:

```bash
python3 scripts/render_paper_ladder_configs.py \
  --dataset-fraction 1.0 \
  --batch-size 64 \
  --num-workers 16
```

This writes the five full paper runs plus the short `joint_test` config against the full `WebFace4M` manifests. The rendered ladder currently uses `128 / 48 / 32` phase batch sizes with `attack_chunk_size=32`.

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

2. Render a server-local config and launch the full run:

```bash
FYP_WEBFACE4M_ROOT=/shared/fyp/data/WebFace4M \
FYP_OUTPUT_ROOT=/shared/fyp/runs \
sh scripts/launch_cloud_train.sh \
  Training/arcface_webface4m_resnet18_cloud_template.json \
  arcface-r18-server-full
```

For throughput tuning on the same DDP/data path, use a benchmark preset instead of the full run.

The small `resnet18` preset is useful for loader/DDP smoke checks, but if it under-drives the GPUs use the `iresnet100` benchmark preset instead so the compute mix stays closer to the real run:

```bash
FYP_WEBFACE4M_ROOT=/shared/fyp/data/WebFace4M \
FYP_OUTPUT_ROOT=/shared/fyp/runs \
python3 scripts/render_cloud_config.py \
  --base-config Training/arcface_webface4m_iresnet100_benchmark_cuda.json \
  --output-config Training/generated/ir100-bench-bs12.json \
  --run-name ir100-bench-bs12 \
  --batch-size 12 \
  --gradient-accumulation-steps 1 \
  --num-workers 16 \
  --dataset-fraction 0.02 \
  --enable-distributed \
  --disable-sync-batchnorm \
  --disable-gradient-checkpointing \
  --disable-torch-compile

OMP_NUM_THREADS=1 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  main_train_ensemble.py \
  --config Training/generated/ir100-bench-bs12.json \
  2>&1 | tee /shared/fyp/runs/logs/ir100-bench-bs12.log
```

If you specifically want the lighter `resnet18` systems-only check:

```bash
FYP_WEBFACE4M_ROOT=/shared/fyp/data/WebFace4M \
FYP_OUTPUT_ROOT=/shared/fyp/runs \
python3 scripts/render_cloud_config.py \
  --base-config Training/arcface_webface4m_resnet18_benchmark_cuda.json \
  --output-config Training/generated/r18-bench-bs64.json \
  --run-name r18-bench-bs64 \
  --batch-size 64 \
  --gradient-accumulation-steps 1 \
  --num-workers 16 \
  --dataset-fraction 0.01 \
  --enable-distributed \
  --disable-sync-batchnorm \
  --disable-gradient-checkpointing \
  --disable-torch-compile

OMP_NUM_THREADS=1 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  main_train_ensemble.py \
  --config Training/generated/r18-bench-bs64.json \
  2>&1 | tee /shared/fyp/runs/logs/r18-bench-bs64.log
```

Summarize the last 50 batch windows from a captured log:

```bash
python3 scripts/summarize_train_log.py /shared/fyp/runs/logs/r18-bench-bs64.log
```

If you want multi-GPU launch through `torchrun`, set:

```bash
USE_TORCHRUN=1 NPROC_PER_NODE=4 sh scripts/launch_cloud_train.sh \
  Training/arcface_webface4m_resnet18_cloud_template.json \
  arcface-r18-server-ddp-full
```

The default cloud template uses the 24-epoch `ResNet18` schedule instead of the older 12-epoch smoke cadence.

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
  --run-dir /shared/fyp/runs/arcface-r18-server-full \
  --export-root /shared/fyp/exports \
  --bundle
```

## Verified Runs

The committed `TrainingRuns/` files are the final paper evidence bundle. They contain metrics and reproducibility metadata, not checkpoints. Regenerate checkpoints by rerunning the corresponding generated config under `Training/generated/paper_ladder/`.
