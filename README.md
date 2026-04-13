# FYP

Face recognition. Attack evaluation. ERG4902 FYP.

## Main Entry Points

- `python3 main_recognition.py <dataset_dir>`
- `python3 main_attack.py <gallery_dir> --attack-method advfacegan --attack-generator-script <script>`

## Models

- `InsightFace`
- `FaceNet`
- `AdaFace`
- `MagFace`

Recognition roles:

- `InsightFace`: strong modern ArcFace-family baseline
- `FaceNet`: classic baseline
- `AdaFace`: quality-adaptive modern baseline
- `MagFace`: quality-aware modern baseline

## Backend Order

- PyTorch: `cuda -> mps -> cpu`
- InsightFace / ONNX: `CoreML -> CUDA -> CPU`

Recognition runtime layout:

- `Backends/workdirs/recognition/facenet/venv`
- `Backends/workdirs/recognition/insightface/venv`
- `Backends/workdirs/recognition/adaface/venv`
- `Backends/workdirs/recognition/magface/venv`
- `Backends/assets/recognition/facenet/`
- `Backends/assets/recognition/insightface/`
- `Backends/assets/recognition/adaface/`
- `Backends/assets/recognition/magface/`

Recognition external source layout:

- `Backends/sources/AdaFace_upstream/`
- `Backends/sources/MagFace_upstream/`

Protection runtime layout:

- `Backends/workdirs/protection/<method>/venv`
- `Backends/assets/protection/<method>/`
- `Models/protection/<method>/wrapper.py`

Warning: forced backend missing. Kaboom likely.

## Install

Apple / CoreML-first:

```bash
pip install torch torchvision
pip install onnxruntime insightface facenet-pytorch
```

NVIDIA / CUDA-first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu insightface facenet-pytorch
```

Warning: wrong CUDA version. Broken env.

Refs:

- [PyTorch install guide](https://pytorch.org/get-started/locally)
- [ONNX Runtime install guide](https://onnxruntime.ai/docs/install/)

## Force Backend

- `--torch-device auto|cuda|mps|cpu`
- `--onnx-provider auto|coreml|cuda|cpu`

Examples:

```bash
python3 main_recognition.py <dataset_dir> --torch-device auto --onnx-provider auto
python3 main_attack.py <gallery_dir> --attack-method advfacegan --attack-generator-script <script> --torch-device cuda --onnx-provider cuda
```

## Layout

- `Dataset/` data and variants
- `Results/` embeddings and metrics
- `Utility/` shared runtime and paths
- `Models/` model code
- `Modifiers/` preprocessing and attack generation
