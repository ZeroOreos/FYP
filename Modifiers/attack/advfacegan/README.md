# AdvFaceGAN Wrapper

This entrypoint is the FYP-facing wrapper for AdvFaceGAN.

Current status:

- The canonical pipeline entrypoint is `generate.py`.
- The implementation logic lives in `wrapper.py`.
- Expected preserved upstream location is `external/AdvFaceGAN_upstream/`.
- Expected checkpoints location is `checkpoints/advfacegan/`.

Important fidelity note:

This backend is currently a best-effort local recreation and checkpoint adapter inspired by the original AdvFaceGAN repository and paper. It is not guaranteed to be a byte-faithful reproduction of the upstream codebase unless the full upstream repository and compatible pretrained weights are installed and validated.
