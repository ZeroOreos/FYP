# AdvFaceGAN Wrapper

This entrypoint is the FYP-facing wrapper for AdvFaceGAN.

Current status:

- The canonical pipeline entrypoint is `generate.py`.
- The implementation logic lives in `wrapper.py`.
- Expected backend source location is `Backends/sources/attack/AdvFaceGAN_upstream/`.
- Expected backend assets location is `Backends/assets/attack/advfacegan/`.

Important fidelity note:

This backend is currently a best-effort local recreation and checkpoint adapter inspired by the original AdvFaceGAN repository and paper. It is not guaranteed to be a byte-faithful reproduction of the upstream codebase unless the full upstream repository and compatible pretrained weights are installed and validated.
