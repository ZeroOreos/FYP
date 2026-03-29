# FaceShifter Wrapper

This entrypoint is the FYP-facing wrapper for FaceShifter.

- Canonical pipeline entrypoint: `generate.py`
- Wrapper implementation: `wrapper.py`
- Expected backend source location: `Backends/sources/FaceShifter_upstream/`
- Expected backend assets location: `Backends/assets/faceshifter/`

Fidelity note:

This path is a best-effort wrapper around the original repository layout and pretrained assets. Results should be described as a best attempt at the original method unless the exact upstream commit, dependencies, and checkpoints are all pinned and verified.

Status note:

This backend is currently blocked by missing trustworthy pretrained AEI weights. For the modern face-swap slot in the FYP, prefer `REFace` as the next implementation target unless usable FaceShifter checkpoints are found.
