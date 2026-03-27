# REFace Wrapper

This entrypoint is the FYP-facing wrapper for REFace.

- Canonical pipeline entrypoint: `generate.py`
- Wrapper implementation: `wrapper.py`
- Expected preserved upstream location: `external/REFace_upstream/`
- Expected checkpoints location: `checkpoints/reface/`

Fidelity note:

This path is a best-effort wrapper around the official REFace repository and its published assets. The local CPU execution path is a patched approximation of the original CUDA-first setup, so results should still be described as a best attempt rather than a guaranteed byte-faithful reproduction.
