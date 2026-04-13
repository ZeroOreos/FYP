# REFace Wrapper

This entrypoint is the FYP-facing wrapper for REFace.

- Canonical pipeline entrypoint: `generate.py`
- Wrapper implementation: `wrapper.py`
- Expected backend source location: `Backends/sources/attack/REFace_upstream/`
- Preferred support dependency location: `Backends/sources/dependencies/taming-transformers/`
- Legacy support dependency fallback: `Backends/sources/taming-transformers/`
- Expected backend assets location: `Backends/assets/attack/reface/`

Fidelity note:

This path is a best-effort wrapper around the official REFace repository and its published assets. The local CPU execution path is a patched approximation of the original CUDA-first setup, so results should still be described as a best attempt rather than a guaranteed byte-faithful reproduction.
