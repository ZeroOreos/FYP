# Greedy-DiM Wrapper

This entrypoint is the FYP-facing wrapper for the `Greedy-DiM` morphing slot.

Current status:

- The canonical pipeline entrypoint is `generate.py`.
- The implementation logic lives in `wrapper.py`.
- Imported native source location is `Backends/sources/attack/surrogate/Greedy-DiM_upstream/`.
- Expected backend assets location is `Backends/assets/attack/dim/`.

Important fidelity note:

This path is currently a local approximation inspired by the DiM morphing slot in the FYP shortlist. It is useful for smoke tests and pipeline integration, but it should not be described as a faithful reproduction of an official DiM release unless the encrypted native runner is unlocked through the upstream CITeR release process and then verified locally with the required DiffAE and ArcFace assets.
