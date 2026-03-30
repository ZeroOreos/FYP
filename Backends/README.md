# Backends

This folder is reserved for backend-related third-party code and local backend support state used by the FYP attack and protection wrappers.

These backends are intended to be as faithful to the original upstream repositories as practical, but the FYP integration should still be described as a best-effort recreation rather than a guaranteed byte-identical reproduction.

Expected layout:

- `sources/attack/AdvFaceGAN_upstream/`
- `sources/attack/DiM_upstream/`
- `sources/attack/DiM_native_upstream/`
- `sources/attack/FaceShifter_upstream/`
- `sources/attack/FOMM_upstream/`
- `sources/attack/MIPGAN_upstream/`
- `sources/attack/MorDIFF_upstream/`
- `sources/attack/REFace_upstream/`
- `sources/attack/SimSwap_upstream/`
- `sources/protection/<Name>_upstream/`
- `sources/dependencies/taming-transformers/`
- `sources/dependencies/diffae/`
- `assets/attack/<method>/`
- `assets/protection/<method>/`
- `workdirs/attack/<method>/`
- `workdirs/protection/<method>/`

Keep local wrapper code out of this folder. Backend source snapshots belong in `sources/`, mutable backend assets belong in `assets/`, and temporary scratch state belongs in `workdirs/`.

Runtime policy:

- Prefer one canonical runtime env per model under `workdirs/<method>/`
- Keep model envs isolated instead of forcing one shared repo-wide PyTorch stack
- Add an accelerator-oriented second env for a model only when needed and document it in the working notes
- Track accelerator support per model after verification rather than assuming every model can share the same MPS-capable stack
- For protection models, prefer one dedicated env per model under `workdirs/protection/<method>/`
- For protection-model device selection, prefer `cuda -> mps -> cpu` unless model-specific verification notes justify a different order

For the canonical inventory and local rules, see:

- `Backends/manifest.json`
- `Backends/WORKFLOW.md`

Current planning note:

- Attack and protection backends should use the same storage pattern so the FYP wrappers under `Models/attack/` and `Models/protection/` stay structurally parallel.
- `sources/attack/REFace_upstream/` is the preferred modern replacement backend for `faceshifter` because official checkpoints appear to be available.
- `sources/attack/FaceShifter_upstream/` is kept for historical/reference purposes, but the current FYP plan should treat it as blocked until trustworthy pretrained weights are found.
