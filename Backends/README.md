# Backends

This folder is reserved for backend-related third-party code and local backend support state used by the FYP wrappers in `Modifiers/attack/`.

These backends are intended to be as faithful to the original upstream repositories as practical, but the FYP integration should still be described as a best-effort recreation rather than a guaranteed byte-identical reproduction.

Expected layout:

- `sources/AdvFaceGAN_upstream/`
- `sources/FaceShifter_upstream/`
- `sources/FOMM_upstream/`
- `sources/MIPGAN_upstream/`
- `sources/MorDIFF_upstream/`
- `sources/REFace_upstream/`
- `sources/SimSwap_upstream/`
- `sources/dependencies/taming-transformers/`
- `assets/<method>/`
- `workdirs/<method>/`

Keep local wrapper code out of this folder. Backend source snapshots belong in `sources/`, mutable backend assets belong in `assets/`, and temporary scratch state belongs in `workdirs/`.

For the canonical inventory and local rules, see:

- `Backends/manifest.json`
- `Backends/WORKFLOW.md`

Current planning note:

- `sources/REFace_upstream/` is the preferred modern replacement backend for `faceshifter` because official checkpoints appear to be available.
- `sources/FaceShifter_upstream/` is kept for historical/reference purposes, but the current FYP plan should treat it as blocked until trustworthy pretrained weights are found.
