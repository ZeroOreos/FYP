# External Backends

This folder is reserved for preserved third-party attack repositories used by the FYP wrappers in `Modifiers/attack/`.

These backends are intended to be as faithful to the original upstream repositories as practical, but the FYP integration should still be described as a best-effort recreation rather than a guaranteed byte-identical reproduction.

Expected layout:

- `AdvFaceGAN_upstream/`
- `FaceShifter_upstream/`
- `FOMM_upstream/`
- `MIPGAN_upstream/`
- `MorDIFF_upstream/`
- `REFace_upstream/`
- `SimSwap_upstream/`

Keep local wrapper code out of this folder. Only preserved upstream code, lightweight patch notes, and backend-specific environment setup should live here.

Current planning note:

- `REFace_upstream/` is the preferred modern replacement backend for `faceshifter` because official checkpoints appear to be available.
- `FaceShifter_upstream/` is kept for historical/reference purposes, but the current FYP plan should treat it as blocked until trustworthy pretrained weights are found.
