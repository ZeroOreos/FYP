# Backends

This folder is reserved for backend-related third-party code and local backend support state used by the FYP ensemble-training pipeline.

These backends are intended to be as faithful to the original upstream repositories as practical, but the FYP integration should still be described as a best-effort recreation rather than a guaranteed byte-identical reproduction.

Expected layout:

- `sources/recognition/CosFace_upstream/`
- `sources/recognition/CurricularFace_upstream/`
- `sources/attack/primary/BPFA_upstream/`
- `sources/attack/primary/DFANet_upstream/`
- `sources/attack/surrogate/AdvFaceGAN_upstream/`
- `sources/attack/surrogate/Adv-Makeup_upstream/`
- `sources/attack/surrogate/Greedy-DiM_upstream/`
- `sources/dependencies/taming-transformers/`
- `sources/dependencies/diffae/`
- `assets/attack/<method>/`
- `assets/recognition/<method>/`
- `workdirs/attack/<method>/`
- `workdirs/recognition/<method>/`

Keep local wrapper code out of this folder. Backend source snapshots belong in `sources/`, mutable backend assets belong in `assets/`, and temporary scratch state belongs in `workdirs/`.

Runtime policy:

- Prefer one canonical runtime env per model under `workdirs/<method>/`
- Keep model envs isolated instead of forcing one shared repo-wide PyTorch stack
- Add an accelerator-oriented second env for a model only when needed and document it in the working notes
- Track accelerator support per model after verification rather than assuming every model can share the same MPS-capable stack

For the canonical inventory and local rules, see:

- `Backends/manifest.json`
- `Backends/WORKFLOW.md`

Current planning note:

- Keep the backend inventory aligned with the current ensemble definition in `notes/todo.txt`
- The recognizer side is intentionally restricted to `ArcFace`, `CosFace`, and `CurricularFace`
- The primary attacker side is implemented natively in `Training/attacks.py` while preserving upstream slots under `sources/attack/primary/`
- The surrogate attacker side is organized into vendored upstream sources plus repo-local wrappers under `Modifiers/attack/`
