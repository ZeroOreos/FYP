Third-party license notes

This FYP is for educational / research use.

Quick check:

- FOMM
  MIT
  Good vendor candidate

- SimSwap
  Non-commercial / research-use oriented
  Fine for FYP research use
  Not a clean default vendor dependency

- FaceShifter
  License depends on the exact codebase used
  Check before vendoring

- REFace
  MIT
  Good modern face-swap candidate if checkpoints stay available

- AdvFaceGAN
  Check before vendoring

- MIPGAN
  Code access looks controlled
  Not a simple vendor target

- MorDIFF
  Often research-only / non-commercial
  Check before vendoring

- InsightFace / INSwapper
  Code MIT, models restricted
  Good local research backend, but model terms matter

- FaceSwap
  GPL-3.0
  Open source, but strong copyleft

Rule of thumb:

- Prefer FOMM for clean native vendoring
- Treat SimSwap as research-only unless licensing is clarified further
- Prefer REFace over FaceShifter when a modern reproducible swap baseline is needed
- Do not vendor weights until their redistribution terms are confirmed
