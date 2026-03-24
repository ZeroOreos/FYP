Third-party license notes

This FYP is for educational / research use.

Quick check:

- FOMM
  MIT
  Good vendor candidate

- LivePortrait
  Appears MIT
  Good vendor candidate, re-check upstream when integrating

- SimSwap
  Non-commercial / research-use oriented
  Fine for FYP research use
  Not a clean default vendor dependency

- FaceShifter
  License depends on the exact codebase used
  Check before vendoring

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

- Prefer FOMM and LivePortrait for clean native vendoring
- Treat SimSwap as research-only unless licensing is clarified further
- Do not vendor weights until their redistribution terms are confirmed
