# Adv-Makeup Wrapper

This wrapper provides the repo-facing entrypoint for the `Adv-Makeup` surrogate attacker.

Current support level:

- Native upstream snapshot is vendored under `Backends/sources/attack/surrogate/Adv-Makeup_upstream/`.
- The wrapper currently exposes a local approximation path that generates eye-region makeup perturbations with the same pair-input and cache-output contract used by the rest of the surrogate attack tooling.
- Full upstream training/inference reproduction remains gated on the original Adv-Makeup checkpoint stack and its dataset-specific runtime assumptions.

Expected backend locations:

- repo: `Backends/sources/attack/surrogate/Adv-Makeup_upstream/`
- assets: `Backends/assets/attack/adv_makeup/`
- workdir: `Backends/workdirs/attack/adv_makeup/`

Usage shape:

- input: dataset root, pair `.npz`, output cache root, summary json
- output: cached attacked images under `output_dir/<victim_identity>/<victim_basename>`
- training integration: attach the cache root to an attack policy with `name="adv_makeup"` and `kind="cached"`
