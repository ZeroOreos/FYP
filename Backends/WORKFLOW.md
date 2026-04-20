# Backends Workflow

This folder stores backend-related source snapshots, dependencies, assets, and workdirs used by the FYP attack and recognition pipelines.

The goal is to keep these externals reproducible without turning the main repo into a Git-management headache.

## Layout

- `Backends/sources/recognition/<Name>_upstream/`: preserved third-party recognition backends that belong to the active recognizer ensemble
- `Backends/sources/attack/primary/<Name>_upstream/`: preserved third-party primary-attacker backends
- `Backends/sources/attack/surrogate/<Name>_upstream/`: preserved third-party surrogate-attacker backends
- `Backends/sources/dependencies/`: support repositories that are dependencies of a backend but are not attack methods themselves
- `Backends/assets/attack/<method>/`: local attack-model weights and support assets
- `Backends/workdirs/attack/<method>/`: temporary attack-wrapper workdirs
- `Backends/assets/recognition/<method>/`: local recognition-model weights and support assets
- `Backends/workdirs/recognition/<method>/`: local recognition runtime envs and scratch state

Current support dependencies:

- `Backends/sources/dependencies/diffae/` remains available for morphing-related external methods when needed
- `Backends/sources/dependencies/taming-transformers/` remains available for external image-generation backends when needed

Legacy compatibility:

- legacy attack wrappers may still accept older sibling paths such as `Backends/sources/taming-transformers/` if present
- new work should prefer `Backends/sources/dependencies/taming-transformers/`

## Rules

- Keep wrapper code out of `Backends/`; attack wrappers belong in `Modifiers/attack/`
- Native training-time attacks such as `pgd`, `bpfa`, and `dfanet` still belong in `Training/attacks.py` even if their upstream slots are reserved under `Backends/sources/attack/primary/`
- Keep runtime outputs, caches, and downloaded weights out of upstream trees
- Do not leave nested `.git` directories inside vendored externals
- Record every backend and dependency in `Backends/manifest.json`
- Record local behavior changes and fidelity notes in `notes/working/20_backend_adaptations.txt`
- Remove stale backends when they no longer belong to the active ensemble design instead of letting them linger as implied options

## Bootstrap Philosophy

Bootstrap should be boring.

Bootstrap in this repo should:

- verify the expected local layout exists
- create missing local working directories when that is safe
- detect nested Git repos, missing required files, and likely dirty-runtime locations
- tell us what is missing or inconsistent

Bootstrap should not:

- re-clone upstream repositories on every run
- re-download checkpoints automatically during normal validation
- rewrite vendor trees unless we explicitly ask it to
- hide provenance by making large implicit changes

That keeps bootstrap useful without making it redundant or destructive.

## Patch Strategy

When we patch vendored upstream code, the intended long-term pattern is:

- preserve the upstream snapshot under `Backends/sources/...`
- keep a concise rationale in `notes/working/20_backend_adaptations.txt`
- when a patch becomes stable or substantial, capture it under `Backends/patches/`

The repo is not fully migrated to patch files yet, but that is the preferred direction.

## Git Hygiene

Preferred model:

- vendored snapshots are plain directories tracked by the main repo
- no embedded Git repos
- no ad hoc runtime files committed inside upstream trees

Noise that should be cleaned instead of preserved:

- temporary workdirs under `Backends/workdirs/attack/<method>/` except `.gitkeep`
- smoke-test artifacts under `tmp/`
- upstream demo outputs copied into vendored source trees
- bulky example media copied into vendored backend trees
- nested Git metadata inside vendored directories

Why not heavy bootstrap automation?

- these backends often need manual checkpoint placement, offline caches, or targeted compatibility patches
- a script that silently clones, installs, patches, and downloads everything would be harder to trust than maintain
- for this project, auditability matters more than convenience

## Commands

Audit the current layout:

```bash
python3 scripts/bootstrap_backends.py audit
```

Create expected local support directories and then audit:

```bash
python3 scripts/bootstrap_backends.py prepare
```

Fail the command if warnings are found:

```bash
python3 scripts/bootstrap_backends.py audit --strict
```
