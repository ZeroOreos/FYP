# Backends Workflow

This folder stores backend-related source snapshots, dependencies, assets, and workdirs used by the FYP wrappers under `Modifiers/attack/`.

The goal is to keep these externals reproducible without turning the main repo into a Git-management headache.

## Layout

- `Backends/sources/<Name>_upstream/`: preserved third-party attack backends
- `Backends/sources/dependencies/`: support repositories that are dependencies of a backend but are not attack methods themselves
- `Backends/assets/<method>/`: local model weights and support assets
- `Backends/workdirs/<method>/`: temporary wrapper workdirs

Current support dependency:

- `Backends/sources/dependencies/taming-transformers/` for `REFace_upstream`

Legacy compatibility:

- `Modifiers/attack/reface/wrapper.py` still accepts the older sibling path `Backends/sources/taming-transformers/` if present
- new work should prefer `Backends/sources/dependencies/taming-transformers/`

## Rules

- Keep wrapper code out of `Backends/`; wrappers belong in `Modifiers/attack/`
- Keep runtime outputs, caches, and downloaded weights out of upstream trees
- Do not leave nested `.git` directories inside vendored externals
- Record every backend and dependency in `Backends/manifest.json`
- Record local behavior changes and fidelity notes in `notes/working/20_backend_adaptations.txt`

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

- temporary attack workdirs under `Backends/workdirs/<method>/` except `.gitkeep`
- smoke-test artifacts under `tmp/`
- upstream demo outputs copied into ignored paths such as `Backends/sources/SimSwap_upstream/output/`
- bulky example media in ignored runtime-only paths such as:
  `Backends/sources/SimSwap_upstream/crop_224/`
  `Backends/sources/SimSwap_upstream/demo_file/`
  `Backends/sources/SimSwap_upstream/docs/img/`
  `Backends/sources/REFace_upstream/assets/`
  `Backends/sources/REFace_upstream/examples/`
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
