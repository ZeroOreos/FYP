#!/usr/bin/env sh
set -eu

# Purge generated local artifacts so the workspace is ready to upload to a server.
# This intentionally removes ignored run/data/cache outputs, not tracked source files.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_ROOT"

rm -rf \
  TrainingRuns \
  TrainingExports \
  Training/generated \
  Results \
  runs \
  tmp \
  Dataset/WebFace4M/raw \
  Dataset/WebFace4M/manifests \
  Dataset/WebFace4M/subset_manifests \
  Dataset/WebFace4M/images \
  Dataset/WebFace4M/extract_manifest.json \
  Dataset/pairs/*.npz \
  Dataset/pairs/*.npy \
  Backends/assets \
  Backends/sources \
  Backends/workdirs \
  Training/__pycache__ \
  scripts/__pycache__ \
  Utility/__pycache__

find Modifiers -type d -name '__pycache__' -prune -exec rm -rf {} +

echo "[INFO] Local artifacts purged"
