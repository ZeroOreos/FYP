#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

REMOTE_HOST=${REMOTE_HOST:-}
REMOTE_ROOT=${REMOTE_ROOT:-/home/wen/data/FYP}

if [ -z "$REMOTE_HOST" ]; then
  echo "Set REMOTE_HOST, e.g. REMOTE_HOST=wen@your-server" >&2
  exit 1
fi

rsync -az --delete \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "TrainingRuns/" \
  --exclude "Dataset/WebFace4M/raw/" \
  --exclude "Dataset/WebFace4M/images/" \
  --exclude "Dataset/WebFace42M/" \
  "$PROJECT_ROOT/" \
  "$REMOTE_HOST:$REMOTE_ROOT/"

echo "[INFO] synced $PROJECT_ROOT -> $REMOTE_HOST:$REMOTE_ROOT"
