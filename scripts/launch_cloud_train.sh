#!/usr/bin/env sh
set -eu

# Render a server-local config and run training on the server.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

if [ "$#" -lt 1 ]; then
  echo "usage: $0 BASE_CONFIG_JSON [RUN_NAME]" >&2
  exit 1
fi

BASE_CONFIG=$1
RUN_NAME=${2:-}
RENDERED_DIR=${FYP_RENDERED_CONFIG_ROOT:-"$PROJECT_ROOT/Training/generated"}
USE_TORCHRUN=${USE_TORCHRUN:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

mkdir -p "$RENDERED_DIR"

STAMP=$(date -u +"%Y%m%dT%H%M%SZ")
if [ -z "$RUN_NAME" ]; then
  RUN_NAME=$(basename "$BASE_CONFIG" .json)-$STAMP
fi

RENDERED_CONFIG="$RENDERED_DIR/$RUN_NAME.json"

python3 "$PROJECT_ROOT/scripts/render_cloud_config.py" \
  --base-config "$BASE_CONFIG" \
  --output-config "$RENDERED_CONFIG" \
  --run-name "$RUN_NAME"

RUN_DIR=$(python3 - "$RENDERED_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload["output_dir"])
PY
)

echo "[INFO] rendered_config=$RENDERED_CONFIG"
echo "[INFO] run_dir=$RUN_DIR"

if [ "$USE_TORCHRUN" = "1" ]; then
  if [ "$NNODES" = "1" ]; then
    torchrun \
      --standalone \
      --nproc_per_node="$NPROC_PER_NODE" \
      "$PROJECT_ROOT/main_train_ensemble.py" \
      --config "$RENDERED_CONFIG"
  else
    torchrun \
      --nnodes="$NNODES" \
      --node_rank="$NODE_RANK" \
      --nproc_per_node="$NPROC_PER_NODE" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      "$PROJECT_ROOT/main_train_ensemble.py" \
      --config "$RENDERED_CONFIG"
  fi
else
  python3 "$PROJECT_ROOT/main_train_ensemble.py" --config "$RENDERED_CONFIG"
fi

echo "[INFO] training complete"
echo "[INFO] run_dir=$RUN_DIR"
