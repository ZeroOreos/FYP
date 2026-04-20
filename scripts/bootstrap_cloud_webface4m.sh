#!/usr/bin/env sh
set -eu

# Run:
# FYP_WEBFACE4M_ROOT=/shared/fyp/data/WebFace4M sh scripts/bootstrap_cloud_webface4m.sh
#
# Optional:
# START_SHARD=0 END_SHARD=120 OVERWRITE=0 \
# FYP_WEBFACE4M_ROOT=/shared/fyp/data/WebFace4M \
# sh scripts/bootstrap_cloud_webface4m.sh
#
# This script downloads WebFace4M shards onto the remote server and builds
# server-local manifest files under $FYP_WEBFACE4M_ROOT/manifests.


SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

TARGET_ROOT=${FYP_WEBFACE4M_ROOT:-"$PROJECT_ROOT/Dataset/WebFace4M"}
START_SHARD=${START_SHARD:-0}
END_SHARD=${END_SHARD:-120}
VAL_FRACTION=${VAL_FRACTION:-0.02}
MIN_VAL_PER_CLASS=${MIN_VAL_PER_CLASS:-1}
MAX_VAL_PER_CLASS=${MAX_VAL_PER_CLASS:-5}
OVERWRITE=${OVERWRITE:-0}
EXTRACT_IMAGES=${EXTRACT_IMAGES:-0}
EXTRACT_ROOT=${EXTRACT_ROOT:-"$TARGET_ROOT/images"}

echo "[INFO] target_root=$TARGET_ROOT"
echo "[INFO] shard_range=$START_SHARD..$END_SHARD"

OVERWRITE_FLAG=""
if [ "$OVERWRITE" = "1" ]; then
  OVERWRITE_FLAG="--overwrite"
fi

python3 "$PROJECT_ROOT/scripts/import_webface4m.py" \
  --target-root "$TARGET_ROOT" \
  --start-shard "$START_SHARD" \
  --end-shard "$END_SHARD" \
  $OVERWRITE_FLAG

IMAGE_ROOT_ARG=""
if [ "$EXTRACT_IMAGES" = "1" ]; then
  python3 "$PROJECT_ROOT/scripts/extract_webface4m_images.py" \
    --target-root "$TARGET_ROOT" \
    --output-root "$EXTRACT_ROOT" \
    $OVERWRITE_FLAG
  IMAGE_ROOT_ARG="--image-root $EXTRACT_ROOT"
fi

python3 "$PROJECT_ROOT/scripts/prepare_webface4m_manifests.py" \
  --target-root "$TARGET_ROOT" \
  $IMAGE_ROOT_ARG \
  --val-fraction "$VAL_FRACTION" \
  --min-val-per-class "$MIN_VAL_PER_CLASS" \
  --max-val-per-class "$MAX_VAL_PER_CLASS"

echo "[INFO] bootstrap complete"
echo "[INFO] train_manifest=$TARGET_ROOT/manifests/train.jsonl"
echo "[INFO] val_manifest=$TARGET_ROOT/manifests/val.jsonl"
if [ "$EXTRACT_IMAGES" = "1" ]; then
  echo "[INFO] extracted_images=$EXTRACT_ROOT"
fi
