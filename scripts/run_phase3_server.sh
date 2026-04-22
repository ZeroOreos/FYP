#!/usr/bin/env sh
set -eu

REMOTE_HOST=${REMOTE_HOST:-}
REMOTE_ROOT=${REMOTE_ROOT:-/home/wen/data/FYP}
REMOTE_DATA_ROOT=${REMOTE_DATA_ROOT:-/home/wen/data/FYP/Dataset/WebFace4M}
REMOTE_RUN_ROOT=${REMOTE_RUN_ROOT:-/home/wen/data/runs}
RUN_NAME=${RUN_NAME:-joint-test}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

if [ -z "$REMOTE_HOST" ]; then
  echo "Set REMOTE_HOST, e.g. REMOTE_HOST=wen@your-server" >&2
  exit 1
fi

ssh "$REMOTE_HOST" "
set -eu
cd \"$REMOTE_ROOT\"
export FYP_WEBFACE4M_ROOT=\"$REMOTE_DATA_ROOT\"
export FYP_OUTPUT_ROOT=\"$REMOTE_RUN_ROOT\"
export USE_TORCHRUN=1
export NPROC_PER_NODE=\"$NPROC_PER_NODE\"
sh scripts/launch_cloud_train.sh Training/joint_pool_webface4m_test.json \"$RUN_NAME\"
"
