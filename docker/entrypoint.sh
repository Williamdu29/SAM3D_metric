#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export PYTHONPATH="/workspace:/workspace/src:/workspace/libs/SAM3D:/opt/SAM3D:${PYTHONPATH:-}"
export SAM3D_CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-/workspace/checkpoints}"

echo "[SAM3D Docker] workspace: $(pwd)"
echo "[SAM3D Docker] python: $(which python)"
echo "[SAM3D Docker] checkpoint dir: ${SAM3D_CHECKPOINT_DIR}"

python - <<'PY'
import torch
print("[SAM3D Docker] torch:", torch.__version__)
print("[SAM3D Docker] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[SAM3D Docker] gpu:", torch.cuda.get_device_name(0))
PY

exec "$@"