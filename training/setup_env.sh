#!/usr/bin/env bash
# One-time server setup for training/ (conda env `sigma`: torch 2.6.0+cu124, vllm 0.8.5, ray 2.55 already present).
# Installs only into the env, nothing system-wide.  Idempotent.
#   rproj run 'bash training/setup_env.sh'
set -euo pipefail
PIP="${PIP:-/home/peilin/miniconda3/envs/sigma/bin/pip}"
WHEEL_DIR="${WHEEL_DIR:-$HOME/wheels}"
FA_WHEEL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$FA_WHEEL"
$PIP install "hydra-core>=1.3" "omegaconf>=2.3" "tensordict==0.6.2" codetiming pylatexenc pybind11 torchdata "pyarrow>=19"
if ! "$(dirname "$PIP")/python" -c "import flash_attn" 2>/dev/null; then
  mkdir -p "$WHEEL_DIR"
  [ -f "$WHEEL_DIR/$FA_WHEEL" ] || curl -L --max-time 1800 -o "$WHEEL_DIR/$FA_WHEEL" "$FA_URL"
  $PIP install --no-deps "$WHEEL_DIR/$FA_WHEEL"
fi
"$(dirname "$PIP")/python" - <<'EOF'
import importlib
for m in ("hydra", "omegaconf", "tensordict", "codetiming", "flash_attn", "vllm", "ray", "torch", "transformers"):
    mod = importlib.import_module(m); print(f"{m:14s} {getattr(mod, '__version__', '?')}")
EOF
