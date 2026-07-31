#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

PYTHON="${PYTHON:-python3}"
TAG="${TAG:?TAG is required, e.g. q3_4b}"
MODEL="${MODEL:?MODEL is required}"
GPU="${GPU:-0}"
MODEL_NAME="${MODEL##*/}"
CKPT="${CKPT:-models/Sigma-Mem/${MODEL_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval_sigma_candidate_yesno_${TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/candidate_yesno_${TAG}}"
TEST_DIR="${TEST_DIR:-data/counterfactual_3peer}"

mkdir -p "$LOG_ROOT"

if [ ! -f "$CKPT/sym_memory.pt" ]; then
  echo "[error] missing checkpoint: $CKPT/sym_memory.pt" >&2
  exit 1
fi

for split in cf_0 cf_50 cf_70 cf_90; do
  echo "[eval] tag=$TAG split=$split gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m tests.experiments.common.evaluate_sigma \
    --config configs/symmetric_memory_candidate_yesno.yaml \
    --checkpoint "$CKPT" \
    --central_model "$MODEL" \
    --offline_data "$TEST_DIR/${split}.jsonl" \
    --output "$EVAL_ROOT/proto_${split}" \
    > "$LOG_ROOT/eval_${split}.log" 2>&1
  tail -3 "$LOG_ROOT/eval_${split}.log"
done

"$PYTHON" - <<PY
import json
from pathlib import Path

root = Path("$EVAL_ROOT")
for split in ["cf_0", "cf_50", "cf_70", "cf_90"]:
    path = root / f"proto_{split}" / "eval_metrics.json"
    if not path.exists():
        continue
    m = json.loads(path.read_text())
    print(f"{split}: {m['accuracy'] * 100:.2f}")
PY
