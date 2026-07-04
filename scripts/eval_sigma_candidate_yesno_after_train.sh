#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

PYTHON="${PYTHON:-/home/peilin/miniconda3/envs/sigma/bin/python}"
TAG="${TAG:?TAG is required, e.g. q3_4b}"
MODEL="${MODEL:?MODEL is required}"
GPU="${GPU:?GPU is required}"
PID_FILE="${PID_FILE:?PID_FILE is required}"
CKPT="${CKPT:-outputs/sigma_candidate_yesno_${TAG}/proto}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval_sigma_candidate_yesno_${TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/candidate_yesno_${TAG}}"
TEST_DIR="${TEST_DIR:-data/CF_unified}"
THRESHOLD_ARGS=()

mkdir -p "$LOG_ROOT"

if [ -n "${CONFUSION_GATE:-}" ] && [ "$CONFUSION_GATE" != "off" ]; then
  THRESHOLD_ARGS+=(--confusion_gate "$CONFUSION_GATE")
  if [ -n "${CONFUSION_THRESHOLD:-}" ]; then
    THRESHOLD_ARGS+=(--confusion_threshold "$CONFUSION_THRESHOLD")
  fi
fi

if [ -f "$PID_FILE" ]; then
  train_pid="$(cat "$PID_FILE")"
  echo "[wait] tag=$TAG train_pid=$train_pid"
  while kill -0 "$train_pid" 2>/dev/null; do
    sleep 60
  done
fi

if [ ! -f "$CKPT/sym_memory.pt" ]; then
  echo "[error] missing checkpoint: $CKPT/sym_memory.pt" >&2
  exit 1
fi

for split in p0 p50 p70 p90; do
  echo "[eval] tag=$TAG split=$split gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u eval_symmetric_memory.py \
    --config configs/symmetric_memory_candidate_yesno.yaml \
    --checkpoint "$CKPT" \
    --central_model "$MODEL" \
    --offline_data "$TEST_DIR/${split}.jsonl" \
    --output "$EVAL_ROOT/proto_${split}" \
    --score_mode candidate_yesno \
    --peer_mode joint \
    --per_peer_decay off \
    "${THRESHOLD_ARGS[@]}" \
    > "$LOG_ROOT/eval_${split}.log" 2>&1
  tail -3 "$LOG_ROOT/eval_${split}.log"
done

"$PYTHON" - <<PY
import json
from pathlib import Path

root = Path("$EVAL_ROOT")
for split in ["p0", "p50", "p70", "p90"]:
    path = root / f"proto_{split}" / "eval_metrics.json"
    if not path.exists():
        continue
    m = json.loads(path.read_text())
    print(f"{split}: {m['accuracy'] * 100:.2f}")
PY
