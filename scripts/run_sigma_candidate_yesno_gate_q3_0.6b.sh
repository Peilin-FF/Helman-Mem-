#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[error] line=$LINENO status=$status" >&2' ERR

cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

PYTHON="${PYTHON:-/home/peilin/miniconda3/envs/sigma/bin/python}"
GPU="${GPU:-4}"
MODEL="${MODEL:-/mnt/data/peilin/HF_MODEL/Qwen3-0.6B}"
TRAIN="${TRAIN:-data/mixed_train_labeled.jsonl}"
TEST_DIR="${TEST_DIR:-data/CF_unified}"
GATE="${GATE:-margin}"
THRESHOLD="${THRESHOLD:-2.0}"
TAG="${TAG:-q3_0.6b}"

CKPT="outputs/sigma_candidate_yesno_gate_${GATE}${THRESHOLD}_${TAG}/proto"
EVAL_ROOT="outputs/eval_sigma_candidate_yesno_gate_${GATE}${THRESHOLD}_${TAG}"
LOG_ROOT="logs/candidate_yesno_gate_${GATE}${THRESHOLD}_${TAG}"
mkdir -p "$LOG_ROOT"

EXTRA_TRAIN_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then
  EXTRA_TRAIN_ARGS+=(--max_steps "$MAX_STEPS")
fi

EXTRA_EVAL_ARGS=()
if [ -n "${MAX_EXAMPLES:-}" ]; then
  EXTRA_EVAL_ARGS+=(--max_examples "$MAX_EXAMPLES")
fi

echo "[run] GPU=$GPU model=$MODEL"
echo "[run] checkpoint=$CKPT"
echo "[run] gate=$GATE threshold=$THRESHOLD score_mode=candidate_yesno"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u train_symmetric_memory.py \
  --config configs/symmetric_memory_candidate_yesno.yaml \
  --central_model "$MODEL" \
  --offline_data "$TRAIN" \
  --output_dir "$CKPT" \
  --score_mode candidate_yesno \
  --peer_mode joint \
  --per_peer_decay off \
  --diff_write on \
  --confusion_gate "$GATE" \
  --confusion_threshold "$THRESHOLD" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  > "$LOG_ROOT/train.log" 2>&1

for split in p0 p50 p70 p90; do
  echo "[eval] split=$split"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u eval_symmetric_memory.py \
    --config configs/symmetric_memory_candidate_yesno.yaml \
    --checkpoint "$CKPT" \
    --central_model "$MODEL" \
    --offline_data "$TEST_DIR/${split}.jsonl" \
    --output "$EVAL_ROOT/proto_${split}" \
    --score_mode candidate_yesno \
    --peer_mode joint \
    --per_peer_decay off \
    --confusion_gate "$GATE" \
    --confusion_threshold "$THRESHOLD" \
    "${EXTRA_EVAL_ARGS[@]}" \
    > "$LOG_ROOT/eval_${split}.log" 2>&1
done

EVAL_ROOT="$EVAL_ROOT" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EVAL_ROOT"])
for split in ["p0", "p50", "p70", "p90"]:
    path = root / f"proto_{split}" / "eval_metrics.json"
    if not path.exists():
        continue
    m = json.loads(path.read_text())
    gate = ""
    if m.get("confusion_gate") != "off":
        gate = f" gate_use={m.get('memory_used')}/{m.get('memory_gate_total')} ({m.get('memory_use_rate')})"
    print(f"{split}: {m['accuracy'] * 100:.2f}{gate}")
PY
