#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_BIN="${CONDA_BIN:-/home/peilin/miniconda3/bin/conda}"
GPU="${GPU:-4}"
MODEL="/mnt/data/peilin/HF_MODEL/Qwen3-0.6B"
LORA_CKPT="outputs/lora_ar_8192_q3_0.6b"
TRAIN_DATA="data/mixed_train_labeled.jsonl"
TEST_DIR="data/CF_unified"
CFG="configs/symmetric_memory.yaml"
TAG="q3_0.6b"
GATE="margin"
THR="2.0"

CKPT="outputs/sigma_on_lora_ar_8192_joint_confusion_${GATE}${THR}_${TAG}/proto"
EVAL_ROOT="outputs/eval_sigma_on_lora_ar_8192_joint_confusion_${GATE}${THR}_${TAG}"
LOG_ROOT="logs/confusion_gate_lora_sigma_${TAG}"
mkdir -p "$LOG_ROOT" "$(dirname "$CKPT")" "$EVAL_ROOT"

echo "[config] gpu=$GPU model=$MODEL"
echo "[config] center_lora=$LORA_CKPT"
echo "[config] ckpt=$CKPT"
echo "[config] eval_root=$EVAL_ROOT"
echo "[config] confusion_gate=$GATE threshold=$THR"

echo "[train] start"
CUDA_VISIBLE_DEVICES="$GPU" "$CONDA_BIN" run --no-capture-output -n sigma python -u train_symmetric_memory.py \
  --config "$CFG" \
  --central_model "$MODEL" \
  --center_lora_checkpoint "$LORA_CKPT" \
  --phi_mode proto \
  --peer_mode joint \
  --use_joint off \
  --per_peer_decay on \
  --diff_write on \
  --confusion_gate "$GATE" \
  --confusion_threshold "$THR" \
  --offline_data "$TRAIN_DATA" \
  --output_dir "$CKPT" \
  > "$LOG_ROOT/train.log" 2>&1
echo "[train] done"

for split in p0 p50 p70 p90; do
  out="$EVAL_ROOT/proto_${split}"
  mkdir -p "$out"
  echo "[eval] start split=$split out=$out"
  CUDA_VISIBLE_DEVICES="$GPU" "$CONDA_BIN" run --no-capture-output -n sigma python -u eval_symmetric_memory.py \
    --config "$CFG" \
    --central_model "$MODEL" \
    --center_lora_checkpoint "$LORA_CKPT" \
    --checkpoint "$CKPT" \
    --phi_mode proto \
    --peer_mode joint \
    --use_joint off \
    --per_peer_decay on \
    --confusion_gate "$GATE" \
    --confusion_threshold "$THR" \
    --offline_data "$TEST_DIR/${split}.jsonl" \
    --output "$out" \
    > "$LOG_ROOT/eval_${split}.log" 2>&1
  echo "[eval] done split=$split"
done

echo "[done] metrics:"
find "$EVAL_ROOT" -name eval_metrics.json -print | sort
