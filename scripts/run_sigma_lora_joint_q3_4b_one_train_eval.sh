#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

CONDA_BIN="${CONDA_BIN:-/home/peilin/miniconda3/bin/conda}"
MODEL=/mnt/data/peilin/HF_MODEL/Qwen3-4B
TRAIN=data/mixed_train_labeled.jsonl
TEST_DIR=data/CF_unified
TRAIN_CFG=configs/sigma_lora_joint_train4096.yaml
EVAL_CFG=configs/symmetric_memory.yaml
MODE=one
TAG=q3_4b
LOGD=logs/sigma_lora_joint_${TAG}_${MODE}
mkdir -p "$LOGD"

ckpt="outputs/sigma_lora_joint_${TAG}_${MODE}/proto"
if [ ! -f "$ckpt/sym_memory.pt" ] || [ ! -d "$ckpt/lora_adapter" ]; then
  echo "[train] gpu=${TRAIN_GPU:-0} tag=$TAG mode=$MODE -> $ckpt"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-0}" "$CONDA_BIN" run --no-capture-output -n sigma python -u train_symmetric_memory.py \
    --config "$TRAIN_CFG" \
    --central_model "$MODEL" \
    --train_center_lora on \
    --peer_mode "$MODE" \
    --phi_mode proto \
    --use_joint off \
    --per_peer_decay on \
    --diff_write on \
    --offline_data "$TRAIN" \
    --output_dir "$ckpt" \
    > "$LOGD/train.log" 2>&1
else
  echo "[train:skip] $ckpt"
fi

splits=(p0 p50 p70 p90)
for split in "${splits[@]}"; do
  out="outputs/eval_sigma_lora_joint_${TAG}_${MODE}/proto_${split}"
  if [ -f "$out/eval_metrics.json" ]; then
    echo "[eval:skip] split=$split -> $out"
    continue
  fi
  echo "[eval] gpu=${EVAL_GPU:-0} tag=$TAG mode=$MODE split=$split -> $out"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" "$CONDA_BIN" run --no-capture-output -n sigma python -u eval_symmetric_memory.py \
    --config "$EVAL_CFG" \
    --peer_mode "$MODE" \
    --central_model "$MODEL" \
    --center_lora_checkpoint "$ckpt" \
    --checkpoint "$ckpt" \
    --phi_mode proto \
    --use_joint off \
    --per_peer_decay on \
    --offline_data "$TEST_DIR/${split}.jsonl" \
    --output "$out" \
    > "$LOGD/eval_${split}.log" 2>&1
done

echo "=== done: outputs/eval_sigma_lora_joint_${TAG}_${MODE}/proto_*/eval_metrics.json ==="
