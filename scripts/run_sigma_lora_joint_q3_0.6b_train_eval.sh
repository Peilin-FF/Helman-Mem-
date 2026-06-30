#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

CONDA_BIN="${CONDA_BIN:-/home/peilin/miniconda3/bin/conda}"
MODEL=/mnt/data/peilin/HF_MODEL/Qwen3-0.6B
TRAIN=data/mixed_train_labeled.jsonl
TEST_DIR=data/CF_unified
TRAIN_CFG=configs/sigma_lora_joint_train4096.yaml
EVAL_CFG=configs/symmetric_memory.yaml
LOGD=logs/sigma_lora_joint_q3_0.6b
mkdir -p "$LOGD"

train_one() {
  local mode="$1"
  local gpu="$2"
  local out="outputs/sigma_lora_joint_q3_0.6b_${mode}/proto"
  if [ -f "$out/sym_memory.pt" ] && [ -d "$out/lora_adapter" ]; then
    echo "[train:skip] mode=$mode -> $out"
    return
  fi
  echo "[train] gpu=$gpu mode=$mode -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n sigma python -u train_symmetric_memory.py \
    --config "$TRAIN_CFG" \
    --central_model "$MODEL" \
    --train_center_lora on \
    --peer_mode "$mode" \
    --phi_mode proto \
    --use_joint off \
    --per_peer_decay on \
    --diff_write on \
    --offline_data "$TRAIN" \
    --output_dir "$out" \
    > "$LOGD/train_${mode}.log" 2>&1
}

eval_one() {
  local mode="$1"
  local gpu="$2"
  local split="$3"
  local ckpt="outputs/sigma_lora_joint_q3_0.6b_${mode}/proto"
  local out="outputs/eval_sigma_lora_joint_q3_0.6b_${mode}/proto_${split}"
  if [ -f "$out/eval_metrics.json" ]; then
    echo "[eval:skip] mode=$mode split=$split -> $out"
    return
  fi
  echo "[eval] gpu=$gpu mode=$mode split=$split -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n sigma python -u eval_symmetric_memory.py \
    --config "$EVAL_CFG" \
    --peer_mode "$mode" \
    --central_model "$MODEL" \
    --center_lora_checkpoint "$ckpt" \
    --checkpoint "$ckpt" \
    --phi_mode proto \
    --use_joint off \
    --per_peer_decay on \
    --offline_data "$TEST_DIR/${split}.jsonl" \
    --output "$out" \
    > "$LOGD/eval_${mode}_${split}.log" 2>&1
}

train_one one "${GPU_ONE:-0}" &
pid_one=$!
train_one joint "${GPU_JOINT:-1}" &
pid_joint=$!
wait "$pid_one" "$pid_joint"

splits=(p0 p50 p70 p90)
for i in "${!splits[@]}"; do
  eval_one one "${EVAL_GPU_ONE:-0}" "${splits[$i]}" &
  eval_one joint "${EVAL_GPU_JOINT:-1}" "${splits[$i]}" &
  wait
done

echo "=== done: outputs/eval_sigma_lora_joint_q3_0.6b_{one,joint}/proto_*/eval_metrics.json ==="
