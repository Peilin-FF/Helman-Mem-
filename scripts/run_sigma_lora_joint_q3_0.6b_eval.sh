#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

CONDA_BIN="${CONDA_BIN:-/home/peilin/miniconda3/bin/conda}"
GPU="${GPU:-4}"
TAG=q3_0.6b
MODEL=/mnt/data/peilin/HF_MODEL/Qwen3-0.6B
LORA=outputs/lora_ar_q3_0.6b
CKPT=outputs/sigma_on_lora_ar_q3_0.6b/proto
LOGD=logs/sigma_on_lora_ar
mkdir -p "$LOGD"

for split in p0 p50 p70 p90; do
  out="outputs/eval_sigma_on_lora_ar_${TAG}_joint/proto_${split}"
  if [ -f "$out/eval_metrics.json" ]; then
    echo "[skip] $out"
    continue
  fi

  echo "[eval-joint] gpu=$GPU tag=$TAG split=$split -> $out"
  CUDA_VISIBLE_DEVICES="$GPU" "$CONDA_BIN" run --no-capture-output -n sigma python -u eval_symmetric_memory.py \
    --config configs/symmetric_memory.yaml \
    --peer_mode joint \
    --central_model "$MODEL" \
    --center_lora_checkpoint "$LORA" \
    --checkpoint "$CKPT" \
    --phi_mode proto \
    --use_joint off \
    --per_peer_decay on \
    --offline_data "data/CF_unified/${split}.jsonl" \
    --output "$out"
done
