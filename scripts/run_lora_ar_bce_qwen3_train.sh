#!/usr/bin/env bash
# Train LoRA selector baselines on data/mixed_train_labeled.jsonl.
# Runs local Qwen3/Qwen3.5 center models x 2 selector losses:
#   - lora-AR  (autoregressive " Peer j" target)
#   - lora-BCE (per-peer correctness BCE head)
#
# Prompt identity is anonymous to match the symmetric-memory method's visible prompt.
# Training max_length is 4096.
set -u
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

CFG=configs/lora_joint_selector_train4096.yaml
TRAIN=data/mixed_train_labeled.jsonl
HFM=/mnt/data/peilin/HF_MODEL
LOGD=logs/lora_qwen3
mkdir -p "$LOGD"
CONDA_BIN="${CONDA_BIN:-/home/peilin/miniconda3/bin/conda}"
EXTRA_ARGS=()
if [ "${MAX_STEPS:-}" != "" ]; then
  EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4}"

# tag | model path | conda env
MODELS=(
  "q3_0.6b|$HFM/Qwen3-0.6B|sigma"
  "q3_4b|$HFM/Qwen3-4B|sigma"
  "q3_8b|$HFM/Qwen3-8B|sigma"
  "q35_4b|$HFM/Qwen3.5-4B|sigma3_5"
  "q35_9b|$HFM/Qwen3.5-9B|sigma3_5"
)

# output suffix | model_variant
VARIANTS=(
  "ar|ar_shared_state_selector"
  "bce|joint_bce_shared_state_selector"
)

jobs=()
for m in "${MODELS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    jobs+=("$m|$v")
  done
done

echo "=== train LoRA selectors: ${#jobs[@]} jobs ==="
i=0
for job in "${jobs[@]}"; do
  IFS='|' read -r tag model_path env_name suffix variant <<< "$job"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  out="outputs/lora_${suffix}_${tag}"
  log="$LOGD/train_${suffix}_${tag}.log"
  echo "[launch] gpu=$gpu env=$env_name variant=$variant model=$model_path -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u train_joint_selector.py \
    --config "$CFG" \
    --central_model "$model_path" \
    --offline_data "$TRAIN" \
    --model_variant "$variant" \
    --use_shared_state false \
    --use_lora true \
    --output_dir "$out" \
    "${EXTRA_ARGS[@]}" \
    > "$log" 2>&1 &
  i=$((i + 1))
  if (( i % ${#GPUS[@]} == 0 )); then
    wait
  fi
done
wait
echo "=== done. checkpoints under outputs/lora_{ar,bce}_{q3,q35}_* ==="
