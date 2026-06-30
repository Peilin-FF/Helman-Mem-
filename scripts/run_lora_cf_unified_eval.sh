#!/usr/bin/env bash
# Evaluate trained LoRA AR/BCE selectors on data/CF_unified/{p0,p50,p70,p90}.jsonl.
# Uses max_length=8192 and anonymous prompts to match the symmetric-memory eval setup.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

CFG=configs/lora_joint_selector_eval8192.yaml
TEST_DIR=data/CF_unified
HFM=/mnt/data/peilin/HF_MODEL
LOGD=logs/lora_cf_eval
OUTD=outputs/lora_cf_eval
CONDA_BIN="${CONDA_BIN:-/home/peilin/miniconda3/bin/conda}"
mkdir -p "$LOGD" "$OUTD"

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4}"
read -r -a SPLITS <<< "${SPLITS:-p0 p50 p70 p90}"

EXTRA_ARGS=()
if [ "${MAX_SAMPLES:-}" != "" ]; then
  EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

if [ "${WAIT_FOR_PID:-}" != "" ]; then
  echo "[eval] waiting for training PID $WAIT_FOR_PID"
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 60
  done
fi

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

wait_for_checkpoint() {
  local suffix="$1"
  local tag="$2"
  local ckpt="outputs/lora_${suffix}_${tag}"
  local train_log="logs/lora_qwen3/train_${suffix}_${tag}.log"
  while ! grep -Fq "[train_joint] saved -> $ckpt" "$train_log" 2>/dev/null; do
    echo "[eval] waiting for $ckpt"
    sleep 60
  done
}

jobs=()
for m in "${MODELS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    for split in "${SPLITS[@]}"; do
      jobs+=("$m|$v|$split")
    done
  done
done

echo "=== eval LoRA selectors on CF_unified: ${#jobs[@]} jobs ==="
i=0
for job in "${jobs[@]}"; do
  IFS='|' read -r tag model_path env_name suffix variant split <<< "$job"
  ckpt="outputs/lora_${suffix}_${tag}"
  test_file="$TEST_DIR/${split}.jsonl"
  out="$OUTD/lora_${suffix}_${tag}/${split}"
  log="$LOGD/eval_${suffix}_${tag}_${split}.log"
  wait_for_checkpoint "$suffix" "$tag"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  echo "[launch] gpu=$gpu env=$env_name split=$split variant=$variant ckpt=$ckpt -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u eval_joint_selector.py \
    --config "$CFG" \
    --checkpoint "$ckpt" \
    --central_model "$model_path" \
    --offline_data "$test_file" \
    --model_variant "$variant" \
    --use_shared_state false \
    --init_state zeros \
    --mode read_only \
    --test_order orig \
    --eval_write_policy none \
    --output "$out" \
    "${EXTRA_ARGS[@]}" \
    > "$log" 2>&1 &
  i=$((i + 1))
  if (( i % ${#GPUS[@]} == 0 )); then
    wait
  fi
done
wait
echo "=== done. metrics under $OUTD/lora_{ar,bce}_{q3,q35}_*/{p0,p50,p70,p90}/eval_metrics.json ==="
