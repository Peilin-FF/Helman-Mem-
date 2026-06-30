#!/usr/bin/env bash
# New experiment: train Sigma Mem on top of trained LoRA-AR center models, then
# evaluate on data/CF_unified/{p0,p50,p70,p90}.jsonl.
#
# This does not touch the standalone LoRA AR/BCE eval outputs.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

CONDA_BIN="${CONDA_BIN:-/home/peilin/miniconda3/bin/conda}"
HFM=/mnt/data/peilin/HF_MODEL
TRAIN_CFG=configs/symmetric_memory_train4096.yaml
EVAL_CFG=configs/symmetric_memory.yaml
TRAIN_CFG_9B=/tmp/sigma_on_lora_ar_train_q35_9b.yaml
EVAL_CFG_9B=/tmp/sigma_on_lora_ar_eval_q35_9b.yaml
TRAIN=data/mixed_train_labeled.jsonl
TEST_DIR=data/CF_unified
LOGD=logs/sigma_on_lora_ar
mkdir -p "$LOGD"

cp "$TRAIN_CFG" "$TRAIN_CFG_9B"
cat >> "$TRAIN_CFG_9B" <<'YAML'
device_map: auto
YAML
cp "$EVAL_CFG" "$EVAL_CFG_9B"
cat >> "$EVAL_CFG_9B" <<'YAML'
device_map: auto
YAML

read -r -a GPUS <<< "${GPUS:-1 2 3 4}"
read -r -a SPLITS <<< "${SPLITS:-p0 p50 p70 p90}"
read -r -a ARMS <<< "${ARMS:-proto}"

EXTRA_TRAIN_ARGS=()
if [ "${MAX_STEPS:-}" != "" ]; then
  EXTRA_TRAIN_ARGS+=(--max_steps "$MAX_STEPS")
fi

# tag | base model path | LoRA-AR checkpoint | conda env
MODELS=(
  "q3_0.6b|$HFM/Qwen3-0.6B|outputs/lora_ar_q3_0.6b|sigma"
  "q3_4b|$HFM/Qwen3-4B|outputs/lora_ar_q3_4b|sigma"
  "q3_8b|$HFM/Qwen3-8B|outputs/lora_ar_q3_8b|sigma"
  "q35_4b|$HFM/Qwen3.5-4B|outputs/lora_ar_q35_4b|sigma3_5"
  "q35_9b|$HFM/Qwen3.5-9B|outputs/lora_ar_q35_9b|sigma3_5"
)

wait_for_lora_ar() {
  local ckpt="$1"
  while [ ! -f "$ckpt/joint_selector_head.pt" ] || [ ! -d "$ckpt/lora_adapter" ]; do
    echo "[sigma+lora] waiting for LoRA-AR checkpoint $ckpt"
    sleep 60
  done
}

echo "=== TRAIN Sigma Mem on LoRA-AR centers: ${#MODELS[@]} models ==="
i=0
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag model_path lora_ckpt env_name <<< "$entry"
  if [ "$tag" = "q35_9b" ]; then
    continue
  fi
  wait_for_lora_ar "$lora_ckpt"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  out="outputs/sigma_on_lora_ar_${tag}/proto"
  log="$LOGD/train_${tag}.log"
  if [ -f "$out/sym_memory.pt" ]; then
    echo "[train:skip] $out already exists"
    continue
  fi
  echo "[train] gpu=$gpu env=$env_name base=$model_path lora=$lora_ckpt -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u train_symmetric_memory.py \
    --config "$TRAIN_CFG" \
    --central_model "$model_path" \
    --center_lora_checkpoint "$lora_ckpt" \
    --phi_mode proto \
    --use_joint off \
    --per_peer_decay on \
    --diff_write on \
    --offline_data "$TRAIN" \
    --output_dir "$out" \
    "${EXTRA_TRAIN_ARGS[@]}" \
    > "$log" 2>&1 &
  i=$((i + 1))
  if (( i % ${#GPUS[@]} == 0 )); then
    wait
  fi
done
wait

for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag model_path lora_ckpt env_name <<< "$entry"
  if [ "$tag" != "q35_9b" ]; then
    continue
  fi
  wait_for_lora_ar "$lora_ckpt"
  out="outputs/sigma_on_lora_ar_${tag}/proto"
  log="$LOGD/train_${tag}.log"
  if [ -f "$out/sym_memory.pt" ]; then
    echo "[train:skip] $out already exists"
    continue
  fi
  echo "[train] gpus=${Q35_9B_GPUS:-1,2,3,4} env=$env_name base=$model_path lora=$lora_ckpt -> $out"
  CUDA_VISIBLE_DEVICES="${Q35_9B_GPUS:-1,2,3,4}" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u train_symmetric_memory.py \
    --config "$TRAIN_CFG_9B" \
    --central_model "$model_path" \
    --center_lora_checkpoint "$lora_ckpt" \
    --phi_mode proto \
    --use_joint off \
    --per_peer_decay on \
    --diff_write on \
    --offline_data "$TRAIN" \
    --output_dir "$out" \
    "${EXTRA_TRAIN_ARGS[@]}" \
    > "$log" 2>&1
done
echo "=== TRAIN done ==="

echo "=== EVAL Sigma Mem on LoRA-AR centers ==="
jobs=()
for entry in "${MODELS[@]}"; do
  for arm in "${ARMS[@]}"; do
    for split in "${SPLITS[@]}"; do
      jobs+=("$entry|$arm|$split")
    done
  done
done

i=0
for job in "${jobs[@]}"; do
  IFS='|' read -r tag model_path lora_ckpt env_name arm split <<< "$job"
  if [ "$tag" = "q35_9b" ]; then
    continue
  fi
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  ckpt="outputs/sigma_on_lora_ar_${tag}/proto"
  data="$TEST_DIR/${split}.jsonl"
  out="outputs/eval_sigma_on_lora_ar_${tag}/${arm}_${split}"
  log="$LOGD/eval_${tag}_${arm}_${split}.log"
  if [ -f "$out/eval_metrics.json" ]; then
    echo "[eval:skip] $out already exists"
    continue
  fi
  extra=()
  if [ "$arm" = "ablate" ]; then
    extra+=(--ablate_memory)
  fi
  echo "[eval] gpu=$gpu env=$env_name arm=$arm split=$split ckpt=$ckpt -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u eval_symmetric_memory.py \
    --config "$EVAL_CFG" \
    --central_model "$model_path" \
    --center_lora_checkpoint "$lora_ckpt" \
    --checkpoint "$ckpt" \
    --phi_mode proto \
    --use_joint off \
    --per_peer_decay on \
    "${extra[@]}" \
    --offline_data "$data" \
    --output "$out" \
    > "$log" 2>&1 &
  i=$((i + 1))
  if (( i % ${#GPUS[@]} == 0 )); then
    wait
  fi
done
wait

for job in "${jobs[@]}"; do
  IFS='|' read -r tag model_path lora_ckpt env_name arm split <<< "$job"
  if [ "$tag" != "q35_9b" ]; then
    continue
  fi
  ckpt="outputs/sigma_on_lora_ar_${tag}/proto"
  data="$TEST_DIR/${split}.jsonl"
  out="outputs/eval_sigma_on_lora_ar_${tag}/${arm}_${split}"
  log="$LOGD/eval_${tag}_${arm}_${split}.log"
  if [ -f "$out/eval_metrics.json" ]; then
    echo "[eval:skip] $out already exists"
    continue
  fi
  extra=()
  if [ "$arm" = "ablate" ]; then
    extra+=(--ablate_memory)
  fi
  echo "[eval] gpus=${Q35_9B_GPUS:-1,2,3,4} env=$env_name arm=$arm split=$split ckpt=$ckpt -> $out"
  CUDA_VISIBLE_DEVICES="${Q35_9B_GPUS:-1,2,3,4}" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u eval_symmetric_memory.py \
    --config "$EVAL_CFG_9B" \
    --central_model "$model_path" \
    --center_lora_checkpoint "$lora_ckpt" \
    --checkpoint "$ckpt" \
    --phi_mode proto \
    --use_joint off \
    --per_peer_decay on \
    "${extra[@]}" \
    --offline_data "$data" \
    --output "$out" \
    > "$log" 2>&1
done
echo "=== done. metrics under outputs/eval_sigma_on_lora_ar_*/*/eval_metrics.json ==="
