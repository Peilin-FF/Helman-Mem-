#!/usr/bin/env bash
# Train + eval the symmetric-memory (proto) trust selector across 5 center models.
#
# Final method on every model: per_peer_decay=on + diff_write=on (learned per-peer decay).
# Two arms per model: ours (proto steering) and prompt-only (--ablate_memory).
# Per-peer SELECTION COUNTS are recorded by eval_symmetric_memory.py (peer_selection in
# eval_metrics.json + per-example selections.jsonl).
#
# Qwen3.5 (qwen3_5) is NOT recognized by the base transformers 4.56; those two models run
# under the tf5_overlay env (transformers 5.10.2) via a PYTHONPATH prefix. Qwen3 models use
# the current env. No code changes needed (hidden_size is read off the loaded instance).
set -u
cd "$(dirname "$0")/.."
HFM=/workspace/cloud_android/fengpeilin/HF_Models
OVERLAY=/workspace/cloud_android/fengpeilin/tf5_overlay
CFG=configs/symmetric_memory.yaml
TRAIN=data/v3/mixed_train_labeled.jsonl
read -r -a GPUS <<< "${GPUS:-2 3 4 5 7}"   # 0/1/6 hold leaked mem
LOGD=logs/m5; mkdir -p "$LOGD"

# tag | model_dir | overlay(1/0)
MODELS=(
  "q3_0.6b|$HFM/Qwen3-0.6B|0"
  "q3_4b|$HFM/Qwen3-4B-Instruct-2507|0"
  "q3_8b|$HFM/Qwen3-8B|0"
  "q35_4b|$HFM/Qwen3.5-4B|1"
  "q35_9b|$HFM/Qwen3.5-9B|1"
)

pp_prefix() {  # echo PYTHONPATH for a model based on overlay flag
  if [ "$1" = "1" ]; then echo "$OVERLAY:."; else echo "."; fi
}

# ---- 1. TRAIN (one per GPU, in parallel) ----
echo "=== TRAIN 5 models (ours: per_peer_decay+diff_write) ==="
i=0
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag mdir ov <<< "$entry"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  PP=$(pp_prefix "$ov")
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES="$gpu" \
    python -u train_symmetric_memory.py \
      --config "$CFG" --central_model "$mdir" \
      --phi_mode proto --use_joint off --per_peer_decay on --diff_write on \
      --offline_data "$TRAIN" --output_dir "outputs/m5_${tag}/proto" \
      > "$LOGD/train_${tag}.log" 2>&1 &
  i=$((i + 1))
done
wait
echo "=== TRAIN done ==="

# ---- 2. EVAL (proto + ablate) x (unified math code rag) x (p0 p50 p70 p90) ----
echo "=== EVAL 5 models x 2 arms x 4 views x 4 props ==="
VIEWS=(unified math code rag)
PROPS=(p0 p50 p70 p90)
jobs=()
for entry in "${MODELS[@]}"; do
  for arm in proto ablate; do
    for v in "${VIEWS[@]}"; do
      for p in "${PROPS[@]}"; do jobs+=("$entry|$arm|$v|$p"); done
    done
  done
done

i=0
for job in "${jobs[@]}"; do
  IFS='|' read -r tag mdir ov arm v p <<< "$job"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  PP=$(pp_prefix "$ov")
  data="data/v3_unified/${p}.jsonl"; [ "$v" != unified ] && data="data/v3_unified_${v}/${p}.jsonl"
  extra=""; [ "$arm" = ablate ] && extra="--ablate_memory"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES="$gpu" \
    python -u eval_symmetric_memory.py \
      --config "$CFG" --central_model "$mdir" \
      --checkpoint "outputs/m5_${tag}/proto" --phi_mode proto --use_joint off --per_peer_decay on $extra \
      --offline_data "$data" --output "outputs/eval_m5_${tag}/${arm}_${v}_${p}" \
      > "$LOGD/eval_${tag}_${arm}_${v}_${p}.log" 2>&1 &
  i=$((i + 1))
  if (( i % ${#GPUS[@]} == 0 )); then wait; fi
done
wait
echo "=== EVAL done. metrics under outputs/eval_m5_*/*/eval_metrics.json (peer_selection recorded) ==="
