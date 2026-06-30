#!/usr/bin/env bash
# Eval the 5 m5 trust-memory checkpoints. Records peer_selection counts (eval_metrics.json
# + selections.jsonl). max_length=8192 for ALL (comparable to the earlier 3-model results).
# Qwen3.5 models run under tf5_overlay. The 9B uses device_map=auto (multi-GPU) to avoid OOM;
# the four smaller models run single-GPU in parallel across GPUs 2/3/4/5.
set -u
cd "$(dirname "$0")/.."
HFM=/workspace/cloud_android/fengpeilin/HF_Models
OV=/workspace/cloud_android/fengpeilin/tf5_overlay
CFG=configs/symmetric_memory.yaml          # 8192, single-device
CFG9B=/tmp/sym_9b_eval.yaml                 # 8192 + device_map auto
LOGD=logs/m5; mkdir -p "$LOGD"
VIEWS=(unified math code rag)
PROPS=(p0 p50 p70 p90)

# ---- Phase 1: the 4 single-GPU models, parallel across GPUs 2/3/4/5 ----
# tag|dir|gpu|overlay
SMALL=(
  "q3_0.6b|$HFM/Qwen3-0.6B|2|0"
  "q3_4b|$HFM/Qwen3-4B-Instruct-2507|3|0"
  "q3_8b|$HFM/Qwen3-8B|4|0"
  "q35_4b|$HFM/Qwen3.5-4B|5|1"
)
echo "=== EVAL phase 1: 4 single-GPU models ==="
pids=()
for entry in "${SMALL[@]}"; do
  IFS='|' read -r tag dir gpu ov <<< "$entry"
  PP="."; [ "$ov" = 1 ] && PP="$OV:."
  (
    for arm in proto ablate; do
      for v in "${VIEWS[@]}"; do
        for p in "${PROPS[@]}"; do
          data="data/v3_unified/${p}.jsonl"; [ "$v" != unified ] && data="data/v3_unified_${v}/${p}.jsonl"
          extra=""; [ "$arm" = ablate ] && extra="--ablate_memory"
          HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES=$gpu \
            python -u eval_symmetric_memory.py --config "$CFG" --central_model "$dir" \
              --checkpoint "outputs/m5_${tag}/proto" --phi_mode proto --use_joint off --per_peer_decay on $extra \
              --offline_data "$data" --output "outputs/eval_m5_${tag}/${arm}_${v}_${p}" \
              > "$LOGD/eval_${tag}_${arm}_${v}_${p}.log" 2>&1
        done
      done
    done
  ) &
  pids+=($!)
done
wait
echo "=== phase 1 done ==="

# ---- Phase 2: the 9B with device_map across all free GPUs (2,3,4,5,7) ----
echo "=== EVAL phase 2: q35_9b (device_map auto) ==="
for arm in proto ablate; do
  for v in "${VIEWS[@]}"; do
    for p in "${PROPS[@]}"; do
      data="data/v3_unified/${p}.jsonl"; [ "$v" != unified ] && data="data/v3_unified_${v}/${p}.jsonl"
      extra=""; [ "$arm" = ablate ] && extra="--ablate_memory"
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$OV:." CUDA_VISIBLE_DEVICES=2,3,4,5,7 \
        python -u eval_symmetric_memory.py --config "$CFG9B" --central_model "$HFM/Qwen3.5-9B" \
          --checkpoint "outputs/m5_q35_9b/proto" --phi_mode proto --use_joint off --per_peer_decay on $extra \
          --offline_data "$data" --output "outputs/eval_m5_q35_9b/${arm}_${v}_${p}" \
          > "$LOGD/eval_q35_9b_${arm}_${v}_${p}.log" 2>&1
    done
  done
done
echo "=== ALL EVAL done. peer_selection in outputs/eval_m5_*/*/eval_metrics.json ==="
