#!/usr/bin/env bash
# Eval pps-AR (per-peer state + AR + anon) on the three RESULTS_final datasets.
# Goal: beat the prompt-only zero-shot baseline (the modest bar the user set).
# pps-AR scores each candidate "Peer s" with peer-s's OWN delta state (per-peer),
# online feedback writes labels_only per peer. init=trained, online_feedback.
# Streams: unified p0/p50/p70/p90, diag252, chaotic seed1/2/3.
# Output: outputs/eval_ppsar/<sz>/<stream>/
set -u
cd /workspace/cloud_android/fengpeilin/MAS
BASE_ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/cloud_android/fengpeilin/HF_HOME HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
HFM=/workspace/cloud_android/fengpeilin/HF_Models
CFG=configs/joint_ar_pps_anon.yaml; VAR=ar_shared_state_selector
OUT=outputs/eval_ppsar
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
mkdir -p "$OUT" logs/eval_ppsar
declare -A M=( [06]="$HFM/Qwen3-0.6B" [4b]="$HFM/Qwen3-4B-Instruct-2507" [8b]="$HFM/Qwen3-8B" )
# settings: trS_none (offline read-only) + trS_feedback (online per-peer write)
JOBS=()
for sz in 06 4b 8b; do
  ck=outputs/joint_ar_pps_anon_${sz}_episode
  [ -f "$ck/joint_selector_head.pt" ] || { echo "skip $sz (no ckpt)"; continue; }
  for stinfo in "trS_none|read_only|none" "trS_feedback|online_feedback|feedback"; do
    IFS='|' read -r sname mode write <<<"$stinfo"
    for p in p0 p50 p70 p90; do JOBS+=("$sz|data/v3_unified/${p}.jsonl|unified_${p}|$sname|$mode|$write"); done
    JOBS+=("$sz|data/v3_diagnostic/balanced_single_correct.jsonl|diag252|$sname|$mode|$write")
    for s in 1 2 3; do JOBS+=("$sz|data/chaotic_data/nonstat_stream_seed${s}.jsonl|chaotic_seed${s}|$sname|$mode|$write"); done
  done
done
echo "pps-AR eval jobs: ${#JOBS[@]}"
run_job(){ local g=$1 spec=$2; IFS='|' read -r sz df nm sname mode write <<<"$spec"
  local out=$OUT/${sz}/${nm}_${sname}; [ -f "$out/eval_metrics.json" ] && return; mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$g env $BASE_ENV PYTHONPATH=. python eval_joint_selector.py \
    --config "$CFG" --checkpoint outputs/joint_ar_pps_anon_${sz}_episode --central_model "${M[$sz]}" \
    --model_variant "$VAR" --use_shared_state true --offline_data "$df" --init_state trained \
    --mode "$mode" --test_order random --eval_write_policy "$write" --trust_scheduler none \
    --output "$out" > "logs/eval_ppsar/${sz}_${nm}_${sname}.log" 2>&1; }
declare -A P; for g in "${GPUS[@]}"; do P[$g]=0; done; qi=0
while true; do da=1
  for g in "${GPUS[@]}"; do pid=${P[$g]}
    if [ "$pid" -ne 0 ] && kill -0 "$pid" 2>/dev/null; then da=0; continue; fi
    if [ $qi -lt ${#JOBS[@]} ]; then run_job "$g" "${JOBS[$qi]}" & P[$g]=$!; qi=$((qi+1)); da=0; fi
  done
  [ $qi -ge ${#JOBS[@]} ] && [ $da -eq 1 ] && break; sleep 10
done
echo "=== pps-AR eval DONE $(date +%H:%M:%S) ==="