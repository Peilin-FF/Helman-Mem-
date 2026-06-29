#!/usr/bin/env bash
# Train + eval the symmetric-memory (proto) trust selector on the NEW 3-task set.
#
# Train: data/v3/mixed_train_labeled.jsonl (math+code+rag, per-task trust gradient;
#   code revived via APPS + fixed exec grader). All three tasks are SEEN here.
# Eval: the recent per-task CF streams data/v3_unified_{task}/{p0,p70}.jsonl. p0 is the
#   clean held-out benchmark; p70 swaps strong<->weak responses so an identity-only
#   selector is misled and a response-reading trust memory should win. We also run an
#   --ablate_memory arm (no steering = NO-WRITE control) to isolate the memory's lift.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.

MODEL=Qwen/Qwen3-4B-Instruct-2507
CFG=configs/symmetric_memory.yaml
TRAIN=data/v3/mixed_train_labeled.jsonl
read -r -a GPUS <<< "${GPUS:-2 3 4 5 7}"     # 0/1/6 hold leaked mem from orphan vLLM
TASKS=(math code rag)
PROPS=(p0 p70)
OUTD=outputs/mixed3
EVALD=outputs/eval_mixed3
LOGD=logs/mixed3
mkdir -p "$OUTD" "$EVALD" "$LOGD"

echo "=== TRAIN proto on 3-task mixed set (gpu ${GPUS[0]}) ==="
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u train_symmetric_memory.py \
  --config "$CFG" --central_model "$MODEL" \
  --phi_mode proto --use_joint off \
  --offline_data "$TRAIN" \
  --output_dir "$OUTD/proto" \
  > "$LOGD/train_proto.log" 2>&1
echo "=== TRAIN done -> $OUTD/proto ==="

echo "=== EVAL (arm x task x prop) ==="
# Build the job list: proto (trained memory) + ablate (no-write baseline).
jobs=()
for arm in proto ablate; do
  for t in "${TASKS[@]}"; do
    for p in "${PROPS[@]}"; do jobs+=("$arm:$t:$p"); done
  done
done

i=0
for job in "${jobs[@]}"; do
  arm="${job%%:*}"; rest="${job#*:}"; t="${rest%%:*}"; p="${rest##*:}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  extra=""; [ "$arm" = ablate ] && extra="--ablate_memory"
  CUDA_VISIBLE_DEVICES="$gpu" python -u eval_symmetric_memory.py \
    --config "$CFG" --central_model "$MODEL" \
    --checkpoint "$OUTD/proto" --phi_mode proto --use_joint off $extra \
    --offline_data "data/v3_unified_${t}/${p}.jsonl" \
    --output "$EVALD/${arm}_${t}_${p}" \
    > "$LOGD/eval_${arm}_${t}_${p}.log" 2>&1 &
  i=$((i + 1))
  if (( i % ${#GPUS[@]} == 0 )); then wait; fi
done
wait
echo "=== EVAL done ==="

echo "=== RESULTS (accuracy %) — proto vs ablate, per task ==="
printf "%-6s %-8s %8s %8s\n" task arm p0 p70
for t in "${TASKS[@]}"; do
  for arm in proto ablate; do
    row=""
    for p in "${PROPS[@]}"; do
      f="$EVALD/${arm}_${t}_${p}/eval_metrics.json"
      acc=$(python -c "import json;print(f\"{json.load(open('$f'))['accuracy']*100:.2f}\")" 2>/dev/null || echo ERR)
      row="$row $acc"
    done
    printf "%-6s %-8s %8s %8s\n" "$t" "$arm" $row
  done
done
echo "=== done. metrics under $EVALD/*/eval_metrics.json ==="
