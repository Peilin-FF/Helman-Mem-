#!/usr/bin/env bash
# Peer-generation + execution-scoring for the APPS code training source.
# APPS replaces BigCodeBench (where every peer scored ~0%). We keep only the easy,
# stdin-style slice (see load_apps) so peers actually spread.
#
# Code correctness is scored by EXECUTING peer code against stdin/stdout tests
# (FEEDBACK_CODE_EXEC_ALLOW=1), not the sympy/F1 grader used for math/rag.
# One dataset, 3 shards (one per idle GPU), resumable via completed_ids.
set -u
cd "$(dirname "$0")/../../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=.

read -r -a GPUS <<< "${GPUS:-0 1 6}"
NSHARDS=${#GPUS[@]}
OUTD=data/mixed_train_sources
LOGD=logs/peergen_newtrain
mkdir -p "$OUTD" "$LOGD"

echo "=== peer-gen: apps (${NSHARDS} shards) ==="
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  out="$OUTD/apps_shard${i}.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m data.builders.common.generate_setting_a_peers \
    --config configs/peergen/apps.yaml \
    --num_shards "$NSHARDS" --shard_index "$i" \
    --output "$out" \
    > "$LOGD/apps_shard${i}.log" 2>&1 &
done
wait
echo "=== apps gen done: $(cat "$OUTD"/apps_shard*.jsonl 2>/dev/null | wc -l) records ==="

# Merge + execute-score (code pass@1). FEEDBACK_CODE_EXEC_ALLOW=1 acknowledges running
# model-generated code; this host is disposable.
raw="$OUTD/apps.raw.jsonl"
graded="$OUTD/apps.graded.jsonl"
cat "$OUTD"/apps_shard*.jsonl > "$raw"
echo "=== score apps (execute) -> $graded ==="
FEEDBACK_CODE_EXEC_ALLOW=1 python -u -m data.builders.common.score_code_peers \
  --input "$raw" --output "$graded" --timeout 8.0

echo "=== apps per-peer pass@1 ==="
python -u -c "
import json
n=0; s={'peer_0':0,'peer_1':0,'peer_2':0}
for l in open('$graded'):
    r=json.loads(l); pc=r.get('peer_correct',{}); n+=1
    for k in s: s[k]+=int(round(float(pc.get(k,0))))
print('apps n=',n,'correct:',{k:f'{100*v/n:.0f}%' for k,v in s.items()})
"
echo "=== done -> $graded. Next: bash data/builders/mixed_train/build_new_train.sh ==="
