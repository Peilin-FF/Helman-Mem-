#!/usr/bin/env bash
# Peer-generation for the NEW training sources (GSM8K + SQuAD), 8-GPU sharded + resumable.
# Mirrors the offline vLLM peer-gen pipeline. One dataset at a time; each dataset is split
# into 8 shards (one per GPU). Re-running resumes via completed_ids in each shard file.
#
# After this completes, label + merge -> 3-peer mixed training set:
#   see data/builders/mixed_train/build_new_train.sh (label + build step).
set -u
cd "$(dirname "$0")/../../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=.

# GPUs to shard across. Override with e.g. GPUS="0 1 2 3 4 5 6 7" when all are free;
# default is the 3 currently-idle GPUs (0 1 6) so we don't stomp on other jobs.
read -r -a GPUS <<< "${GPUS:-0 1 6}"
NSHARDS=${#GPUS[@]}
OUTD=data/mixed_train_sources
LOGD=logs/peergen_newtrain
mkdir -p "$OUTD" "$LOGD"

# dataset -> peergen config
declare -A CFG=(
  [gsm8k]=configs/peergen/gsm8k.yaml
  [squad]=configs/peergen/squad.yaml
)

for ds in gsm8k squad; do
  echo "=== peer-gen: $ds (${NSHARDS} shards) ==="
  for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    out="$OUTD/${ds}_shard${i}.jsonl"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m data.builders.common.generate_setting_a_peers \
      --config "${CFG[$ds]}" \
      --num_shards "$NSHARDS" --shard_index "$i" \
      --output "$out" \
      > "$LOGD/${ds}_shard${i}.log" 2>&1 &
  done
  wait
  echo "=== $ds done: $(cat "$OUTD/${ds}"_shard*.jsonl 2>/dev/null | wc -l) records ==="
done

echo "=== ALL peer-gen done. Next: bash data/builders/mixed_train/build_new_train.sh ==="
