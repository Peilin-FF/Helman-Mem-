#!/usr/bin/env bash
# Label + build step for the mixed training set, run after run_peergen_newtrain.sh.
#   1. merge per-shard peer-gen output -> one jsonl per dataset
#   2. grade per-peer correctness (precompute_peer_correct.py: math=sympy, rag=F1)
#   3. merge GSM8K(math) + SQuAD(rag) + APPS(code) -> mixed_train/train.jsonl
set -eu
cd "$(dirname "$0")/../../.."
export PYTHONPATH=.

OUTD=data/mixed_train_sources

for ds in gsm8k squad; do
  raw="$OUTD/${ds}.raw.jsonl"
  graded="$OUTD/${ds}.graded.jsonl"
  echo "=== merge $ds shards -> $raw ==="
  cat "$OUTD/${ds}"_shard*.jsonl > "$raw"
  echo "  $(wc -l < "$raw") records"
  echo "=== grade $ds -> $graded ==="
  python -u -m data.builders.common.precompute_peer_correct "$raw" "$graded"
done

echo "=== build mixed 3-task training set (1000/task) ==="
python -u -m data.builders.mixed_train.build_new_train_labeled --per_task 1000

echo "=== done -> data/mixed_train/train.jsonl ==="
