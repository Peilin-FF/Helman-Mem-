#!/usr/bin/env bash
# Label + build step for the NEW training set, run AFTER scripts/run_peergen_newtrain.sh.
#   1. merge per-shard peer-gen output -> one jsonl per dataset
#   2. grade per-peer correctness (precompute_peer_correct.py: math=sympy, rag=F1)
#   3. merge GSM8K(math) + SQuAD(rag) + BigCodeBench(code) -> 3-task mixed_train_labeled
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=.

OUTD=data/v3/peergen

for ds in gsm8k squad; do
  raw="$OUTD/${ds}.raw.jsonl"
  graded="$OUTD/${ds}.graded.jsonl"
  echo "=== merge $ds shards -> $raw ==="
  cat "$OUTD/${ds}"_shard*.jsonl > "$raw"
  echo "  $(wc -l < "$raw") records"
  echo "=== grade $ds -> $graded ==="
  python -u scripts/precompute_peer_correct.py "$raw" "$graded"
done

echo "=== build mixed 3-task training set (1000/task) ==="
python -u scripts/build_new_train_labeled.py --per_task 1000

echo "=== done -> data/v3/mixed_train_labeled.jsonl ==="
