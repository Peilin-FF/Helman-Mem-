#!/usr/bin/env bash
# Build the parquet files for one central model from its memory-annotated prompt streams (stream order kept).
#   rproj run 'MODEL_TAG=q3_4b bash training/scripts/build_data.sh'
# Produces under outputs/rl/data/$MODEL_TAG:
#   labels only after the answer (ours):
#     train_memory.parquet        every peer solution, annotated with the memory's reliability estimate
#     train_hint_memory.parquet   the single solution of the peer the memory trusts most
#     train_peers.parquet         every peer solution, no annotation (no-memory ablation)
#     train_v30_memory.parquet    as train_memory, but only 30% of prompts have a verifier after the answer (memory pseudo-reward for the rest)
#   labels before the answer (classical baselines):
#     train_hint_label.parquet    the most reliable *verified-correct* peer's solution
#     train_hint_random.parquet   a random verified-correct peer's solution
#     train_peers_labeled.parquet every peer solution marked verified correct / incorrect (labels, no memory)
#   train_none.parquet            question only (plain RLVR)
#   val_indist.parquet            512 in-distribution prompts, evenly spaced along the stream, question-only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
TAG="${MODEL_TAG:-q3_4b}"
OUT="outputs/rl/data/$TAG"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-outputs/gen/$TAG/prompts_train_fixed.jsonl}"
VAL_PROMPTS="${VAL_PROMPTS:-outputs/gen/$TAG/prompts_indist_shuffled0.jsonl}"
mkdir -p "$OUT"
for guided in memory hint_memory peers hint_label hint_random peers_labeled none; do
  python -m training.sigma_rl.build_rl_data --prompts "$TRAIN_PROMPTS" --records data/mixed_train_big/train.jsonl \
      --out "$OUT/train_$guided.parquet" --guided "$guided" "$@"
done
python -m training.sigma_rl.build_rl_data --prompts "$TRAIN_PROMPTS" --records data/mixed_train_big/train.jsonl \
    --out "$OUT/train_v30_memory.parquet" --guided memory --verified_fraction 0.3 "$@"
python -m training.sigma_rl.build_rl_data --prompts "$VAL_PROMPTS" --records data/indist/test.jsonl \
    --out "$OUT/val_indist.parquet" --guided none --every 8 --limit 512
ls -la "$OUT"
