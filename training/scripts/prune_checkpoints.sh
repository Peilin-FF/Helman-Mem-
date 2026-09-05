#!/usr/bin/env bash
# Delete the weights of checkpoints that have been evaluated and will not be used again, keeping their evaluation outputs.
#   bash training/scripts/prune_checkpoints.sh <run dir under outputs/rl> <step> [<step> ...]
#   e.g. bash training/scripts/prune_checkpoints.sh q3_4b_grpo_peers 70 140 210
# Only *.safetensors and the index are removed from hf/global_step_<step>; eval_* directories, config and tokenizer stay.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$1"; shift
for s in "$@"; do
  d="$ROOT/outputs/rl/$RUN/hf/global_step_$s"
  [ -d "$d" ] || { echo "skip $d (missing)"; continue; }
  n=$(find "$d" -maxdepth 1 \( -name '*.safetensors' -o -name 'model.safetensors.index.json' \) -print -delete | wc -l)
  echo "pruned $d: $n weight file(s) removed"
done
df -h "$ROOT/outputs" | tail -1
