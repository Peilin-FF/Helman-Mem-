#!/usr/bin/env bash
# Delete the weights of intermediate checkpoints that will not be used again (evaluations live in outputs/eval/, untouched).
#   bash training/scripts/prune_checkpoints.sh <run> <step> [<step> ...]
#   e.g. bash training/scripts/prune_checkpoints.sh run3b_tilt 80 120 160
# Only *.safetensors and the index are removed from outputs/train/<run>/hf/global_step_<step>; config and tokenizer stay.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$1"; shift
for s in "$@"; do
  d="$ROOT/outputs/train/$RUN/hf/global_step_$s"
  [ -d "$d" ] || { echo "skip $d (missing)"; continue; }
  n=$(find "$d" -maxdepth 1 \( -name '*.safetensors' -o -name 'model.safetensors.index.json' \) -print -delete | wc -l)
  echo "pruned $d: $n weight file(s) removed"
done
df -h "$ROOT/outputs" | tail -1
