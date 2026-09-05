#!/usr/bin/env bash
# Evaluate an HF checkpoint (question-only, greedy) with the existing evaluator on the in-distribution and OOD streams.
#   rproj submit 'GPU=1 CKPT=outputs/rl/q3_4b_grpo_hint/hf/global_step_40 bash training/scripts/eval_hf.sh'
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${CKPT:?set CKPT=<HF checkpoint dir>}"
GPU="${GPU:-0}"
TAG="${MODEL_TAG:-q3_4b}"
MODEL="${MODEL:-/mnt/data/peilin/HF_MODEL/Qwen3-4B}"
MAXN="${MAX_EXAMPLES:-}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" FEEDBACK_CODE_EXEC_ALLOW=1 CUDA_VISIBLE_DEVICES="$GPU"
for split in indist ood; do
  python -m tests.experiments.common.evaluate_memory_generator --central_model "$MODEL" --checkpoint "$CKPT" \
      --prompts "outputs/gen/$TAG/prompts_${split}_shuffled0.jsonl" --records "data/$split/test.jsonl" --mode solo \
      --output "$CKPT/eval_${split}_solo" --batch_size 16 ${MAXN:+--max_examples $MAXN}
done
