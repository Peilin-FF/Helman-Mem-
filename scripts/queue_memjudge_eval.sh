#!/bin/bash
# Wait for a memory-judge checkpoint, then evaluate it on the requested streams/orders on one GPU.
#   usage: scripts/queue_memjudge_eval.sh <model_tag> <checkpoint_dir> <out_root> <gpu> "<streams>" "<orders>" <memory on|off> [extra evaluator args...]
set -euo pipefail
tag=$1; ckpt=$2; out_root=$3; gpu=$4; streams=$5; orders=$6; memory=$7; shift 7
extra=("$@")
declare -A HF=( [q3_0_6b]=Qwen3-0.6B [q3_4b]=Qwen3-4B [q3_8b]=Qwen3-8B [q35_4b]=Qwen3.5-4B [q35_9b]=Qwen3.5-9B )
declare -A DATA=( [train]=data/mixed_train_big/train.jsonl [indist]=data/indist/test.jsonl [ood]=data/ood/test.jsonl )
declare -A FEAT=( [train]=outputs/context_features/${tag}_big_ph/train [indist]=outputs/context_features/${tag}_indist_ph/ood [ood]=outputs/context_features/${tag}_ph/ood )
if [ "$ckpt" != "none" ]; then
  until [ -f "$ckpt/memory_judge.pt" ]; do echo "waiting for $ckpt/memory_judge.pt ($(date))"; sleep 120; done
  sleep 30
fi
ckpt_arg=(); [ "$ckpt" != "none" ] && ckpt_arg=(--checkpoint "$ckpt")
suffix=""; for a in "${extra[@]}"; do suffix+="_$(echo "$a" | tr -d ' -')"; done
for s in $streams; do
  for o in $orders; do
    out=$out_root/eval_${s}_${o}_mem${memory}${suffix}
    if [ -f "$out/eval_metrics.json" ]; then echo "skip $out"; continue; fi
    echo "=== $out ($(date))"
    CUDA_VISIBLE_DEVICES=$gpu python -m tests.experiments.common.evaluate_memory_judge \
      --config configs/symmetric_memory_candidate_yesno.yaml --central_model /mnt/data/peilin/HF_MODEL/${HF[$tag]} \
      "${ckpt_arg[@]}" --offline_data ${DATA[$s]} --features ${FEAT[$s]} --order $o --memory $memory --output $out "${extra[@]}"
  done
done
echo "QUEUE_DONE $(date)"
