#!/bin/bash
# Evaluate a memory-judge checkpoint (or the frozen judge) on the in-distribution and OOD streams.
#   usage: scripts/run_memjudge_eval.sh <model_tag> <checkpoint_dir|none> <out_root> <gpu> [streams] [orders] [memory]
#   model_tag: q3_4b | q35_4b ; streams: "indist ood" ; orders: "fixed shuffled0" ; memory: on|off
# Run from the project root with PYTHONPATH=. and the right conda env active (rproj submit does both).
set -euo pipefail
tag=$1; ckpt=$2; out_root=$3; gpu=$4
streams=${5:-"indist ood"}; orders=${6:-"fixed shuffled0"}; memory=${7:-on}
declare -A HF=( [q3_0_6b]=Qwen3-0.6B [q3_4b]=Qwen3-4B [q3_8b]=Qwen3-8B [q35_4b]=Qwen3.5-4B [q35_9b]=Qwen3.5-9B )
declare -A DATA=( [train]=data/mixed_train_big/train.jsonl [indist]=data/indist/test.jsonl [ood]=data/ood/test.jsonl )
declare -A FEAT=( [train]=outputs/context_features/${tag}_big_ph/train [indist]=outputs/context_features/${tag}_indist_ph/ood [ood]=outputs/context_features/${tag}_ph/ood )
ckpt_arg=(); [ "$ckpt" != "none" ] && ckpt_arg=(--checkpoint "$ckpt")
for s in $streams; do
  for o in $orders; do
    out=$out_root/eval_${s}_${o}_mem${memory}
    if [ -f "$out/eval_metrics.json" ]; then echo "skip $out"; continue; fi
    CUDA_VISIBLE_DEVICES=$gpu python -m tests.experiments.common.evaluate_memory_judge \
      --config configs/symmetric_memory_candidate_yesno.yaml --central_model /mnt/data/peilin/HF_MODEL/${HF[$tag]} \
      "${ckpt_arg[@]}" --offline_data ${DATA[$s]} --features ${FEAT[$s]} --order $o --memory $memory --output $out
  done
done
