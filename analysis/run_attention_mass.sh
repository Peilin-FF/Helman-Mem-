#!/usr/bin/env bash
# Where the central model's attention goes among the six peer blocks, with and without the tilt (design log section 21):
# the frozen model and the three trained arms, 300 prompts per stream with a non-flat record and disagreeing peers.
#   bash analysis/run_attention_mass.sh /models/Qwen3-4B        (4 GPUs: 0-3; results in outputs/analysis/attention_mass)
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
BASE=${1:?the frozen model directory}
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1
R=outputs/record/q3_4b; O=outputs/analysis/attention_mass; L=logs/analysis; mkdir -p $L $O
run () {   # gpu name model
  for s in indist ood; do
    CUDA_VISIBLE_DEVICES=$1 python -m analysis.attention_mass --model $3 --name $2 --stream $s \
      --prompts $R/${s}6/shuffled0.fit-train6.jsonl --records data/${s}6/test.jsonl --n 300 --gammas 0,3 --out $O \
      > $L/attn_mass_$2_$s.out 2>&1
  done
}
last () { ls -d outputs/train/$1/hf/global_step_* | sort -t_ -k3 -n | tail -1; }
run 0 frozen        "$BASE" &
run 1 ours          "$(last run3b_tilt)" &
run 2 control       "$(last ctrl3b_peers)" &
run 3 question-only "$(last solo3b_q)" &
wait
python -m analysis.attention_mass_summary --dir $O --out $O/summary.json
