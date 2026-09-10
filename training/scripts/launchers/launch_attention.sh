# Attention mass on the peer blocks, with and without the tilt, for the four thinking-off models:
# 300 prompts per stream with a non-flat record and disagreeing peers, one forward pass each, no generation.
set -u
[ -d training ] || cd /mnt/data/peilin/sigma-mem
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1
P=outputs/gen/q3_4b; L=logs; O=outputs/address/attention_mass; mkdir -p $L $O
run () {   # gpu, name, model dir
  for s in indist ood; do
    CUDA_VISIBLE_DEVICES=$1 python scripts/attention_mass.py --model $3 --name $2 --stream $s \
      --prompts $P/prompts_${s}6_probe.jsonl --records data/${s}6/test.jsonl --n 300 --gammas 0,3 --out $O \
      > $L/attn_mass_$2_$s.out 2>&1
  done
  echo "done $2"
}
run 0 frozen        /mnt/data/peilin/HF_MODEL/Qwen3-4B &
run 1 ours          outputs/rl/run3b_tilt/hf/global_step_276 &
run 2 control       outputs/rl/ctrl3b_peers/hf/global_step_276 &
run 3 question-only outputs/rl/solo3b_q/hf/global_step_276 &
wait
python scripts/attention_mass_summary.py --dir $O --out $O/summary.json
echo ATTN_COMPLETE
