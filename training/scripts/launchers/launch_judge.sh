# Training the judgement (docs section 22). One continuous epoch on the 4B (277 steps of 64 events from the frozen
# model, KL 0.01 to the frozen model throughout; no 40-step phase and no reference reset), 8 GPUs.
#   A2  method A, gamma = 0: the reply opens with a 'Trust: a > b > ...' ranking of the peers, rewarded
#       0.5(1 + Spearman) against the record's estimate on non-flat records (lambda 0.5) plus the verifier reward;
#       the record is a TARGET only (strict distillation; measures how much of it content recovers)
#   B0  method B stage 0, gamma = 0: label-free evidence lines (outputs/evidence/*.json: sample tests, arithmetic
#       steps, span in passage) added to the prompt after the peers; no verdict line
#   A1  method A with gamma = 3 in training: the record as target and input (upper bound)
# Tests per arm on the in-distribution stream and the FULL OOD stream, then the attention measurement (A arms).
# Usage: bash launch_judge.sh [A2|B0|A1|all]     (all = A2, B0, A1 in that order)
set -u
[ -d training ] || cd /mnt/data/peilin/sigma-mem
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
D=outputs/rl/data/q3_4b_6peer; P=outputs/gen/q3_4b; L=logs; mkdir -p $L
COMMON="data.max_prompt_length=4608 data.enable_thinking=False data.max_response_length=768 actor_rollout_ref.rollout.n=8 actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 memory.guided_rollouts=0 memory.verdict_flat=0.1 trainer.save_freq=40 trainer.test_freq=20 trainer.total_epochs=1 trainer.resume_mode=disable data.shuffle=True"
TILT="data.attn_gamma=3.0 actor_rollout_ref.model.attn_bias=True actor_rollout_ref.rollout.attn_bias=True actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.attn_implementation=sdpa"
EV="python -m tests.experiments.common.evaluate_memory_generator --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B --engine vllm --gpu_memory_utilization 0.85 --thinking off --max_new_tokens 768"
last_ck () { ls -d outputs/rl/$1/hf/global_step_* | awk -F_ '{print $NF, $0}' | sort -n | tail -1 | cut -d" " -f2; }
ev () {   # gpu, checkpoint, stream(indist|oodfull), mode, extra, tag
  local st=$3; local pf=$3; [ $3 = oodfull ] && pf=ood
  local outdir=$2/eval_${st}6_$6; local name=$(basename $(dirname $(dirname $2)))
  [ -f $outdir/eval_metrics.json ] && { echo "skip $name/$st/$6"; return 0; }
  CUDA_VISIBLE_DEVICES=$1 $EV --checkpoint $2 --prompts $P/prompts_${pf}6_probe.jsonl --records data/${pf}6/test.jsonl --mode $4 $5 --output $outdir > $L/evj_${name}_${st}_$6.out 2>&1
}
attention () {   # checkpoint, name: attention over the peer blocks at gamma 0 and 3, both streams, on GPU 2
  for s in indist ood; do
    CUDA_VISIBLE_DEVICES=2 python scripts/attention_mass.py --model $1 --name $2 --stream $s --prompts $P/prompts_${s}6_probe.jsonl --records data/${s}6/test.jsonl --n 300 --gammas 0,3 --out outputs/address/attention_mass > $L/attn_mass_$2_$s.out 2>&1
  done
}
tests_A () {   # checkpoint, name: verdict line in every peers condition, plus peers without the instruction
  ( ev 0 $1 indist peers "--verdict --attn_gamma 3" tilt ) & ( ev 1 $1 oodfull peers "--verdict --attn_gamma 3" tilt ) &
  ( ev 2 $1 indist peers "--verdict" peers ) & ( ev 3 $1 oodfull peers "--verdict" peers ) &
  ( ev 4 $1 indist solo "" solo ) & ( ev 5 $1 oodfull solo "" solo ) &
  ( ev 6 $1 indist peers "--verdict --attn_gamma 3 --swap_record" tilt_swapped ) & ( ev 7 $1 oodfull peers "--verdict --attn_gamma 3 --swap_record" tilt_swapped ) &
  wait
  ( ev 0 $1 indist peers "" peers_noverdict ) & ( ev 1 $1 oodfull peers "" peers_noverdict ) & ( attention $1 $2 ) &
  wait
}
tests_B () {   # checkpoint, name: the same label-free evidence lines handed over at test time
  ( ev 0 $1 indist peers "--evidence outputs/evidence/indist6.json" evid ) & ( ev 1 $1 oodfull peers "--evidence outputs/evidence/ood6.json" evid ) &
  ( ev 2 $1 indist peers "" peers ) & ( ev 3 $1 oodfull peers "" peers ) &
  ( ev 4 $1 indist solo "" solo ) & ( ev 5 $1 oodfull solo "" solo ) &
  ( ev 6 $1 indist peers "--evidence outputs/evidence/indist6.json --attn_gamma 3" evid_tilt ) & ( ev 7 $1 oodfull peers "--evidence outputs/evidence/ood6.json --attn_gamma 3" evid_tilt ) &
  wait
}
build_evidence_data () {
  [ -f $D/full_peers_evidence.parquet ] || python -m training.sigma_rl.build_rl_data --prompts $P/prompts_train6_fixed.jsonl --records data/mixed_train_big6/train.jsonl --out $D/full_peers_evidence.parquet --guided none --prompt_source peers --solo_fraction 0.25 --evidence outputs/evidence/train6.json 2>&1 | grep -i "build-rl-data\|error"
  [ -f $D/val_peers_evidence.parquet ] || python -m training.sigma_rl.build_rl_data --prompts $P/prompts_indist6_shuffled0.jsonl --records data/indist6/test.jsonl --out $D/val_peers_evidence.parquet --guided none --prompt_source peers --every 8 --limit 512 --evidence outputs/evidence/indist6.json 2>&1 | grep -i "build-rl-data\|error"
  python -c "
import pandas as pd
for f in ['$D/full_peers_evidence.parquet', '$D/val_peers_evidence.parquet']:
    df = pd.read_parquet(f); n = int(df['prompt'].astype(str).str.contains('Evidence \\\\(automatic').sum()); print(f, len(df), 'rows;', n, 'with an evidence block')
"
}
train_arm () {   # name, extra overrides, train parquet, val parquet: ONE continuous epoch (277 steps of 64 events) from the
                 # frozen model, reference fixed at the frozen model, no restart at step 40 (user, 2026-09-11)
  echo "=== $1 (one epoch, seed 1): $(date)"
  GPUS=0,1,2,3,4,5,6,7 EXP=$1 TRAIN=$3 VAL=$4 bash training/scripts/train_grpo.sh $COMMON $2 > $L/$1.train.out 2>&1
  echo "=== $1 finished (exit $?): $(date)"
  ls outputs/rl/$1/hf/global_step_*/config.json > /dev/null 2>&1 || { echo "no checkpoint for $1"; return 1; }
}
what=${1:-all}
if [ $what = A2 ] || [ $what = all ]; then
  train_arm judgeA2 "memory.verdict_lambda=0.5" $D/full_peers_verdict.parquet $D/val_peers_verdict.parquet && { CK=$(last_ck judgeA2); echo "=== tests: judgeA2 $CK: $(date)"; tests_A $CK judgeA2; }
fi
if [ $what = B0 ] || [ $what = all ]; then
  build_evidence_data
  train_arm judgeB0 "memory.verdict_lambda=0.0" $D/full_peers_evidence.parquet $D/val_peers_evidence.parquet && { CK=$(last_ck judgeB0); echo "=== tests: judgeB0 $CK: $(date)"; tests_B $CK judgeB0; }
fi
if [ $what = A1 ] || [ $what = all ]; then
  train_arm judgeA1 "memory.verdict_lambda=0.5 $TILT" $D/full_peers_verdict.parquet $D/val_peers_verdict.parquet && { CK=$(last_ck judgeA1); echo "=== tests: judgeA1 $CK: $(date)"; tests_A $CK judgeA1; }
fi
for f in outputs/rl/judgeA2/hf/global_step_*/eval_*/eval_metrics.json outputs/rl/judgeB0/hf/global_step_*/eval_*/eval_metrics.json outputs/rl/judgeA1/hf/global_step_*/eval_*/eval_metrics.json; do
  [ -f "$f" ] && python -c "
import json; m=json.load(open('$f')); v=m.get('verdict',{})
print('%-60s %6.2f  %s  verdict: auc=%s fav=%s rho_rec=%s' % ('$f'.replace('/eval_metrics.json','').replace('outputs/rl/',''), 100*m['accuracy'], {k: round(100*x,1) for k,x in m['by_task'].items()}, v.get('auc_vs_correct'), v.get('favourite_right'), v.get('spearman_with_record')))"
done
echo JUDGE_COMPLETE
