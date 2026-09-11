# Method A (docs section 22): distil the record into the model's own verdict. The reply opens with a
# 'Trust: a > b > ...' ranking of the peers, rewarded 0.5(1 + Spearman) against the record's estimate on
# non-flat records (lambda 0.5), plus the verifier reward for the answer. Same schedule as run 3b on the 4B.
#   A2  gamma = 0 in training: the record is a TARGET only (strict distillation; measures recoverability)
#   A1  gamma = 3 in training: the record is target and input (upper bound)
# Tests for each arm: peers+tilt / peers / question only / swapped tilt, all with the verdict line, plus
# peers WITHOUT the verdict instruction (accuracy comparable to the control); in-dist and the full OOD stream;
# then the attention measurement at gamma 0 and 3.   Usage: bash launch_judge.sh [A2|A1|all]
set -u
[ -d training ] || cd /mnt/data/peilin/sigma-mem
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
D=outputs/rl/data/q3_4b_6peer; P=outputs/gen/q3_4b; L=logs; mkdir -p $L
COMMON="data.max_prompt_length=4608 data.enable_thinking=False data.max_response_length=768 actor_rollout_ref.rollout.n=8 actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 memory.guided_rollouts=0 memory.verdict_lambda=0.5 memory.verdict_flat=0.1 trainer.save_freq=40 trainer.test_freq=20 trainer.total_epochs=1 trainer.resume_mode=disable data.shuffle=True"
TILT="data.attn_gamma=3.0 actor_rollout_ref.model.attn_bias=True actor_rollout_ref.rollout.attn_bias=True actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.attn_implementation=sdpa"
EV="python -m tests.experiments.common.evaluate_memory_generator --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B --engine vllm --gpu_memory_utilization 0.85 --thinking off --max_new_tokens 768"
last_ck () { ls -d outputs/rl/$1/hf/global_step_* | awk -F_ '{print $NF, $0}' | sort -n | tail -1 | cut -d" " -f2; }
ev () {   # gpu, checkpoint, stream(indist|oodfull), mode, extra, tag
  local st=$3; local pf=$3; [ $3 = oodfull ] && pf=ood
  local outdir=$2/eval_${st}6_$6; local name=$(basename $(dirname $(dirname $2)))
  [ -f $outdir/eval_metrics.json ] && { echo "skip $name/$st/$6"; return 0; }
  CUDA_VISIBLE_DEVICES=$1 $EV --checkpoint $2 --prompts $P/prompts_${pf}6_probe.jsonl --records data/${pf}6/test.jsonl --mode $4 $5 --output $outdir > $L/evj_${name}_${st}_$6.out 2>&1
}
tests () {
  ( ev 0 $1 indist peers "--verdict --attn_gamma 3" tilt ) & ( ev 1 $1 oodfull peers "--verdict --attn_gamma 3" tilt ) &
  ( ev 2 $1 indist peers "--verdict" peers ) & ( ev 3 $1 oodfull peers "--verdict" peers ) &
  ( ev 4 $1 indist solo "" solo ) & ( ev 5 $1 oodfull solo "" solo ) &
  ( ev 6 $1 indist peers "--verdict --attn_gamma 3 --swap_record" tilt_swapped ) & ( ev 7 $1 oodfull peers "--verdict --attn_gamma 3 --swap_record" tilt_swapped ) &
  wait
  ( ev 0 $1 indist peers "" peers_noverdict ) & ( ev 1 $1 oodfull peers "" peers_noverdict ) &
  ( for s in indist ood; do CUDA_VISIBLE_DEVICES=2 python scripts/attention_mass.py --model $1 --name $2 --stream $s --prompts $P/prompts_${s}6_probe.jsonl --records data/${s}6/test.jsonl --n 300 --gammas 0,3 --out outputs/address/attention_mass > $L/attn_mass_$2_$s.out 2>&1; done ) &
  wait
}
train_arm () {   # name, extra overrides
  echo "=== $1 phase 1 (40 steps, seed 1): $(date)"
  GPUS=0,1,2,3,4,5,6,7 EXP=${1}_p1 TRAIN=$D/full_peers_verdict.parquet VAL=$D/val_peers_verdict.parquet bash training/scripts/train_grpo.sh $COMMON $2 trainer.total_training_steps=40 > $L/${1}_p1.train.out 2>&1
  echo "=== phase 1 finished (exit $?): $(date)"
  local CK0=outputs/rl/${1}_p1/hf/global_step_40; ls $CK0/config.json || { echo "no phase-1 checkpoint for $1"; return 1; }
  echo "=== $1 phase 2 (from step 40, seed 2, full epoch): $(date)"
  GPUS=0,1,2,3,4,5,6,7 EXP=$1 MODEL=$CK0 TRAIN=$D/full_peers_verdict.parquet VAL=$D/val_peers_verdict.parquet bash training/scripts/train_grpo.sh $COMMON $2 +data.seed=2 > $L/$1.train.out 2>&1
  echo "=== phase 2 finished (exit $?): $(date)"
}
what=${1:-all}
if [ $what = A2 ] || [ $what = all ]; then train_arm judgeA2 "" && { CK=$(last_ck judgeA2); echo "=== tests: judgeA2 $CK: $(date)"; tests $CK judgeA2; }; fi
if [ $what = A1 ] || [ $what = all ]; then train_arm judgeA1 "$TILT" && { CK=$(last_ck judgeA1); echo "=== tests: judgeA1 $CK: $(date)"; tests $CK judgeA1; }; fi
for f in outputs/rl/judgeA2/hf/global_step_*/eval_*/eval_metrics.json outputs/rl/judgeA1/hf/global_step_*/eval_*/eval_metrics.json; do [ -f "$f" ] && python -c "import json; m=json.load(open('$f')); v=m.get('verdict',{}); print('%-62s %6.2f  %s  verdict: auc=%s fav=%s rho_rec=%s' % ('$f'.replace('/eval_metrics.json','').replace('outputs/rl/',''), 100*m['accuracy'], {k: round(100*x,1) for k,x in m['by_task'].items()}, v.get('auc_vs_correct'), v.get('favourite_right'), v.get('spearman_with_record')))"; done
echo JUDGE_COMPLETE
