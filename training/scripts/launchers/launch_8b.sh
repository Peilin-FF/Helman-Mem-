# Qwen3-8B as the central model: the tilt arm and the no-memory control with the identical schedule as the 4B
# runs (40 steps from the frozen model, then the full epoch from that checkpoint with the reference reset and
# seed 2), then the four test conditions on the in-distribution stream and the FULL OOD stream for both arms
# and for the frozen 8B model. The memory is unchanged: the record and its addresses come from the frozen
# Qwen3-4B judge (the same prompt files and parquets as the 4B runs), so only the central model changes.
# Usage: bash launch_8b.sh [tilt|ctrl|tests|all]   (default all)
set -u
[ -d training ] || cd /mnt/data/peilin/sigma-mem
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
M8=/mnt/data/peilin/HF_MODEL/Qwen3-8B
D=outputs/rl/data/q3_4b_6peer; P=outputs/gen/q3_4b; B=outputs/gen/q3_8b/base6_nothink; L=logs; mkdir -p $L $B
# 8B: half the micro-batch of the 4B runs (2 / 4 / 4), everything else identical
COMMON="data.max_prompt_length=4608 data.enable_thinking=False data.max_response_length=768 actor_rollout_ref.rollout.n=8 actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 memory.guided_rollouts=0 trainer.save_freq=40 trainer.test_freq=20 trainer.total_epochs=1 trainer.resume_mode=disable data.shuffle=True"
TILT="data.attn_gamma=3.0 actor_rollout_ref.model.attn_bias=True actor_rollout_ref.rollout.attn_bias=True actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.attn_implementation=sdpa"
EV="python -m tests.experiments.common.evaluate_memory_generator --central_model $M8 --engine vllm --gpu_memory_utilization 0.85 --thinking off --max_new_tokens 768"
last_ck () { ls -d outputs/rl/$1/hf/global_step_* | awk -F_ '{print $NF, $0}' | sort -n | tail -1 | cut -d" " -f2; }
ev () {   # gpu, checkpoint-or-base, stream(indist|oodfull), mode, extra, tag
  local st=$3; local pf=$3; [ $3 = oodfull ] && pf=ood
  local ckarg=""; local outdir=$B/${st}6_$6; local name=base8b
  if [ "$2" != base ]; then ckarg="--checkpoint $2"; outdir=$2/eval_${st}6_$6; name=$(basename $(dirname $(dirname $2))); fi
  [ -f $outdir/eval_metrics.json ] && { echo "skip $name/$st/$6"; return 0; }
  CUDA_VISIBLE_DEVICES=$1 $EV $ckarg --prompts $P/prompts_${pf}6_probe.jsonl --records data/${pf}6/test.jsonl --mode $4 $5 --output $outdir > $L/ev8b_${name}_${st}_$6.out 2>&1
}
tests () {   # 8 conditions on 8 GPUs; OOD is the whole stream
  ( ev 0 $1 indist peers "--attn_gamma 3" tilt ) & ( ev 1 $1 oodfull peers "--attn_gamma 3" tilt ) &
  ( ev 2 $1 indist peers "" peers ) & ( ev 3 $1 oodfull peers "" peers ) &
  ( ev 4 $1 indist solo "" solo ) & ( ev 5 $1 oodfull solo "" solo ) &
  ( ev 6 $1 indist peers "--attn_gamma 3 --swap_record" tilt_swapped ) & ( ev 7 $1 oodfull peers "--attn_gamma 3 --swap_record" tilt_swapped ) &
  wait
}
train_arm () {   # name, extra overrides
  echo "=== $1 phase 1 (40 steps, seed 1): $(date)"
  GPUS=0,1,2,3,4,5,6,7 EXP=${1}_p1 MODEL=$M8 TRAIN=$D/full_peers.parquet VAL=$D/val_peers.parquet bash training/scripts/train_grpo.sh $COMMON $2 trainer.total_training_steps=40 > $L/${1}_p1.train.out 2>&1
  echo "=== phase 1 finished (exit $?): $(date)"
  local CK0=outputs/rl/${1}_p1/hf/global_step_40; ls $CK0/config.json || { echo "no phase-1 checkpoint for $1"; return 1; }
  echo "=== $1 phase 2 (from step 40, seed 2, full epoch): $(date)"
  GPUS=0,1,2,3,4,5,6,7 EXP=$1 MODEL=$CK0 TRAIN=$D/full_peers.parquet VAL=$D/val_peers.parquet bash training/scripts/train_grpo.sh $COMMON $2 +data.seed=2 > $L/$1.train.out 2>&1
  echo "=== phase 2 finished (exit $?): $(date)"
}
what=${1:-all}
if [ $what = tilt ] || [ $what = all ]; then train_arm run8b_tilt "$TILT"; CK=$(last_ck run8b_tilt); echo "=== tests: run8b_tilt $CK: $(date)"; tests $CK; fi
if [ $what = ctrl ] || [ $what = all ]; then train_arm ctrl8b_peers ""; CK=$(last_ck ctrl8b_peers); echo "=== tests: ctrl8b_peers $CK: $(date)"; tests $CK; fi
if [ $what = tests ] || [ $what = all ]; then echo "=== tests: frozen Qwen3-8B: $(date)"; tests base; fi
for f in $B/*/eval_metrics.json outputs/rl/run8b_tilt/hf/global_step_*/eval_*/eval_metrics.json outputs/rl/ctrl8b_peers/hf/global_step_*/eval_*/eval_metrics.json; do [ -f "$f" ] && python -c "import json; m=json.load(open('$f')); print('%-70s %6.2f %s' % ('$f'.replace('/eval_metrics.json',''), 100*m['accuracy'], {k: round(100*v,1) for k,v in m['by_task'].items()}))"; done
echo RUN8B_COMPLETE
