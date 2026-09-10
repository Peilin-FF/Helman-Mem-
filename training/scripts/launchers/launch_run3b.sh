# run 3: thinking OFF. tilt arm only (KL 0.01, 768-token answers), then the four test conditions for both arms and the frozen model
set -u
[ -d training ] || cd /mnt/data/peilin/sigma-mem
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
D=outputs/rl/data/q3_4b_6peer; P=outputs/gen/q3_4b; B=outputs/gen/q3_4b/base6_nothink
COMMON="data.max_prompt_length=4608 data.enable_thinking=False data.max_response_length=768 actor_rollout_ref.rollout.n=8 actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 memory.guided_rollouts=0 trainer.save_freq=40 trainer.test_freq=20 trainer.total_epochs=1 trainer.resume_mode=disable data.shuffle=True +data.seed=2"
TILT="data.attn_gamma=3.0 actor_rollout_ref.model.attn_bias=True actor_rollout_ref.rollout.attn_bias=True actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.attn_implementation=sdpa"
EV="python -m tests.experiments.common.evaluate_memory_generator --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B --engine vllm --gpu_memory_utilization 0.85 --thinking off --max_new_tokens 768"
last_ck () { ls -d outputs/rl/$1/hf/global_step_* | sort -t_ -k3 -n | tail -1; }
ev () {   # gpu, checkpoint-or-base, stream, mode, extra args, tag
  local extra=""   # OOD is tested on the whole stream (17,403 events), never subsampled
  local ckarg=""; local outdir=$B/${3}6_$6; local name=base
  if [ "$2" != base ]; then ckarg="--checkpoint $2"; outdir=$2/eval_${3}6_$6; name=$(basename $(dirname $(dirname $2))); fi
  CUDA_VISIBLE_DEVICES=$1 $EV $ckarg --prompts $P/prompts_${3}6_probe.jsonl --records data/${3}6/test.jsonl --mode $4 $5 $extra --output $outdir > /mnt/data/peilin/ev3b_${name}_${3}_$6.out 2>&1
}
tests () {   # checkpoint-or-base: 8 conditions on 8 GPUs
  ( ev 0 $1 indist peers "--attn_gamma 3" tilt ) & ( ev 1 $1 ood peers "--attn_gamma 3" tilt ) &
  ( ev 2 $1 indist peers "" peers ) & ( ev 3 $1 ood peers "" peers ) &
  ( ev 4 $1 indist solo "" solo ) & ( ev 5 $1 ood solo "" solo ) &
  ( ev 6 $1 indist peers "--attn_gamma 3 --swap_record" tilt_swapped ) & ( ev 7 $1 ood peers "--attn_gamma 3 --swap_record" tilt_swapped ) &
  wait
}
echo "=== run3b tilt arm (thinking off, from step 40, KL 0.01) on GPUs 0-7: $(date)"
GPUS=0,1,2,3,4,5,6,7 EXP=run3b_tilt MODEL=outputs/rl/run3_tilt/hf/global_step_40 TRAIN=$D/full_peers.parquet VAL=$D/val_peers.parquet bash training/scripts/train_grpo.sh $COMMON $TILT > /mnt/data/peilin/run3b_tilt.train.out 2>&1
echo "=== run3 tilt arm finished (exit $?): $(date)"
CK1=$(last_ck run3b_tilt); echo "checkpoint $CK1"
echo "=== tests: tilt arm"; tests $CK1
echo "=== tests: frozen model"; tests base
for f in $B/*/eval_metrics.json $CK1/eval_*/eval_metrics.json; do python -c "import json; m=json.load(open('$f')); print('$f'.replace('/eval_metrics.json',''), round(100*m['accuracy'],2), {k: round(100*v,1) for k,v in m['by_task'].items()})"; done
echo RUN3B_COMPLETE
