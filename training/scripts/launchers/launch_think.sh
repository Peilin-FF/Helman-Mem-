# thinking-mode run: the tilt arm with Qwen3 thinking ON and an 8,192-token answer budget, prompts rebuilt with the brevity
# reminder in the system prompt; same schedule as run 3/3b (40 steps, then the full epoch from that checkpoint with the
# reference reset and seed 2); then the four test conditions with thinking on for this model and the frozen model
set -u
[ -d training ] || cd /mnt/data/peilin/sigma-mem
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 12.8k-token sequences: logits [1, 12.8k, 152k] in fp32 are 7.8 GB, so one sequence per micro-batch
D=outputs/rl/data/q3_4b_6peer; P=outputs/gen/q3_4b; B=outputs/gen/q3_4b/base6_think8k; L=logs; mkdir -p $L $B
RESP=8192
COMMON="data.max_prompt_length=4608 data.enable_thinking=True data.max_response_length=$RESP actor_rollout_ref.rollout.n=8 actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 memory.guided_rollouts=0 trainer.save_freq=40 trainer.test_freq=20 trainer.total_epochs=1 trainer.resume_mode=disable data.shuffle=True"
TILT="data.attn_gamma=3.0 actor_rollout_ref.model.attn_bias=True actor_rollout_ref.rollout.attn_bias=True actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.attn_implementation=sdpa"
EV="python -m tests.experiments.common.evaluate_memory_generator --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B --engine vllm --gpu_memory_utilization 0.85 --thinking on --max_new_tokens $RESP"
last_ck () { ls -d outputs/rl/$1/hf/global_step_* | awk -F_ '{print $NF, $0}' | sort -n | tail -1 | cut -d" " -f2; }
ev () {   # gpu, checkpoint-or-base, stream, mode, extra args, tag
  local extra=""; [ $3 = ood ] && extra="--every 4"
  local ckarg=""; local outdir=$B/${3}6_$6; local name=base
  if [ "$2" != base ]; then ckarg="--checkpoint $2"; outdir=$2/eval_${3}6_$6; name=$(basename $(dirname $(dirname $2))); fi
  CUDA_VISIBLE_DEVICES=$1 $EV $ckarg --prompts $P/prompts_${3}6_think.jsonl --records data/${3}6/test.jsonl --mode $4 $5 $extra --output $outdir > $L/ev_think_${name}_${3}_$6.out 2>&1
}
tests () {
  ( ev 0 $1 indist peers "--attn_gamma 3" tilt ) & ( ev 1 $1 ood peers "--attn_gamma 3" tilt ) &
  ( ev 2 $1 indist peers "" peers ) & ( ev 3 $1 ood peers "" peers ) &
  ( ev 4 $1 indist solo "" solo ) & ( ev 5 $1 ood solo "" solo ) &
  ( ev 6 $1 indist peers "--attn_gamma 3 --swap_record" tilt_swapped ) & ( ev 7 $1 ood peers "--attn_gamma 3 --swap_record" tilt_swapped ) &
  wait
}
echo "=== prompts with the brevity reminder (system prompt of feedback_state.memory_generator): $(date)"
[ -f $P/prompts_train6_think.jsonl ] || CUDA_VISIBLE_DEVICES=0 python scripts/build_generation_prompts.py --model q3_4b --fit-stream train6 --stream train6 --order fixed --out $P/prompts_train6_think.jsonl > $L/prompts_train6_think.out 2>&1
[ -f $P/prompts_indist6_think.jsonl ] || CUDA_VISIBLE_DEVICES=1 python scripts/build_generation_prompts.py --model q3_4b --fit-stream train6 --stream indist6 --order shuffled0 --out $P/prompts_indist6_think.jsonl > $L/prompts_indist6_think.out 2>&1 &
[ -f $P/prompts_ood6_think.jsonl ] || CUDA_VISIBLE_DEVICES=2 python scripts/build_generation_prompts.py --model q3_4b --fit-stream train6 --stream ood6 --order shuffled0 --out $P/prompts_ood6_think.jsonl > $L/prompts_ood6_think.out 2>&1 &
wait
wc -l $P/prompts_train6_think.jsonl $P/prompts_indist6_think.jsonl $P/prompts_ood6_think.jsonl
grep -c "think briefly" $P/prompts_train6_think.jsonl
echo "=== RL data: $(date)"
[ -f $D/full_think.parquet ] || python -m training.sigma_rl.build_rl_data --prompts $P/prompts_train6_think.jsonl --records data/mixed_train_big6/train.jsonl --out $D/full_think.parquet --guided none --prompt_source peers --solo_fraction 0.25 2>&1 | grep -i 'build-rl-data\|error' | tail -2
[ -f $D/val_think.parquet ] || python -m training.sigma_rl.build_rl_data --prompts $P/prompts_indist6_think.jsonl --records data/indist6/test.jsonl --out $D/val_think.parquet --guided none --prompt_source peers --every 8 --limit 512 2>&1 | grep -i 'build-rl-data\|error' | tail -2
echo "=== think phase 1 (40 steps, seed 1) on GPUs 0-7: $(date)"
GPUS=0,1,2,3,4,5,6,7 EXP=think_tilt TRAIN=$D/full_think.parquet VAL=$D/val_think.parquet bash training/scripts/train_grpo.sh $COMMON $TILT trainer.total_training_steps=40 > $L/think_tilt.train.out 2>&1
echo "=== phase 1 finished (exit $?): $(date)"
CK0=outputs/rl/think_tilt/hf/global_step_40; ls $CK0/config.json
echo "=== think phase 2 (from step 40, seed 2, full epoch) on GPUs 0-7: $(date)"
GPUS=0,1,2,3,4,5,6,7 EXP=thinkb_tilt MODEL=$CK0 TRAIN=$D/full_think.parquet VAL=$D/val_think.parquet bash training/scripts/train_grpo.sh $COMMON $TILT +data.seed=2 > $L/thinkb_tilt.train.out 2>&1
echo "=== phase 2 finished (exit $?): $(date)"
CK=$(last_ck thinkb_tilt); echo "think checkpoint $CK"; [ -n "$CK" ] || { echo "no checkpoint: abort"; exit 1; }
echo "=== tests: thinking-mode arm: $(date)"; tests $CK
echo "=== tests: frozen model, thinking on, 8k, reminder prompts: $(date)"; tests base
for f in $CK/eval_*/eval_metrics.json $B/*/eval_metrics.json; do python -c "import json; m=json.load(open('$f')); print('$f'.replace('/eval_metrics.json',''), round(100*m['accuracy'],2), {k: round(100*v,1) for k,v in m['by_task'].items()})"; done
echo THINK_COMPLETE
