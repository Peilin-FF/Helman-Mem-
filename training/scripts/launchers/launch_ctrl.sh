# control for run 3/3b: same schedule and settings, question + six peer answers, NO memory steering anywhere
set -u
[ -d training ] || cd /mnt/data/peilin/sigma-mem
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
D=outputs/rl/data/q3_4b_6peer; P=outputs/gen/q3_4b; B=outputs/gen/q3_4b/base6_nothink
COMMON="data.max_prompt_length=4608 data.enable_thinking=False data.max_response_length=768 actor_rollout_ref.rollout.n=8 actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 memory.guided_rollouts=0 trainer.save_freq=40 trainer.test_freq=20 trainer.total_epochs=1 trainer.resume_mode=disable data.shuffle=True"
EV="python -m tests.experiments.common.evaluate_memory_generator --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B --engine vllm --gpu_memory_utilization 0.85 --thinking off --max_new_tokens 768"
last_ck () { ls -d outputs/rl/$1/hf/global_step_* | awk -F_ '{print $NF, $0}' | sort -n | tail -1 | cut -d" " -f2; }
ev () {
  local extra=""   # OOD is tested on the whole stream (17,403 events), never subsampled
  CUDA_VISIBLE_DEVICES=$1 $EV --checkpoint $2 --prompts $P/prompts_${3}6_probe.jsonl --records data/${3}6/test.jsonl --mode $4 $5 $extra --output $2/eval_${3}6_$6 > /mnt/data/peilin/evc_$(basename $(dirname $(dirname $2)))_${3}_$6.out 2>&1
}
tests () {
  ( ev 0 $1 indist peers "--attn_gamma 3" tilt ) & ( ev 1 $1 ood peers "--attn_gamma 3" tilt ) &
  ( ev 2 $1 indist peers "" peers ) & ( ev 3 $1 ood peers "" peers ) &
  ( ev 4 $1 indist solo "" solo ) & ( ev 5 $1 ood solo "" solo ) &
  ( ev 6 $1 indist peers "--attn_gamma 3 --swap_record" tilt_swapped ) & ( ev 7 $1 ood peers "--attn_gamma 3 --swap_record" tilt_swapped ) &
  wait
}
CKF=outputs/rl/run3b_tilt/hf/global_step_276
echo "=== tests: run 3b final checkpoint $CKF: $(date)"; tests $CKF
for f in $CKF/eval_*/eval_metrics.json; do python -c "import json; m=json.load(open('$f')); print('$f'.replace('/eval_metrics.json',''), round(100*m['accuracy'],2), {k: round(100*v,1) for k,v in m['by_task'].items()})"; done
echo "=== control phase 1 (40 steps, seed 1) on GPUs 0-7: $(date)"
GPUS=0,1,2,3,4,5,6,7 EXP=ctrl3_peers TRAIN=$D/full_peers.parquet VAL=$D/val_peers.parquet bash training/scripts/train_grpo.sh $COMMON trainer.total_training_steps=40 > /mnt/data/peilin/ctrl3_peers.train.out 2>&1
echo "=== phase 1 finished (exit $?): $(date)"
CK0=outputs/rl/ctrl3_peers/hf/global_step_40; ls $CK0/config.json
echo "=== control phase 2 (from step 40, seed 2, full epoch) on GPUs 0-7: $(date)"
GPUS=0,1,2,3,4,5,6,7 EXP=ctrl3b_peers MODEL=$CK0 TRAIN=$D/full_peers.parquet VAL=$D/val_peers.parquet bash training/scripts/train_grpo.sh $COMMON +data.seed=2 > /mnt/data/peilin/ctrl3b_peers.train.out 2>&1
echo "=== phase 2 finished (exit $?): $(date)"
CK=$(last_ck ctrl3b_peers); echo "control checkpoint $CK"
echo "=== tests: control"; tests $CK
for f in $CK/eval_*/eval_metrics.json; do python -c "import json; m=json.load(open('$f')); print('$f'.replace('/eval_metrics.json',''), round(100*m['accuracy'],2), {k: round(100*v,1) for k,v in m['by_task'].items()})"; done
echo CTRL_COMPLETE
