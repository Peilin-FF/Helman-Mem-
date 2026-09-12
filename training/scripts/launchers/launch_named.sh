# The memory as a teacher (2026-09-13): stable peer identities + on-policy self-distillation from the tilted self.
#   NSD    rollouts and the policy update run WITHOUT the tilt (gamma 0); the KL term pulls the student toward the
#          current policy's log-probs computed WITH the tilt (gamma 3) on the same samples (memory.self_distill=True,
#          kl_loss_coef 0.3, no frozen reference). The record is never an input; the test-time model needs no module.
#   NCTRL  the plain control recipe (KL 0.01 to the frozen model, gamma 0, flash-attn) on the same named prompts.
# Both use the two-phase schedule of run 3b / ctrl3b (40 steps seed 1, then a full epoch seed 2 from step 40).
# Prompts: the six peers carry fixed names (Peer k (Ada|Ben|Cara|Dev|Eli|Faye) by canonical peer id) in every event; the
# prompt order still varies per event. Files: outputs/gen/q3_4b/prompts_{train6_fixed,indist6_probe,ood6_probe}_named.jsonl.
# Usage: bash training/scripts/launchers/launch_named.sh <prompts|smoke|nsd|nctrl|tests|all>
set -u
[ -d training ] || cd /mnt/data/peilin/sigma-mem
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 WANDB_MODE=offline
WHAT=${1:-all}
D=outputs/rl/data/q3_4b_6peer; P=outputs/gen/q3_4b; B=outputs/gen/q3_4b/base6_nothink; L=/mnt/data/peilin
COMMON="data.max_prompt_length=4608 data.enable_thinking=False data.max_response_length=768 actor_rollout_ref.rollout.n=8 actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.actor.use_kl_loss=True memory.guided_rollouts=0 trainer.save_freq=40 trainer.test_freq=20 trainer.total_epochs=1 trainer.resume_mode=disable data.shuffle=True"
# self-distillation: the tilt tensor is built (data.attn_gamma) and the HF hooks are installed (model.attn_bias), but the
# rollout never applies it (rollout.attn_bias=False) and the actor applies it only in the teacher pass
SD="memory.self_distill=True actor_rollout_ref.actor.kl_loss_coef=0.3 actor_rollout_ref.actor.kl_loss_type=low_var_kl data.attn_gamma=3.0 actor_rollout_ref.model.attn_bias=True actor_rollout_ref.rollout.attn_bias=False actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.attn_implementation=sdpa"
CTRL="actor_rollout_ref.actor.kl_loss_coef=0.01"
EV="python -m tests.experiments.common.evaluate_memory_generator --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B --engine vllm --gpu_memory_utilization 0.85 --thinking off --max_new_tokens 768"
last_ck () { ls -d outputs/rl/$1/hf/global_step_* | awk -F_ '{print $NF, $0}' | sort -n | tail -1 | cut -d" " -f2; }
bgp () {   # gpu, stream, order, out
  [ -f $4 ] && { echo "skip $4 (exists)"; return 0; }
  CUDA_VISIBLE_DEVICES=$1 python scripts/build_generation_prompts.py --model q3_4b --fit-stream train6 --stream $2 --order $3 --peer-names on --out $4 > $L/named_prompts_$2.out 2>&1 \
    && tail -2 $L/named_prompts_$2.out || { echo "FAILED prompts $2 (see $L/named_prompts_$2.out)"; return 1; }
}
prompts () {
  echo "=== named prompt files: $(date)"
  ( bgp 0 train6 fixed $P/prompts_train6_fixed_named.jsonl ) & ( bgp 1 indist6 shuffled0 $P/prompts_indist6_probe_named.jsonl ) & ( bgp 2 ood6 shuffled0 $P/prompts_ood6_probe_named.jsonl ) & wait
  for f in $P/prompts_train6_fixed_named.jsonl $P/prompts_indist6_probe_named.jsonl $P/prompts_ood6_probe_named.jsonl; do [ -f $f ] || return 1; done
  python scripts/record_quality.py --prompts $P/prompts_indist6_probe_named.jsonl --out $P/record_indist6_named.json 2>&1 | grep -v "^$" | head -4
  [ -f $D/full_peers_named.parquet ] || python -m training.sigma_rl.build_rl_data --prompts $P/prompts_train6_fixed_named.jsonl --records data/mixed_train_big6/train.jsonl --out $D/full_peers_named.parquet --guided none --prompt_source peers --solo_fraction 0.25 2>&1 | grep -i 'build-rl-data\|rows\|error' | tail -3
  [ -f $D/val_peers_named.parquet ] || python -m training.sigma_rl.build_rl_data --prompts $P/prompts_indist6_probe_named.jsonl --records data/indist6/test.jsonl --out $D/val_peers_named.parquet --guided none --prompt_source peers --every 8 --limit 512 2>&1 | grep -i 'build-rl-data\|rows\|error' | tail -3
  python - <<'PY'
import pandas as pd, json
d = pd.read_parquet("outputs/rl/data/q3_4b_6peer/full_peers_named.parquet"); v = pd.read_parquet("outputs/rl/data/q3_4b_6peer/val_peers_named.parquet")
r = d.iloc[3]; print("train rows", len(d), "val rows", len(v)); print("sources", d["extra_info"].apply(lambda e: e["prompt_source"]).value_counts().to_dict())
print(json.dumps(list(r["prompt"])[0]["content"][:400])); u = list(r["prompt"])[-1]["content"]; print(u[u.find("Peer answers"):][:300].replace("\n", " | "))
print("spans", r["extra_info"]["peer_spans"], "probs", r["extra_info"]["memory_prob"])
PY
}
train2 () {   # exp, extra overrides: 40 steps seed 1, then a full epoch seed 2 from step 40
  local exp=$1; shift
  echo "=== $exp phase 1 (40 steps, seed 1) on GPUs 0-7: $(date)"
  GPUS=0,1,2,3,4,5,6,7 EXP=${exp}_p1 TRAIN=$D/full_peers_named.parquet VAL=$D/val_peers_named.parquet bash training/scripts/train_grpo.sh $COMMON "$@" trainer.total_training_steps=40 > $L/${exp}_p1.train.out 2>&1
  echo "=== phase 1 finished (exit $?): $(date)"
  local ck0=outputs/rl/${exp}_p1/hf/global_step_40; ls $ck0/config.json || return 1
  echo "=== $exp phase 2 (from step 40, seed 2, full epoch) on GPUs 0-7: $(date)"
  GPUS=0,1,2,3,4,5,6,7 EXP=$exp MODEL=$ck0 TRAIN=$D/full_peers_named.parquet VAL=$D/val_peers_named.parquet bash training/scripts/train_grpo.sh $COMMON "$@" +data.seed=2 > $L/${exp}.train.out 2>&1
  echo "=== phase 2 finished (exit $?): $(date)"; ls -d outputs/rl/$exp/hf/global_step_*
}
ev () {   # gpu, checkpoint-or-base, stream, mode, extra, tag  (named prompt files; OOD on the whole stream)
  local ckarg="" outdir=$B/${3}6_$6 name=base
  if [ "$2" != base ]; then ckarg="--checkpoint $2"; outdir=$2/eval_${3}6_$6; name=$(basename $(dirname $(dirname $2))); fi
  [ -f $outdir/eval_metrics.json ] && { echo "skip $outdir"; return 0; }
  CUDA_VISIBLE_DEVICES=$1 $EV $ckarg --prompts $P/prompts_${3}6_probe_named.jsonl --records data/${3}6/test.jsonl --mode $4 $5 --output $outdir > $L/evn_${name}_${3}_$6.out 2>&1
}
tests6 () {   # one checkpoint-or-base, six conditions on GPUs $2..$2+5 (named prompts): peers at gamma 0, tilt gamma 3, solo
  local g=$2
  ( ev $((g)) $1 indist peers "" peers_named ) & ( ev $((g+1)) $1 ood peers "" peers_named ) &
  ( ev $((g+2)) $1 indist peers "--attn_gamma 3" tilt_named ) & ( ev $((g+3)) $1 ood peers "--attn_gamma 3" tilt_named ) &
  ( ev $((g+4)) $1 indist solo "" solo ) & ( ev $((g+5)) $1 ood solo "" solo ) &
}
report () { for f in "$@"; do [ -f $f ] && python -c "import json; m=json.load(open('$f')); print('$f'.replace('/eval_metrics.json',''), round(100*m['accuracy'],2), {k: round(100*v,1) for k,v in m['by_task'].items()})"; done; }
tests () {
  local ck1=$(last_ck nsd_named) ck2=$(last_ck nctrl_named); echo "=== tests: NSD $ck1, NCTRL $ck2, frozen: $(date)"
  tests6 $ck1 0; ( ev 6 $ck2 indist peers "" peers_named ) & ( ev 7 $ck2 ood peers "" peers_named ) & wait
  tests6 $ck2 0; ( ev 6 base indist peers "" peers_named ) & ( ev 7 base ood peers "" peers_named ) & wait
  ( ev 0 base indist peers "--attn_gamma 3" tilt_named ) & ( ev 1 base ood peers "--attn_gamma 3" tilt_named ) & wait
  # the named checkpoints on the ORIGINAL (unnamed) prompts: does the learned judgement need the names at test time?
  ( CUDA_VISIBLE_DEVICES=2 $EV --checkpoint $ck1 --prompts $P/prompts_indist6_probe.jsonl --records data/indist6/test.jsonl --mode peers --output $ck1/eval_indist6_peers > $L/evn_nsd_indist_unnamed.out 2>&1 ) &
  ( CUDA_VISIBLE_DEVICES=3 $EV --checkpoint $ck1 --prompts $P/prompts_ood6_probe.jsonl --records data/ood6/test.jsonl --mode peers --output $ck1/eval_ood6_peers > $L/evn_nsd_ood_unnamed.out 2>&1 ) & wait
  report $B/*named*/eval_metrics.json $ck1/eval_*/eval_metrics.json $ck2/eval_*/eval_metrics.json
}
smoke () {   # two NSD steps: the teacher pass must run and sd/teacher_minus_student_* must be non-zero
  echo "=== smoke NSD (2 steps) on GPUs 0-7: $(date)"
  GPUS=0,1,2,3,4,5,6,7 EXP=nsd_smoke TRAIN=$D/full_peers_named.parquet VAL=$D/val_peers_named.parquet bash training/scripts/train_grpo.sh $COMMON $SD trainer.total_training_steps=2 trainer.test_freq=1000 trainer.save_freq=1000 > $L/nsd_smoke.train.out 2>&1
  echo "exit $?"; grep -o "step:[0-9]* .*" $L/nsd_smoke.train.out | tr ' ' '\n' | grep -i "sd/\|kl_loss\|critic/score/mean\|response_length/mean\|time/step\|^step" | tr '\n' ' '; echo
  grep -i "tilt installed\|error\|Traceback" $L/nsd_smoke.train.out | head -5
}
case $WHAT in
  prompts) prompts ;;
  smoke) prompts && smoke ;;
  nsd) train2 nsd_named $SD ;;
  nctrl) train2 nctrl_named $CTRL ;;
  tests) tests ;;
  all) prompts && train2 nsd_named $SD && train2 nctrl_named $CTRL && tests; echo NAMED_COMPLETE ;;
esac
