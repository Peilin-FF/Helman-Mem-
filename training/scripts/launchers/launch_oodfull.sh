# Full OOD stream (all 17,403 events, not every 4th) for the four thinking-off models in the four input
# conditions. The record is built along the stream read-before-write, so the full stream gives it four
# times the feedback of the subsample: this is the mature-record condition. Results go to
# eval_oodfull6_<cond> so the existing every-4th results (eval_ood6_<cond>) are untouched.
set -u
[ -d training ] || cd /mnt/data/peilin/sigma-mem
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
P=outputs/gen/q3_4b; B=$P/base6_nothink; L=logs; mkdir -p $L
EV="python -m tests.experiments.common.evaluate_memory_generator --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B --engine vllm --gpu_memory_utilization 0.85 --thinking off --max_new_tokens 768"
MODELS="base outputs/rl/run3b_tilt/hf/global_step_276 outputs/rl/ctrl3b_peers/hf/global_step_276 outputs/rl/solo3b_q/hf/global_step_276"

one () {   # gpu, model, cond-tag
  local gpu=$1 mdl=$2 tag=$3 mode extra ckarg outdir name
  case $tag in
    tilt)         mode=peers; extra="--attn_gamma 3" ;;
    peers)        mode=peers; extra="" ;;
    solo)         mode=solo;  extra="" ;;
    tilt_swapped) mode=peers; extra="--attn_gamma 3 --swap_record" ;;
  esac
  if [ "$mdl" = base ]; then ckarg=""; outdir=$B/oodfull6_$tag; name=base
  else ckarg="--checkpoint $mdl"; outdir=$mdl/eval_oodfull6_$tag; name=$(basename $(dirname $(dirname $mdl))); fi
  [ -f $outdir/eval_metrics.json ] && { echo "skip $name/$tag (done)"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $EV $ckarg --prompts $P/prompts_ood6_probe.jsonl --records data/ood6/test.jsonl \
      --mode $mode $extra --output $outdir > $L/ev_oodfull_${name}_${tag}.out 2>&1
  echo "done $name/$tag"
}

# 16 evaluations, 8 at a time
i=0
for m in $MODELS; do
  for t in tilt peers solo tilt_swapped; do
    one $((i % 8)) $m $t &
    i=$((i + 1))
    [ $((i % 8)) -eq 0 ] && { echo "=== waiting for round $((i / 8)): $(date)"; wait; }
  done
done
wait
echo "=== all evaluations finished: $(date)"
for m in $MODELS; do
  for t in tilt peers solo tilt_swapped; do
    if [ "$m" = base ]; then f=$B/oodfull6_$t/eval_metrics.json; n=base; else f=$m/eval_oodfull6_$t/eval_metrics.json; n=$(basename $(dirname $(dirname $m))); fi
    [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print('%-14s %-13s %6.2f  n=%d  %s' % ('$n','$t',100*d['accuracy'],d['num_samples'],{k: round(100*v,1) for k,v in d['by_task'].items()}))"
  done
done
echo OODFULL_COMPLETE
