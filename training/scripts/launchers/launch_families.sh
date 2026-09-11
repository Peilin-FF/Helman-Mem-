# Other central-model families under the same memory (docs section 23). The record does not change: it is built from
# the frozen Qwen3-4B judge's features and lives in the prompt files (memory_prob per peer, read before write along the
# stream); only the central model that reads the six peer answers changes. Per model and stream, three conditions:
#   tilt          peers + memory: the six peer answers in the prompt and the record's tilt (gamma 3) on the attention
#   peers         the same prompt, no memory
#   solo          the question only
#   tilt_swapped  (SWAP=1, and always in the smoke test) the record permuted by rank: the control that shows the
#                 direction of the tilt is what is read, not its mere presence
# Usage:   GPUS=0,1,2,3 bash training/scripts/launchers/launch_families.sh smoke|full [tag ...]
#          python scripts/families_table.py [--smoke]     the comparison table, frozen Qwen3-4B as the reference row
# Tags:    llama3     Meta-Llama-3-8B            BASE model (no chat template: plain prompt layout, weak instruction following)
#          llama31    Meta-Llama-3.1-8B-Instruct  the instruct Llama already on the server
#          ministral  Ministral-8B-Instruct-2410  (sliding window 32k, disabled in the engine: our contexts are shorter)
#          qwen25     Qwen2.5-7B-Instruct
#          phi4       phi-4 (14B)
# smoke:   the first N (96) in-distribution events, four conditions, every model, then the sanity table
# full:    the whole in-distribution stream (4,319) and the WHOLE OOD stream (17,403), three conditions (+ swapped with SWAP=1)
# Every evaluation is one single-GPU vLLM job; the tasks are spread over $GPUS as parallel per-GPU chains, OOD first.
# Finished evaluations (eval_metrics.json present) are skipped, so the script can be re-run to fill gaps.
set -u
[ -d training ] || cd /mnt/data/peilin/sigma-mem
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
P=outputs/gen/q3_4b; O=outputs/gen/families; L=logs; mkdir -p $L $O
H=/mnt/data/peilin/HF_MODEL
declare -A MODEL=([llama3]=$H/Meta-Llama-3-8B [llama31]=$H/Meta-Llama-3.1-8B-Instruct [ministral]=$H/Ministral-8B-Instruct-2410 [qwen25]=$H/Qwen2.5-7B-Instruct [phi4]=$H/phi-4)
what=${1:-smoke}; shift || true
models=${*:-llama3 llama31 ministral qwen25 phi4}
IFS=, read -ra G <<< "${GPUS:-0,1,2,3,4,5,6,7}"
if [ $what = smoke ]; then
  streams=indist; conds="tilt peers solo tilt_swapped"; extra_all="--max_examples ${N:-96}"
else
  streams="oodfull indist"; conds="tilt peers solo"; [ "${SWAP:-0}" = 1 ] && conds="$conds tilt_swapped"; extra_all=""
fi
for m in $models; do [ -n "${MODEL[$m]:-}" ] || { echo "unknown model tag $m (${!MODEL[*]})"; exit 1; }; done

one () {   # gpu, model tag, stream, condition
  local gpu=$1 tag=$2 st=$3 cond=$4 mode extra pf=$3
  [ $st = oodfull ] && pf=ood
  case $cond in
    tilt)         mode=peers; extra="--attn_gamma 3" ;;
    peers)        mode=peers; extra="" ;;
    solo)         mode=solo;  extra="" ;;
    tilt_swapped) mode=peers; extra="--attn_gamma 3 --swap_record" ;;
  esac
  local outdir=$O/$tag/${what}_${st}6_$cond log=$L/fam_${tag}_${what}_${st}_$cond.out
  [ -f $outdir/eval_metrics.json ] && { echo "skip $tag/$st/$cond (done)"; return 0; }
  echo "start $tag/$st/$cond on GPU $gpu: $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$gpu python -m tests.experiments.common.evaluate_memory_generator --central_model ${MODEL[$tag]} --engine vllm \
      --gpu_memory_utilization 0.85 --thinking off --max_new_tokens 768 $extra_all \
      --prompts $P/prompts_${pf}6_probe.jsonl --records data/${pf}6/test.jsonl --mode $mode $extra --output $outdir > $log 2>&1
  [ -f $outdir/eval_metrics.json ] && echo "done  $tag/$st/$cond: $(grep -o 'accuracy=[0-9.]*' $log | tail -1) $(date +%H:%M:%S)" \
                                   || echo "FAILED $tag/$st/$cond (see $log): $(grep -m1 -i 'error' $log | cut -c1-160)"
}

tasks=()
for st in $streams; do for m in $models; do for c in $conds; do tasks+=("$m $st $c"); done; done; done
n=${#G[@]}
echo "=== $what: ${#tasks[@]} evaluations over ${n} GPUs (${G[*]}): $(date)"
for ((g = 0; g < n; g++)); do
  ( for ((i = g; i < ${#tasks[@]}; i += n)); do one ${G[$g]} ${tasks[$i]}; done ) &
done
wait
echo "=== all evaluations finished: $(date)"
python scripts/families_table.py $([ $what = smoke ] && echo --smoke)
echo FAMILIES_COMPLETE
