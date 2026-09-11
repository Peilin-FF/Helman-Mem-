# A family's OWN memory (docs section 23): when the central model is Llama, the record's address comes from Llama too,
# never from the Qwen3-4B judge. Same pipeline as for Qwen3-4B, with the family's model as the frozen judge:
#   features   scripts/encode_context_features.py on train6 / indist6 / ood6 (question features and the six candidate-judge
#              hidden states, plain-text Yes/No judge prompts, no chat template), sharded over $GPUS
#   prompts    scripts/build_generation_prompts.py: PCA-256 addresses fit on train6, the Bayesian record run along the
#              test stream read-before-write (cold start, shuffled0 order = the probe files), then scripts/record_quality.py
#   eval       launch_families.sh full <tag> (peers + memory / peers / question only, both whole streams)
# Usage:  GPUS=0,1,2,3 bash training/scripts/launchers/launch_family_memory.sh smoke|features|prompts|eval|all <tag>
#         tags: llama31 ministral qwen25 phi4 (feedback_state/feature_streams.py MODELS; the base llama3 was dropped)
#         smoke = the whole pipeline on the first 48 events of train6 and indist6 under the tag <tag>_smoke (~4 min on 4 GPUs)
# Cost (full): each event needs one question pass and six judge prompts that each carry all six answers, so roughly
# 20k tokens per event on the training and in-distribution streams; expect ~15-20 GPU-hours per 8B family for the three
# streams (phi-4 about twice that). Shards already on disk are skipped, so a partial run can be resumed.
set -u
[ -d training ] || cd /mnt/data/peilin/sigma-mem
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
H=/mnt/data/peilin/HF_MODEL; F=outputs/context_features; L=logs; mkdir -p $L
declare -A MODEL=([llama3]=$H/Meta-Llama-3-8B [llama31]=$H/Meta-Llama-3.1-8B-Instruct [ministral]=$H/Ministral-8B-Instruct-2410 [qwen25]=$H/Qwen2.5-7B-Instruct [phi4]=$H/phi-4)
what=${1:-smoke}; base=${2:?tag}
[ -n "${MODEL[$base]:-}" ] || { echo "unknown tag $base (${!MODEL[*]})"; exit 1; }
IFS=, read -ra G <<< "${GPUS:-0,1,2,3,4,5,6,7}"; n=${#G[@]}
tag=$base; limit=""
[ $what = smoke ] && { tag=${base}_smoke; limit="--max-examples ${N:-48}"; }
declare -A INPUT=([train6]=data/mixed_train_big6/train.jsonl [indist6]=data/indist6/test.jsonl [ood6]=data/ood6/test.jsonl)
declare -A CACHE=([train6]=$F/${tag}_big6_ph/train [indist6]=$F/${tag}_indist6_ph/ood [ood6]=$F/${tag}_6_ph/ood)

encode_stream () {   # stream, number of shards: one shard per GPU, in parallel
  local st=$1 shards=$2 k
  for ((k = 0; k < shards; k++)); do
    local out=${CACHE[$st]}/shard$k.pt
    [ -f $out ] && { echo "skip $st shard $k (exists)"; continue; }
    ( CUDA_VISIBLE_DEVICES=${G[$((k % n))]} python scripts/encode_context_features.py --input ${INPUT[$st]} --output $out \
        --central-model ${MODEL[$base]} --include-context --save-peer-hidden --num-peers 6 --progress-every 500 \
        --num-shards $shards --shard-index $k $limit > $L/enc_${tag}_${st}_$k.out 2>&1 \
      && echo "done $st shard $k: $(date +%H:%M:%S)" || echo "FAILED $st shard $k (see $L/enc_${tag}_${st}_$k.out)" ) &
  done
  wait
}
features () {
  if [ $what = smoke ]; then
    echo "=== features (smoke, ${N:-48} events of train6 and indist6) on GPUs ${G[0]} ${G[1 % n]}: $(date)"
    encode_stream train6 1 & sleep 1; ( CUDA_VISIBLE_DEVICES=${G[1 % n]} python scripts/encode_context_features.py --input ${INPUT[indist6]} \
        --output ${CACHE[indist6]}/shard0.pt --central-model ${MODEL[$base]} --include-context --save-peer-hidden --num-peers 6 \
        --progress-every 10 --num-shards 1 --shard-index 0 $limit > $L/enc_${tag}_indist6_0.out 2>&1 && echo "done indist6 (smoke)" ) &
    wait
  else
    for st in train6 indist6 ood6; do echo "=== features $st over $n GPUs: $(date)"; encode_stream $st $n; done
  fi
}
prompts () {   # the record along each test stream from the family's own addresses (PCA fit on train6); GPU 0 of the list
  local O=outputs/gen/$tag dim=256; mkdir -p $O
  [ $what = smoke ] && dim=${DIM:-32}   # the PCA cannot have more components than fit events (48 in the smoke)
  for st in indist6 ood6; do
    [ $what = smoke ] && [ $st = ood6 ] && continue
    [ -f $O/prompts_${st}_probe.jsonl ] && { echo "skip prompts $st (exists)"; continue; }
    echo "=== prompts $st: $(date)"
    CUDA_VISIBLE_DEVICES=${G[0]} python scripts/build_generation_prompts.py --model $tag --fit-stream train6 --stream $st --order shuffled0 --dim $dim \
        --out $O/prompts_${st}_probe.jsonl > $L/prompts_${tag}_$st.out 2>&1 || { echo "FAILED prompts $st (see $L/prompts_${tag}_$st.out)"; return 1; }
    python scripts/record_quality.py --prompts $O/prompts_${st}_probe.jsonl --out $O/record_${st}.json 2>&1 | grep -v "^$" | head -6
  done
}
evaluate () {
  if [ $what = smoke ]; then   # the three conditions on the smoke prompts, in parallel, results under outputs/gen/families/<tag>_smoke
    local P=outputs/gen/$tag O=outputs/gen/families/$tag i=0 mode extra
    for cond in tilt peers solo; do
      case $cond in tilt) mode=peers; extra="--attn_gamma 3";; peers) mode=peers; extra="";; solo) mode=solo; extra="";; esac
      ( CUDA_VISIBLE_DEVICES=${G[$((i % n))]} python -m tests.experiments.common.evaluate_memory_generator --central_model ${MODEL[$base]} --engine vllm \
          --gpu_memory_utilization ${UTIL:-0.85} --thinking off --max_new_tokens 768 --prompts $P/prompts_indist6_probe.jsonl --records data/indist6/test.jsonl \
          --mode $mode $extra --output $O/smoke_indist6_$cond > $L/fam_${tag}_smoke_indist_$cond.out 2>&1 && echo "done eval $cond" || echo "FAILED eval $cond (see $L/fam_${tag}_smoke_indist_$cond.out)" ) &
      i=$((i + 1))
    done
    wait
    python scripts/families_table.py --smoke
  else
    GPUS=${GPUS:-0,1,2,3,4,5,6,7} ADDR=own bash training/scripts/launchers/launch_families.sh full $base
  fi
}
case $what in
  smoke|all) features && prompts && evaluate ;;
  features)  features ;;
  prompts)   prompts ;;
  eval)      evaluate ;;
  *) echo "usage: launch_family_memory.sh smoke|features|prompts|eval|all <tag>"; exit 1 ;;
esac
echo "FAMILY_MEMORY_DONE $what $tag $(date)"
