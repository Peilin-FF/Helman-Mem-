#!/usr/bin/env bash
set -Euo pipefail

cd "$(dirname "$0")/.."

source scripts/gpu_lock_lib.sh

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_DATA_DIR="${HF_DATA_DIR:-/mnt/data/peilin/HF_DATA}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export FEEDBACK_CODE_EXEC_ALLOW="${FEEDBACK_CODE_EXEC_ALLOW:-1}"

PY_SIGMA="${PY_SIGMA:-/home/peilin/miniconda3/envs/sigma/bin/python}"
PY_SIGMA35="${PY_SIGMA35:-/home/peilin/miniconda3/envs/sigma3_5/bin/python}"
HFM="${HFM:-/mnt/data/peilin/HF_MODEL}"
CONFIG="${CONFIG:-configs/symmetric_memory_candidate_yesno.yaml}"
DATA_DIR="${DATA_DIR:-data/CF_unified}"
LOGD="${LOGD:-logs/missing_3peer_cf_eval}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-1000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"

mkdir -p "$LOGD"

if [[ -n "${GPUS:-}" ]]; then
  read -r -a ALLOWED_GPUS <<< "$GPUS"
else
  mapfile -t ALLOWED_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi
read -r -a SPLITS <<< "${SPLITS:-p0 p50 p70 p90}"

MODELS=(
  "q3_4b|$HFM/Qwen3-4B|outputs/sigma_candidate_yesno_q3_4b/proto|$PY_SIGMA"
  "q3_8b|$HFM/Qwen3-8B|outputs/sigma_candidate_yesno_q3_8b/proto|$PY_SIGMA"
  "q35_4b|$HFM/Qwen3.5-4B|outputs/sigma_candidate_yesno_q35_4b/proto|$PY_SIGMA35"
  "q35_9b|$HFM/Qwen3.5-9B|outputs/sigma_candidate_yesno_q35_9b/proto|$PY_SIGMA35"
)

declare -A MODEL_PATH CKPT PYTHON_BIN
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag model ckpt pybin <<< "$entry"
  MODEL_PATH["$tag"]="$model"
  CKPT["$tag"]="$ckpt"
  PYTHON_BIN["$tag"]="$pybin"
done

out_dir() {
  local tag="$1" arm="$2" split="$3"
  if [[ "$arm" == "center" ]]; then
    echo "outputs/eval_center_candidate_yesno_${tag}/proto_${split}"
  else
    echo "outputs/eval_sigma_candidate_yesno_${tag}/proto_${split}"
  fi
}

task_done() {
  local tag="$1" arm="$2" split="$3"
  [[ -f "$(out_dir "$tag" "$arm" "$split")/eval_metrics.json" ]]
}

need_task() {
  local tag="$1" arm="$2" split="$3"
  case "$tag:$arm" in
    q3_4b:center|q3_8b:center|q35_4b:center|q35_4b:sigma|q35_9b:center|q35_9b:sigma)
      task_done "$tag" "$arm" "$split" && return 1
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

in_allowed_gpus() {
  local gpu="$1" x
  for x in "${ALLOWED_GPUS[@]}"; do
    [[ "$x" == "$gpu" ]] && return 0
  done
  return 1
}

declare -A GPU_TASK PID_TASK PID_LOG TASK_PID

running_count() {
  set +u
  local n="${#PID_TASK[@]}"
  set -u
  echo "$n"
}

find_free_gpu() {
  local task_key="$1"
  local idx used util
  while IFS=',' read -r idx used util; do
    idx="${idx//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    in_allowed_gpus "$idx" || continue
    [[ -n "${GPU_TASK[$idx]:-}" ]] && continue
    if [[ "$used" -le "$GPU_MAX_USED_MB" && "$util" -le "$GPU_MAX_UTIL" ]] && gpu_try_claim "$idx" "$task_key"; then
      echo "$idx"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  return 1
}

launch_eval() {
  local tag="$1" arm="$2" split="$3" gpu="$4"
  local key="$tag|$arm|$split"
  local out log ablate=()
  out="$(out_dir "$tag" "$arm" "$split")"
  log="$LOGD/eval_${tag}_${arm}_${split}_gpu${gpu}.log"
  mkdir -p "$out"
  if [[ "$arm" == "center" ]]; then
    ablate=(--ablate_memory)
  fi
  echo "[$(date '+%F %T')] launch gpu=$gpu $key -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_BIN[$tag]}" -u eval_symmetric_memory.py \
    --config "$CONFIG" \
    --checkpoint "${CKPT[$tag]}" \
    --central_model "${MODEL_PATH[$tag]}" \
    --offline_data "$DATA_DIR/${split}.jsonl" \
    --output "$out" \
    --max_length "$MAX_LENGTH" \
    "${ablate[@]}" \
    > "$log" 2>&1 &
  local pid=$!
  PID_TASK["$pid"]="$key"
  PID_LOG["$pid"]="$log"
  TASK_PID["$key"]="$pid"
  GPU_TASK["$gpu"]="$key"
  gpu_update_claim_pid "$gpu" "$pid" "$key"
}

check_finished() {
  local pid key gpu status log
  for pid in "${!PID_TASK[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    key="${PID_TASK[$pid]}"
    log="${PID_LOG[$pid]}"
    status=0
    wait "$pid" || status=$?
    gpu=""
    for g in "${!GPU_TASK[@]}"; do
      if [[ "${GPU_TASK[$g]}" == "$key" ]]; then
        gpu="$g"
        unset 'GPU_TASK[$g]'
        gpu_release_claim "$g" "$key"
        break
      fi
    done
    unset 'PID_TASK[$pid]' 'PID_LOG[$pid]' 'TASK_PID[$key]'
    if [[ "$status" -ne 0 ]]; then
      echo "[$(date '+%F %T')] fail status=$status gpu=${gpu:-?} $key log=$log" >&2
      tail -40 "$log" >&2 || true
      exit "$status"
    fi
    echo "[$(date '+%F %T')] done gpu=${gpu:-?} $key"
    tail -2 "$log" || true
  done
}

pending_count() {
  local n=0 tag arm split
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag _ <<< "$entry"
    for arm in center sigma; do
      for split in "${SPLITS[@]}"; do
        if need_task "$tag" "$arm" "$split"; then
          n=$((n + 1))
        fi
      done
    done
  done
  echo "$n"
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag _ <<< "$entry"
  [[ -x "${PYTHON_BIN[$tag]}" ]] || { echo "[fail] missing python: ${PYTHON_BIN[$tag]}" >&2; exit 1; }
  [[ -f "${CKPT[$tag]}/sym_memory.pt" ]] || { echo "[fail] missing checkpoint: ${CKPT[$tag]}/sym_memory.pt" >&2; exit 1; }
done
for split in "${SPLITS[@]}"; do
  [[ -f "$DATA_DIR/${split}.jsonl" ]] || { echo "[fail] missing data: $DATA_DIR/${split}.jsonl" >&2; exit 1; }
done

echo "[$(date '+%F %T')] start missing 3-peer CF eval"
echo "data=$DATA_DIR max_length=$MAX_LENGTH splits=${SPLITS[*]} gpus=${ALLOWED_GPUS[*]}"

while true; do
  check_finished
  pending="$(pending_count)"
  running="$(running_count)"
  if [[ "$pending" -eq 0 && "$running" -eq 0 ]]; then
    break
  fi
  launched=0
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag _ <<< "$entry"
    for arm in center sigma; do
      for split in "${SPLITS[@]}"; do
        need_task "$tag" "$arm" "$split" || continue
        key="$tag|$arm|$split"
        [[ -n "${TASK_PID[$key]:-}" ]] && continue
        if gpu="$(find_free_gpu "$key")"; then
          launch_eval "$tag" "$arm" "$split" "$gpu"
          launched=$((launched + 1))
        fi
      done
    done
  done
  check_finished
  pending="$(pending_count)"
  running="$(running_count)"
  echo "[$(date '+%F %T')] status pending=$pending running=$running launched=$launched"
  if [[ "$pending" -eq 0 && "$running" -eq 0 ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "[$(date '+%F %T')] all missing 3-peer CF evals complete"
