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
DATA_ROOT="${DATA_ROOT:-data/CF_unified_peer_counts_llama_bitcpm_shifted}"
CONFIG="${CONFIG:-configs/symmetric_memory_candidate_yesno.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs/eval_cf_unified_peer_counts_llama_bitcpm_shifted_by_center}"
LOGD="${LOGD:-logs/cf_unified_peer_counts_llama_bitcpm_shifted_by_center}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-1000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"

mkdir -p "$OUT_ROOT" "$LOGD"

if [[ -n "${GPUS:-}" ]]; then
  read -r -a ALLOWED_GPUS <<< "$GPUS"
else
  mapfile -t ALLOWED_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi
read -r -a SPLITS <<< "${SPLITS:-p0 p50 p70 p90}"
read -r -a PEER_COUNTS <<< "${PEER_COUNTS:-4 5}"

MODELS=(
  "q3_0.6b|$HFM/Qwen3-0.6B|outputs/sigma_candidate_yesno_q3_0.6b/proto|$PY_SIGMA"
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
  local tag="$1" peers="$2" arm="$3" split="$4"
  echo "$OUT_ROOT/$tag/peers${peers}/${arm}_${split}"
}

task_done() {
  local tag="$1" peers="$2" arm="$3" split="$4"
  [[ -f "$(out_dir "$tag" "$peers" "$arm" "$split")/eval_metrics.json" ]]
}

copy_result_dir() {
  local src="$1" dst="$2" label="$3"
  if [[ -f "$src/eval_metrics.json" && ! -f "$dst/eval_metrics.json" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
    echo "[$(date '+%F %T')] copied $label from $src"
  fi
}

bootstrap_existing() {
  local tag peers split arm src dst
  for peers in "${PEER_COUNTS[@]}"; do
    for split in "${SPLITS[@]}"; do
      for arm in center sigma; do
        src="outputs/eval_cf_unified_peer_counts_llama_bitcpm_shifted_q3_0.6b/peers${peers}/${arm}_${split}"
        dst="$(out_dir q3_0.6b "$peers" "$arm" "$split")"
        copy_result_dir "$src" "$dst" "q3_0.6b peers${peers} $arm $split"
      done
    done
  done

  for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag _model _ckpt _pybin <<< "$entry"
    for split in "${SPLITS[@]}"; do
      for arm in center sigma; do
        src="outputs/eval_cf_peer3_llama_by_center/$tag/${arm}_${split}"
        dst="$(out_dir "$tag" 4 "$arm" "$split")"
        copy_result_dir "$src" "$dst" "$tag peers4 $arm $split"
      done
    done
  done
}

in_allowed_gpus() {
  local gpu="$1" x
  for x in "${ALLOWED_GPUS[@]}"; do
    [[ "$x" == "$gpu" ]] && return 0
  done
  return 1
}

declare -A GPU_TASK PID_TASK PID_LOG TASK_PID

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
  local tag="$1" peers="$2" arm="$3" split="$4" gpu="$5"
  local key="$tag|peers${peers}|$arm|$split"
  local out log data ablate=()
  data="$DATA_ROOT/peers${peers}/${split}.jsonl"
  out="$(out_dir "$tag" "$peers" "$arm" "$split")"
  log="$LOGD/eval_${tag}_peers${peers}_${arm}_${split}_gpu${gpu}.log"
  mkdir -p "$out"
  if [[ "$arm" == "center" ]]; then
    ablate=(--ablate_memory)
  fi
  echo "[$(date '+%F %T')] launch gpu=$gpu $key -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_BIN[$tag]}" -u eval_symmetric_memory.py \
    --config "$CONFIG" \
    --checkpoint "${CKPT[$tag]}" \
    --central_model "${MODEL_PATH[$tag]}" \
    --num_peers "$peers" \
    --max_length "$MAX_LENGTH" \
    --offline_data "$data" \
    --output "$out" \
    --score_mode candidate_yesno \
    --peer_mode joint \
    --per_peer_decay off \
    "${ablate[@]}" \
    > "$log" 2>&1 &
  local pid=$!
  PID_TASK["$pid"]="$key"
  PID_LOG["$pid"]="$log"
  TASK_PID["$key"]="$pid"
  GPU_TASK["$gpu"]="$key"
  gpu_update_claim_pid "$gpu" "$pid" "$key"
}

reap_finished() {
  local pid key gpu rc
  set +u
  local pids=("${!PID_TASK[@]}")
  set -u
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    key="${PID_TASK[$pid]}"
    rc=0
    wait "$pid" || rc=$?
    gpu=""
    for g in "${!GPU_TASK[@]}"; do
      if [[ "${GPU_TASK[$g]}" == "$key" ]]; then
        gpu="$g"
        unset 'GPU_TASK[$g]'
        gpu_release_claim "$g" "$key"
        break
      fi
    done
    echo "[$(date '+%F %T')] finish rc=$rc gpu=${gpu:-?} $key log=${PID_LOG[$pid]}"
    if [[ "$rc" -ne 0 ]]; then
      echo "[fail] $key failed; see ${PID_LOG[$pid]}" >&2
      exit "$rc"
    fi
    unset 'PID_TASK[$pid]' 'PID_LOG[$pid]'
  done
}

active_count() {
  set +u
  echo "${#PID_TASK[@]}"
  set -u
}

validate_inputs() {
  local peers split tag
  for peers in "${PEER_COUNTS[@]}"; do
    for split in "${SPLITS[@]}"; do
      [[ -f "$DATA_ROOT/peers${peers}/${split}.jsonl" ]] || {
        echo "[fail] missing $DATA_ROOT/peers${peers}/${split}.jsonl" >&2
        exit 1
      }
    done
  done
  for tag in "${!MODEL_PATH[@]}"; do
    [[ -d "${MODEL_PATH[$tag]}" ]] || { echo "[fail] missing model ${MODEL_PATH[$tag]}" >&2; exit 1; }
    [[ -f "${CKPT[$tag]}/sym_memory.pt" ]] || { echo "[fail] missing checkpoint ${CKPT[$tag]}/sym_memory.pt" >&2; exit 1; }
    [[ -x "${PYTHON_BIN[$tag]}" ]] || { echo "[fail] missing python ${PYTHON_BIN[$tag]}" >&2; exit 1; }
  done
}

remaining_count() {
  local remaining=0 tag peers split arm entry
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag _model _ckpt _pybin <<< "$entry"
    for peers in "${PEER_COUNTS[@]}"; do
      for split in "${SPLITS[@]}"; do
        for arm in center sigma; do
          task_done "$tag" "$peers" "$arm" "$split" || remaining=$((remaining + 1))
        done
      done
    done
  done
  echo "$remaining"
}

validate_inputs
bootstrap_existing

echo "[$(date '+%F %T')] start shifted 4/5-peer by-center scheduler"
echo "data=$DATA_ROOT"
echo "out=$OUT_ROOT"
echo "peer_counts=${PEER_COUNTS[*]} models=${#MODELS[@]}"
echo "allowed_gpus=${ALLOWED_GPUS[*]} idle=(mem<=${GPU_MAX_USED_MB}MiB util<=${GPU_MAX_UTIL}%) poll=${POLL_SECONDS}s"

while true; do
  reap_finished
  launched=0
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag _model _ckpt _pybin <<< "$entry"
    for peers in "${PEER_COUNTS[@]}"; do
      for split in "${SPLITS[@]}"; do
        for arm in center sigma; do
          task_done "$tag" "$peers" "$arm" "$split" && continue
          key="$tag|peers${peers}|$arm|$split"
          [[ -n "${TASK_PID[$key]:-}" ]] && continue
          if gpu="$(find_free_gpu "$key")"; then
            launch_eval "$tag" "$peers" "$arm" "$split" "$gpu"
            launched=1
          else
            break 4
          fi
        done
      done
    done
  done

  remaining="$(remaining_count)"
  echo "[$(date '+%F %T')] status remaining=$remaining active=$(active_count)"
  [[ "$remaining" -eq 0 && "$(active_count)" -eq 0 ]] && break
  if [[ "$launched" -eq 0 ]]; then
    sleep "$POLL_SECONDS"
  fi
done

echo "[$(date '+%F %T')] all shifted 4/5-peer by-center evals complete"
