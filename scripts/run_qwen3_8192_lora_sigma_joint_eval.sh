#!/usr/bin/env bash
# Dynamic 8192-token experiment scheduler for:
#   Qwen3-0.6B, Qwen3-4B, Qwen3-8B
#
# It polls nvidia-smi and launches a ready task only when an allowed GPU is idle.
# DAG:
#   LoRA-AR train                 -> LoRA-AR eval p0/p50/p70/p90
#   Sigma train, peer_mode=joint  -> Sigma eval p0/p50/p70/p90, peer_mode=joint
#   LoRA-AR train                 -> LoRA-AR -> Sigma train, peer_mode=joint
#   LoRA-AR -> Sigma train        -> LoRA-AR -> Sigma eval p0/p50/p70/p90, peer_mode=joint
#
# No existing 4096 outputs are overwritten; all output dirs include 8192.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

CONDA_BIN="${CONDA_BIN:-/home/peilin/miniconda3/bin/conda}"
HFM="${HFM:-/mnt/data/peilin/HF_MODEL}"
TRAIN="${TRAIN:-data/mixed_train_labeled.jsonl}"
TEST_DIR="${TEST_DIR:-data/CF_unified}"
TRAIN_PEER_MODE="${TRAIN_PEER_MODE:-joint}"
EVAL_PEER_MODE="${EVAL_PEER_MODE:-joint}"

POLL_SECONDS="${POLL_SECONDS:-120}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-1000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
MAX_STEPS="${MAX_STEPS:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

LOG_ROOT="${LOG_ROOT:-logs/qwen3_8192_lora_sigma}"
OUT_LORA_EVAL="${OUT_LORA_EVAL:-outputs/lora_cf_eval_8192}"
mkdir -p "$LOG_ROOT" "$OUT_LORA_EVAL"

if [ -n "${GPUS:-}" ]; then
  read -r -a ALLOWED_GPUS <<< "$GPUS"
else
  mapfile -t ALLOWED_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi

read -r -a SPLITS <<< "${SPLITS:-p0 p50 p70 p90}"

TMPD="${TMPDIR:-/tmp}/sigma_mem_qwen3_8192_$$"
mkdir -p "$TMPD"
trap 'rm -rf "$TMPD"' EXIT

LORA_TRAIN_CFG="$TMPD/lora_joint_selector_train8192.yaml"
LORA_EVAL_CFG="$TMPD/lora_joint_selector_eval8192.yaml"
SIGMA_TRAIN_CFG="$TMPD/symmetric_memory_train8192.yaml"
SIGMA_EVAL_CFG="$TMPD/symmetric_memory_eval8192.yaml"

sed -E 's/^max_length:.*/max_length: 8192/' configs/lora_joint_selector_train4096.yaml > "$LORA_TRAIN_CFG"
cp configs/lora_joint_selector_eval8192.yaml "$LORA_EVAL_CFG"
sed -E 's/^max_length:.*/max_length: 8192/' configs/symmetric_memory_train4096.yaml > "$SIGMA_TRAIN_CFG"
sed -E 's/^max_length:.*/max_length: 8192/' configs/symmetric_memory.yaml > "$SIGMA_EVAL_CFG"

EXTRA_TRAIN_ARGS=()
if [ -n "$MAX_STEPS" ]; then
  EXTRA_TRAIN_ARGS+=(--max_steps "$MAX_STEPS")
fi

EXTRA_LORA_EVAL_ARGS=()
EXTRA_SIGMA_EVAL_ARGS=()
if [ -n "$MAX_EXAMPLES" ]; then
  EXTRA_LORA_EVAL_ARGS+=(--max_samples "$MAX_EXAMPLES")
  EXTRA_SIGMA_EVAL_ARGS+=(--max_examples "$MAX_EXAMPLES")
fi

# tag | model path | conda env
MODELS=(
  "q3_0.6b|$HFM/Qwen3-0.6B|sigma"
  "q3_4b|$HFM/Qwen3-4B|sigma"
  "q3_8b|$HFM/Qwen3-8B|sigma"
)

declare -A MODEL_PATH_BY_TAG
declare -A ENV_BY_TAG
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag model_path env_name <<< "$entry"
  MODEL_PATH_BY_TAG["$tag"]="$model_path"
  ENV_BY_TAG["$tag"]="$env_name"
done

lora_ckpt() {
  echo "outputs/lora_ar_8192_$1"
}

sigma_ckpt() {
  echo "outputs/sigma_8192_${TRAIN_PEER_MODE}_$1/proto"
}

lora_sigma_ckpt() {
  echo "outputs/sigma_on_lora_ar_8192_${TRAIN_PEER_MODE}_$1/proto"
}

lora_eval_out() {
  echo "$OUT_LORA_EVAL/lora_ar_$1/$2"
}

sigma_eval_out() {
  echo "outputs/eval_sigma_8192_${EVAL_PEER_MODE}_$1/proto_$2"
}

lora_sigma_eval_out() {
  echo "outputs/eval_sigma_on_lora_ar_8192_${EVAL_PEER_MODE}_$1/proto_$2"
}

task_key() {
  local kind="$1" tag="$2" split="${3:-}"
  if [ -n "$split" ]; then
    echo "$kind|$tag|$split"
  else
    echo "$kind|$tag|"
  fi
}

task_done() {
  local kind="$1" tag="$2" split="${3:-}" out
  case "$kind" in
    lora_train)
      out="$(lora_ckpt "$tag")"
      [ -f "$out/joint_selector_head.pt" ] && [ -d "$out/lora_adapter" ]
      ;;
    lora_eval)
      out="$(lora_eval_out "$tag" "$split")"
      [ -f "$out/eval_metrics.json" ]
      ;;
    sigma_train)
      out="$(sigma_ckpt "$tag")"
      [ -f "$out/sym_memory.pt" ]
      ;;
    sigma_eval)
      out="$(sigma_eval_out "$tag" "$split")"
      [ -f "$out/eval_metrics.json" ]
      ;;
    lora_sigma_train)
      out="$(lora_sigma_ckpt "$tag")"
      [ -f "$out/sym_memory.pt" ]
      ;;
    lora_sigma_eval)
      out="$(lora_sigma_eval_out "$tag" "$split")"
      [ -f "$out/eval_metrics.json" ]
      ;;
    *)
      return 1
      ;;
  esac
}

deps_met() {
  local kind="$1" tag="$2" split="${3:-}"
  case "$kind" in
    lora_train|sigma_train)
      return 0
      ;;
    lora_eval)
      task_done lora_train "$tag"
      ;;
    sigma_eval)
      task_done sigma_train "$tag"
      ;;
    lora_sigma_train)
      task_done lora_train "$tag"
      ;;
    lora_sigma_eval)
      task_done lora_sigma_train "$tag"
      ;;
    *)
      return 1
      ;;
  esac
}

in_allowed_gpus() {
  local gpu="$1" x
  for x in "${ALLOWED_GPUS[@]}"; do
    [ "$x" = "$gpu" ] && return 0
  done
  return 1
}

declare -A GPU_TASK
declare -A PID_TASK
declare -A PID_LOG
declare -A TASK_PID

active_task_count() {
  set +u
  local n=${#PID_TASK[@]}
  set -u
  echo "$n"
}

find_free_gpu() {
  local idx used util
  while IFS=',' read -r idx used util; do
    idx="${idx//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    in_allowed_gpus "$idx" || continue
    [ -n "${GPU_TASK[$idx]:-}" ] && continue
    if [ "$used" -le "$GPU_MAX_USED_MB" ] && [ "$util" -le "$GPU_MAX_UTIL" ]; then
      echo "$idx"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  return 1
}

launch_task() {
  local kind="$1" tag="$2" split="${3:-}" gpu="$4"
  local model_path="${MODEL_PATH_BY_TAG[$tag]}"
  local env_name="${ENV_BY_TAG[$tag]}"
  local key log out ckpt lora pid
  key="$(task_key "$kind" "$tag" "$split")"

  case "$kind" in
    lora_train)
      out="$(lora_ckpt "$tag")"
      log="$LOG_ROOT/train_lora_ar_${tag}.log"
      echo "[launch] gpu=$gpu $key -> $out"
      CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u train_joint_selector.py \
        --config "$LORA_TRAIN_CFG" \
        --central_model "$model_path" \
        --offline_data "$TRAIN" \
        --model_variant ar_shared_state_selector \
        --use_shared_state false \
        --use_lora true \
        --output_dir "$out" \
        "${EXTRA_TRAIN_ARGS[@]}" \
        > "$log" 2>&1 &
      ;;
    lora_eval)
      ckpt="$(lora_ckpt "$tag")"
      out="$(lora_eval_out "$tag" "$split")"
      log="$LOG_ROOT/eval_lora_ar_${tag}_${split}.log"
      echo "[launch] gpu=$gpu $key -> $out"
      CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u eval_joint_selector.py \
        --config "$LORA_EVAL_CFG" \
        --checkpoint "$ckpt" \
        --central_model "$model_path" \
        --offline_data "$TEST_DIR/${split}.jsonl" \
        --model_variant ar_shared_state_selector \
        --use_shared_state false \
        --init_state zeros \
        --mode read_only \
        --test_order orig \
        --eval_write_policy none \
        --output "$out" \
        "${EXTRA_LORA_EVAL_ARGS[@]}" \
        > "$log" 2>&1 &
      ;;
    sigma_train)
      out="$(sigma_ckpt "$tag")"
      log="$LOG_ROOT/train_sigma_${tag}.log"
      echo "[launch] gpu=$gpu $key peer_mode=$TRAIN_PEER_MODE -> $out"
      CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u train_symmetric_memory.py \
        --config "$SIGMA_TRAIN_CFG" \
        --central_model "$model_path" \
        --phi_mode proto \
        --peer_mode "$TRAIN_PEER_MODE" \
        --use_joint off \
        --per_peer_decay on \
        --diff_write on \
        --offline_data "$TRAIN" \
        --output_dir "$out" \
        "${EXTRA_TRAIN_ARGS[@]}" \
        > "$log" 2>&1 &
      ;;
    sigma_eval)
      ckpt="$(sigma_ckpt "$tag")"
      out="$(sigma_eval_out "$tag" "$split")"
      log="$LOG_ROOT/eval_sigma_${EVAL_PEER_MODE}_${tag}_${split}.log"
      echo "[launch] gpu=$gpu $key peer_mode=$EVAL_PEER_MODE -> $out"
      CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u eval_symmetric_memory.py \
        --config "$SIGMA_EVAL_CFG" \
        --central_model "$model_path" \
        --checkpoint "$ckpt" \
        --phi_mode proto \
        --peer_mode "$EVAL_PEER_MODE" \
        --use_joint off \
        --per_peer_decay on \
        --offline_data "$TEST_DIR/${split}.jsonl" \
        --output "$out" \
        "${EXTRA_SIGMA_EVAL_ARGS[@]}" \
        > "$log" 2>&1 &
      ;;
    lora_sigma_train)
      lora="$(lora_ckpt "$tag")"
      out="$(lora_sigma_ckpt "$tag")"
      log="$LOG_ROOT/train_lora_to_sigma_${tag}.log"
      echo "[launch] gpu=$gpu $key peer_mode=$TRAIN_PEER_MODE -> $out"
      CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u train_symmetric_memory.py \
        --config "$SIGMA_TRAIN_CFG" \
        --central_model "$model_path" \
        --center_lora_checkpoint "$lora" \
        --phi_mode proto \
        --peer_mode "$TRAIN_PEER_MODE" \
        --use_joint off \
        --per_peer_decay on \
        --diff_write on \
        --offline_data "$TRAIN" \
        --output_dir "$out" \
        "${EXTRA_TRAIN_ARGS[@]}" \
        > "$log" 2>&1 &
      ;;
    lora_sigma_eval)
      lora="$(lora_ckpt "$tag")"
      ckpt="$(lora_sigma_ckpt "$tag")"
      out="$(lora_sigma_eval_out "$tag" "$split")"
      log="$LOG_ROOT/eval_lora_to_sigma_${EVAL_PEER_MODE}_${tag}_${split}.log"
      echo "[launch] gpu=$gpu $key peer_mode=$EVAL_PEER_MODE -> $out"
      CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$env_name" python -u eval_symmetric_memory.py \
        --config "$SIGMA_EVAL_CFG" \
        --central_model "$model_path" \
        --center_lora_checkpoint "$lora" \
        --checkpoint "$ckpt" \
        --phi_mode proto \
        --peer_mode "$EVAL_PEER_MODE" \
        --use_joint off \
        --per_peer_decay on \
        --offline_data "$TEST_DIR/${split}.jsonl" \
        --output "$out" \
        "${EXTRA_SIGMA_EVAL_ARGS[@]}" \
        > "$log" 2>&1 &
      ;;
    *)
      echo "[error] unknown task kind: $kind" >&2
      exit 2
      ;;
  esac

  pid=$!
  GPU_TASK["$gpu"]="$key"
  PID_TASK["$pid"]="$gpu|$key"
  PID_LOG["$pid"]="$log"
  TASK_PID["$key"]="$pid"
}

reap_jobs() {
  local running pid info gpu key log rc
  running="$(jobs -pr || true)"
  for pid in "${!PID_TASK[@]}"; do
    if grep -qx "$pid" <<< "$running"; then
      continue
    fi
    info="${PID_TASK[$pid]}"
    gpu="${info%%|*}"
    key="${info#*|}"
    log="${PID_LOG[$pid]}"
    set +e
    wait "$pid"
    rc=$?
    set -e
    unset "PID_TASK[$pid]"
    unset "PID_LOG[$pid]"
    unset "GPU_TASK[$gpu]"
    unset "TASK_PID[$key]"
    if [ "$rc" -ne 0 ]; then
      echo "[fail] $key on gpu=$gpu exited with code $rc. log: $log" >&2
      tail -80 "$log" >&2 || true
      exit "$rc"
    fi
    echo "[done] $key on gpu=$gpu"
  done
}

task_count_total=0
task_count_done=0
all_tasks_done() {
  local task kind tag split
  task_count_total=0
  task_count_done=0
  for task in "${TASKS[@]}"; do
    IFS='|' read -r kind tag split <<< "$task"
    task_count_total=$((task_count_total + 1))
    if task_done "$kind" "$tag" "$split"; then
      task_count_done=$((task_count_done + 1))
    fi
  done
  [ "$task_count_done" -eq "$task_count_total" ]
}

TASKS=()
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag _model_path _env_name <<< "$entry"
  TASKS+=("lora_train|$tag|")
  TASKS+=("sigma_train|$tag|")
  TASKS+=("lora_sigma_train|$tag|")
done
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag _model_path _env_name <<< "$entry"
  for split in "${SPLITS[@]}"; do
    TASKS+=("lora_eval|$tag|$split")
    TASKS+=("sigma_eval|$tag|$split")
    TASKS+=("lora_sigma_eval|$tag|$split")
  done
done

echo "=== Qwen3 8192 dynamic scheduler ==="
echo "allowed GPUs: ${ALLOWED_GPUS[*]}"
echo "idle threshold: memory.used <= ${GPU_MAX_USED_MB} MiB, utilization <= ${GPU_MAX_UTIL}%"
echo "poll interval: ${POLL_SECONDS}s"
echo "TRAIN_PEER_MODE=$TRAIN_PEER_MODE"
echo "EVAL_PEER_MODE=$EVAL_PEER_MODE"
echo "TRAIN=$TRAIN"
echo "TEST_DIR=$TEST_DIR"
echo "logs: $LOG_ROOT"
echo

while true; do
  reap_jobs
  if all_tasks_done; then
    if [ "$(active_task_count)" -eq 0 ]; then
      break
    fi
  fi

  launched=0
  for task in "${TASKS[@]}"; do
    IFS='|' read -r kind tag split <<< "$task"
    key="$(task_key "$kind" "$tag" "$split")"
    if task_done "$kind" "$tag" "$split"; then
      continue
    fi
    if [ -n "${TASK_PID[$key]:-}" ]; then
      continue
    fi
    deps_met "$kind" "$tag" "$split" || continue
    if ! gpu="$(find_free_gpu)"; then
      continue
    fi
    launch_task "$kind" "$tag" "$split" "$gpu"
    launched=1
  done

  if [ "$launched" -eq 0 ]; then
    all_tasks_done || true
    echo "[wait] ${task_count_done}/${task_count_total} tasks done; active=$(active_task_count); waiting for an idle GPU..."
    sleep "$POLL_SECONDS"
  else
    sleep 10
  fi
done

echo "=== all done ==="
echo "LoRA-AR checkpoints: outputs/lora_ar_8192_q3_*"
echo "LoRA-AR eval:        $OUT_LORA_EVAL/lora_ar_q3_*/{p0,p50,p70,p90}/eval_metrics.json"
echo "Sigma checkpoints:  outputs/sigma_8192_${TRAIN_PEER_MODE}_q3_*/proto"
echo "Sigma eval:         outputs/eval_sigma_8192_${EVAL_PEER_MODE}_q3_*/proto_{p0,p50,p70,p90}/eval_metrics.json"
echo "LoRA->Sigma ckpts:  outputs/sigma_on_lora_ar_8192_${TRAIN_PEER_MODE}_q3_*/proto"
echo "LoRA->Sigma eval:   outputs/eval_sigma_on_lora_ar_8192_${EVAL_PEER_MODE}_q3_*/proto_{p0,p50,p70,p90}/eval_metrics.json"
