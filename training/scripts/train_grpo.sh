#!/usr/bin/env bash
# One GRPO run of the central model (Ray + FSDP + vLLM, vendored verl). pipeline.train calls it for every arm; by hand:
#   EXP=run3b_tilt MODEL=/models/Qwen3-4B OUT=outputs/train/run3b_tilt TRAIN=... VAL=... \
#     bash training/scripts/train_grpo.sh data.attn_gamma=3.0 ...
# Environment: EXP, MODEL, OUT, TRAIN, VAL (required); CUDA_VISIBLE_DEVICES (the GPUs; pipeline.run sets it);
# RAY_BASE (default /tmp/kalman_ray_$USER; unix socket paths are limited to 107 characters). Hydra overrides follow.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${EXP:?set EXP=<run name>}"
: "${MODEL:?set MODEL=<initial model directory>}"
: "${OUT:?set OUT=<output directory>}"
: "${TRAIN:?set TRAIN=<train parquet>}"
: "${VAL:?set VAL=<validation parquet>}"
GPUS="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES=<device ids>}"
NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
export PYTHONPATH="$ROOT:$ROOT/training/verl${PYTHONPATH:+:$PYTHONPATH}"
RAY_BASE="${RAY_BASE:-/tmp/kalman_ray_${USER:-user}}"
export RAY_TMPDIR="$RAY_BASE/$(printf '%s' "$OUT" | md5sum | cut -c1-8)"   # one Ray temp dir per run, so runs can go side by side
export FEEDBACK_CODE_EXEC_ALLOW=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export WANDB_DIR="$OUT"
mkdir -p "$OUT" "$RAY_TMPDIR"
# a killed job must not leave Ray workers holding the GPUs: kill this driver's whole process tree, nothing else
PY=""
kill_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done; kill -9 "$1" >/dev/null 2>&1 || true; }
cleanup() { if [ -n "$PY" ]; then kill_tree "$PY"; fi; }   # an if, not &&: under set -e a false test here would turn exit 0 into 1
trap cleanup EXIT HUP INT TERM
echo "[train_grpo] exp=$EXP gpus=$GPUS model=$MODEL train=$TRAIN val=$VAL out=$OUT extra=$*"
python -m training.kalman_rl.main_grpo \
  trainer.experiment_name="$EXP" \
  trainer.default_local_dir="$OUT" \
  trainer.n_gpus_per_node="$NGPU" \
  data.train_files="$TRAIN" \
  data.val_files="$VAL" \
  actor_rollout_ref.model.path="$MODEL" \
  ray_init.temp_dir="$RAY_TMPDIR" \
  hydra.run.dir="$OUT/hydra" \
  "$@" > >(tee -a "$OUT/train.log") 2>&1 &
PY=$!
set +e
wait "$PY"
STATUS=$?
set -e
PY=""
echo "[train_grpo] trainer exited with status $STATUS"
exit "$STATUS"
