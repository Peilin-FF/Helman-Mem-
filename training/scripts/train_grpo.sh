#!/usr/bin/env bash
# Multi-GPU GRPO on the central model with the memory hooks (Ray + FSDP + vLLM, vendored verl).
#
# Run on the server from the project root, through rproj so the code is snapshotted:
#   rproj submit 'GPUS=0,1,2,3 EXP=q3_4b_grpo_memory TRAIN=outputs/rl/data/q3_4b/train_memory.parquet \
#                 VAL=outputs/rl/data/q3_4b/val_indist.parquet bash training/scripts/train_grpo.sh'
# Any hydra override can follow the script, e.g.
#   ... bash training/scripts/train_grpo.sh memory.guided_rollouts=0 actor_rollout_ref.rollout.n=4 trainer.total_training_steps=50
#
# Environment variables: GPUS (required, check `rproj gpu` first), EXP (required), TRAIN, VAL (required),
# MODEL (default Qwen3-4B), OUT (default outputs/rl/$EXP).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${GPUS:?set GPUS=<comma-separated device ids> (check rproj gpu first)}"
: "${EXP:?set EXP=<experiment name>}"
: "${TRAIN:?set TRAIN=<train parquet from training/sigma_rl/build_rl_data.py>}"
: "${VAL:?set VAL=<validation parquet>}"
MODEL="${MODEL:-/mnt/data/peilin/HF_MODEL/Qwen3-4B}"
OUT="${OUT:-$ROOT/outputs/rl/$EXP}"
export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
export PYTHONPATH="$ROOT:$ROOT/training/verl${PYTHONPATH:+:$PYTHONPATH}"
# one Ray temp dir per experiment (short: unix socket paths are limited to 107 chars), so that several
# trainings can run side by side
RAY_BASE="${RAY_BASE:-/mnt/data/peilin/.ray}"
export RAY_TMPDIR="$RAY_BASE/$(printf '%s' "$EXP" | md5sum | cut -c1-8)"
export FEEDBACK_CODE_EXEC_ALLOW=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export WANDB_DIR="$OUT"                      # wandb run files next to the outputs (login: `wandb login` on the server)
mkdir -p "$OUT" "$RAY_TMPDIR"
# a killed job (rproj kill -> SIGHUP) must not leave Ray workers holding the GPUs: kill the whole process tree
# of this job's driver (Ray's gcs / raylet / workers / vLLM engines are all its descendants), nothing else
PY=""
kill_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done; kill -9 "$1" >/dev/null 2>&1 || true; }
cleanup() { [ -n "$PY" ] && kill_tree "$PY"; }
trap cleanup EXIT HUP INT TERM
echo "[train_grpo] exp=$EXP gpus=$GPUS model=$MODEL train=$TRAIN val=$VAL out=$OUT extra=$*"
python -m training.sigma_rl.main_grpo \
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
wait "$PY"
STATUS=$?
PY=""
exit "$STATUS"
