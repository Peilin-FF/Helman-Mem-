#!/usr/bin/env bash
# Multi-GPU full-parameter SFT (torchrun + FSDP) on a parquet built by training/sigma_rl/build_sft_data.py.
#   rproj submit 'GPUS=0,1,2,3 EXP=q3_4b_sft_hinted_memory TRAIN=outputs/rl/data/q3_4b/sft_hinted_memory.parquet bash training/scripts/train_sft.sh'
# Hydra overrides may follow, e.g.  optim.lr=3e-6 trainer.total_epochs=2 data.max_length=6144
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${GPUS:?set GPUS=<comma-separated device ids> (check rproj gpu first)}"
: "${EXP:?set EXP=<experiment name>}"
: "${TRAIN:?set TRAIN=<sft parquet>}"
VAL="${VAL:-$TRAIN}"
MODEL="${MODEL:-/mnt/data/peilin/HF_MODEL/Qwen3-4B}"
OUT="${OUT:-$ROOT/outputs/rl/$EXP}"
export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
export PYTHONPATH="$ROOT:$ROOT/training/verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true
PORT="${PORT:-$((20000 + RANDOM % 20000))}"
mkdir -p "$OUT"
echo "[train_sft] exp=$EXP gpus=$GPUS model=$MODEL train=$TRAIN out=$OUT extra=$*"
torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" --master_port="$PORT" \
  -m verl.trainer.fsdp_sft_trainer \
  --config-path="$ROOT/training/configs" --config-name=sft_sigma \
  data.train_files="$TRAIN" \
  data.val_files="$VAL" \
  data.custom_cls.path="$ROOT/training/sigma_rl/dataset.py" \
  data.custom_cls.name=SigmaSFTDataset \
  model.partial_pretrain="$MODEL" \
  trainer.experiment_name="$EXP" \
  trainer.default_local_dir="$OUT" \
  trainer.n_gpus_per_node="$NGPU" \
  hydra.run.dir="$OUT/hydra" \
  "$@" 2>&1 | tee -a "$OUT/train.log"
