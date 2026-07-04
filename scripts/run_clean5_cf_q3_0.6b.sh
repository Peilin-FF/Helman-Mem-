#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FEEDBACK_CODE_EXEC_ALLOW=1

PYTHON="${PYTHON:-/home/peilin/miniconda3/envs/sigma/bin/python}"
CKPT="${CKPT:-outputs/sigma_candidate_yesno_q3_0.6b/proto}"
MODEL="${MODEL:-/mnt/data/peilin/HF_MODEL/Qwen3-0.6B}"
NEW_PEER_MODEL="${NEW_PEER_MODEL:-/mnt/data/peilin/HF_MODEL/Llama-3.2-3B-Instruct}"
IN_DIR="${IN_DIR:-data/peer_generalization/cf4_unified_quick}"
OUT_DIR="${OUT_DIR:-data/peer_generalization/cf5_clean_llama}"
LOG_DIR="${LOG_DIR:-logs/peer_generalization/cf5_clean_llama_q3_0.6b}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval_peer_generalization_cf5_clean_llama_q3_0.6b}"
EVAL_GPU="${EVAL_GPU:-4}"
read -r -a GEN_GPUS <<< "${GEN_GPUS:-4 5 6 7}"

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/labeled" "$LOG_DIR" "$EVAL_ROOT"

if [ ! -f "$CKPT/sym_memory.pt" ]; then
  echo "[error] missing checkpoint: $CKPT/sym_memory.pt" >&2
  exit 1
fi

splits=(p0 p50 p70 p90)
gen_pids=()
gen_names=()
for idx in "${!splits[@]}"; do
  split="${splits[$idx]}"
  gpu="${GEN_GPUS[$((idx % ${#GEN_GPUS[@]}))]}"
  in_file="$IN_DIR/${split}.jsonl"
  raw_file="$OUT_DIR/raw/${split}.jsonl"
  log_file="$LOG_DIR/add_peer_${split}.log"
  if [ -s "$raw_file" ]; then
    echo "[clean5] reuse raw $raw_file"
    continue
  fi
  echo "[clean5] add peer_4=$NEW_PEER_MODEL split=$split gpu=$gpu"
  env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/add_generated_peer.py \
    --input "$in_file" \
    --output "$raw_file" \
    --model "$NEW_PEER_MODEL" \
    --peer_key peer_4 \
    --remap peer_0=peer_0 \
    --remap peer_1=peer_1 \
    --remap peer_3=peer_2 \
    --remap peer_2=peer_3 \
    --device cuda:0 \
    --use_vllm true \
    --local_files_only true \
    --batch_size 4 \
    --write_chunk_size 64 \
    > "$log_file" 2>&1 &
  gen_pids+=("$!")
  gen_names+=("$split:$log_file")
  echo "[clean5] launched split=$split pid=${gen_pids[-1]} log=$log_file"
done
for i in "${!gen_pids[@]}"; do
  pid="${gen_pids[$i]}"
  name="${gen_names[$i]}"
  if ! wait "$pid"; then
    echo "[error] generation failed for $name" >&2
    log="${name#*:}"
    tail -80 "$log" >&2 || true
    exit 1
  fi
done

for split in "${splits[@]}"; do
  raw_file="$OUT_DIR/raw/${split}.jsonl"
  labeled_file="$OUT_DIR/labeled/${split}.jsonl"
  if [ -s "$labeled_file" ]; then
    echo "[clean5] reuse labeled $labeled_file"
    continue
  fi
  echo "[clean5] fill peer_4 correctness split=$split"
  "$PYTHON" -u scripts/fill_missing_peer_correct.py \
    --input "$raw_file" \
    --output "$labeled_file" \
    --peer_key peer_4 \
    --timeout 10 \
    > "$LOG_DIR/fill_correct_${split}.log" 2>&1
  tail -1 "$LOG_DIR/fill_correct_${split}.log"
done

for split in "${splits[@]}"; do
  labeled_file="$OUT_DIR/labeled/${split}.jsonl"
  echo "[clean5] eval center split=$split"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -u eval_symmetric_memory.py \
    --config configs/symmetric_memory_candidate_yesno.yaml \
    --checkpoint "$CKPT" \
    --central_model "$MODEL" \
    --num_peers 5 \
    --offline_data "$labeled_file" \
    --output "$EVAL_ROOT/center_${split}" \
    --score_mode candidate_yesno \
    --peer_mode joint \
    --per_peer_decay off \
    --ablate_memory \
    > "$LOG_DIR/eval_center_${split}.log" 2>&1
  tail -3 "$LOG_DIR/eval_center_${split}.log"

  echo "[clean5] eval sigma split=$split"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -u eval_symmetric_memory.py \
    --config configs/symmetric_memory_candidate_yesno.yaml \
    --checkpoint "$CKPT" \
    --central_model "$MODEL" \
    --num_peers 5 \
    --offline_data "$labeled_file" \
    --output "$EVAL_ROOT/sigma_${split}" \
    --score_mode candidate_yesno \
    --peer_mode joint \
    --per_peer_decay off \
    > "$LOG_DIR/eval_sigma_${split}.log" 2>&1
  tail -3 "$LOG_DIR/eval_sigma_${split}.log"
done

"$PYTHON" - <<PY
import json
from pathlib import Path
root = Path("$EVAL_ROOT")
for split in ["p0", "p50", "p70", "p90"]:
    vals = []
    for arm in ["center", "sigma"]:
        path = root / f"{arm}_{split}" / "eval_metrics.json"
        if path.exists():
            m = json.loads(path.read_text())
            vals.append(f"{arm}={m['accuracy'] * 100:.2f}")
    print(split, " ".join(vals))
PY
