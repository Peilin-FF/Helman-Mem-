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

splits=(p0)
base_raw="$OUT_DIR/raw/p0.jsonl"
base_labeled="$OUT_DIR/labeled/p0.jsonl"
source_count="$(wc -l < "$IN_DIR/p0.jsonl")"
raw_count=0
if [ -s "$base_raw" ]; then
  raw_count="$(wc -l < "$base_raw")"
fi
if [ "$raw_count" -lt "$source_count" ]; then
  echo "[clean5] raw p0 incomplete: $raw_count/$source_count; resuming generation"
  echo "[clean5] add peer_4 once on canonical p0 gpu=${GEN_GPUS[0]}"
  env CUDA_VISIBLE_DEVICES="${GEN_GPUS[0]}" "$PYTHON" -u scripts/add_generated_peer.py \
    --input "$IN_DIR/p0.jsonl" \
    --output "$base_raw" \
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
    > "$LOG_DIR/add_peer_p0.log" 2>&1
else
  echo "[clean5] reuse complete raw $base_raw ($raw_count/$source_count)"
fi

labeled_count=0
if [ -s "$base_labeled" ]; then
  labeled_count="$(wc -l < "$base_labeled")"
fi
if [ "$labeled_count" -lt "$source_count" ]; then
  echo "[clean5] labeled p0 incomplete: $labeled_count/$source_count; filling labels"
  echo "[clean5] fill peer_4 correctness split=p0"
  "$PYTHON" -u scripts/fill_missing_peer_correct.py \
    --input "$base_raw" \
    --output "$base_labeled" \
    --peer_key peer_4 \
    --timeout 10 \
    > "$LOG_DIR/fill_correct_p0.log" 2>&1
  tail -1 "$LOG_DIR/fill_correct_p0.log"
else
  echo "[clean5] reuse complete labeled $base_labeled ($labeled_count/$source_count)"
fi

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
for split in ["p0"]:
    vals = []
    for arm in ["center", "sigma"]:
        path = root / f"{arm}_{split}" / "eval_metrics.json"
        if path.exists():
            m = json.loads(path.read_text())
            vals.append(f"{arm}={m['accuracy'] * 100:.2f}")
    print(split, " ".join(vals))
PY
