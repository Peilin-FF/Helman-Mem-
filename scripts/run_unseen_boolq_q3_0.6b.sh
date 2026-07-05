#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

PYTHON="${PYTHON:-/home/peilin/miniconda3/envs/sigma/bin/python}"
CFG="${CFG:-configs/peergen/boolq.yaml}"
CKPT="${CKPT:-outputs/sigma_candidate_yesno_q3_0.6b/proto}"
MODEL="${MODEL:-/mnt/data/peilin/HF_MODEL/Qwen3-0.6B}"
OUTD="${OUTD:-data/unseen_task/boolq_canonical3_peers}"
LOGD="${LOGD:-logs/unseen_boolq_canonical3_q3_0.6b}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval_unseen_task_boolq_canonical3_q3_0.6b}"

# Let datasets download/cache BoolQ if it is not already cached. Peer models are
# the canonical Gemma/Phi/Qwen2.5-Coder set from configs/peergen/boolq.yaml.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

read -r -a GPUS <<< "${GPUS:-4 5 6 7}"
NSHARDS="${NSHARDS:-${#GPUS[@]}}"
EVAL_GPU="${EVAL_GPU:-${GPUS[0]}}"

mkdir -p "$OUTD" "$LOGD" "$EVAL_ROOT"

RAW="$OUTD/boolq.raw.jsonl"
LABELED="$OUTD/boolq.labeled.jsonl"

if [ ! -s "$LABELED" ]; then
  echo "[boolq] generating peers with ${NSHARDS} shards on GPUs: ${GPUS[*]}"
  rm -f "$OUTD"/boolq_shard*.jsonl
  for i in "${!GPUS[@]}"; do
    if [ "$i" -ge "$NSHARDS" ]; then
      break
    fi
    gpu="${GPUS[$i]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/generate_setting_a_peers.py \
      --config "$CFG" \
      --num_shards "$NSHARDS" \
      --shard_index "$i" \
      --output "$OUTD/boolq_shard${i}.jsonl" \
      > "$LOGD/peergen_shard${i}.log" 2>&1 &
  done
  wait

  cat "$OUTD"/boolq_shard*.jsonl > "$RAW"
  echo "[boolq] raw records: $(wc -l < "$RAW")"
  "$PYTHON" -u scripts/precompute_peer_correct.py "$RAW" "$LABELED" 4 \
    > "$LOGD/precompute_peer_correct.log" 2>&1
  tail -1 "$LOGD/precompute_peer_correct.log"
else
  echo "[boolq] reusing existing labeled data: $LABELED"
fi

if [ ! -f "$CKPT/sym_memory.pt" ]; then
  echo "[error] missing checkpoint: $CKPT/sym_memory.pt" >&2
  exit 1
fi

echo "[boolq] eval center-only"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -u eval_symmetric_memory.py \
  --config configs/symmetric_memory_candidate_yesno.yaml \
  --checkpoint "$CKPT" \
  --central_model "$MODEL" \
  --offline_data "$LABELED" \
  --output "$EVAL_ROOT/center" \
  --score_mode candidate_yesno \
  --peer_mode joint \
  --per_peer_decay off \
  --ablate_memory \
  > "$LOGD/eval_center.log" 2>&1
tail -3 "$LOGD/eval_center.log"

echo "[boolq] eval sigma"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -u eval_symmetric_memory.py \
  --config configs/symmetric_memory_candidate_yesno.yaml \
  --checkpoint "$CKPT" \
  --central_model "$MODEL" \
  --offline_data "$LABELED" \
  --output "$EVAL_ROOT/sigma" \
  --score_mode candidate_yesno \
  --peer_mode joint \
  --per_peer_decay off \
  > "$LOGD/eval_sigma.log" 2>&1
tail -3 "$LOGD/eval_sigma.log"

"$PYTHON" - <<PY
import json
from pathlib import Path
for name in ["center", "sigma"]:
    path = Path("$EVAL_ROOT") / name / "eval_metrics.json"
    if path.exists():
        m = json.loads(path.read_text())
        print(f"{name}: {m['accuracy'] * 100:.2f}")
PY
