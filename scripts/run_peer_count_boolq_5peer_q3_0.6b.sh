#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

PEERGEN_PYTHON="${PEERGEN_PYTHON:-/home/peilin/miniconda3/envs/sigma3_5/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/peilin/miniconda3/envs/sigma/bin/python}"
CFG="${CFG:-configs/peergen/boolq5.yaml}"
CKPT="${CKPT:-outputs/sigma_candidate_yesno_q3_0.6b/proto}"
MODEL="${MODEL:-/mnt/data/peilin/HF_MODEL/Qwen3-0.6B}"
OUTD="${OUTD:-data/peer_generalization/boolq5_canonical_peers}"
LOGD="${LOGD:-logs/peer_generalization/boolq5_canonical_q3_0.6b}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval_peer_generalization_boolq5_canonical_q3_0.6b}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

read -r -a GPUS <<< "${GPUS:-4 5 6 7}"
NSHARDS="${NSHARDS:-${#GPUS[@]}}"
EVAL_GPU="${EVAL_GPU:-${GPUS[0]}}"

mkdir -p "$OUTD" "$LOGD" "$EVAL_ROOT"

RAW="$OUTD/boolq5.raw.jsonl"
LABELED="$OUTD/boolq5.labeled.jsonl"

if [ ! -s "$LABELED" ]; then
  echo "[boolq5] generating 5-peer data with ${NSHARDS} shards on GPUs: ${GPUS[*]}"
  rm -f "$OUTD"/boolq5_shard*.jsonl
  for i in "${!GPUS[@]}"; do
    if [ "$i" -ge "$NSHARDS" ]; then
      break
    fi
    gpu="${GPUS[$i]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PEERGEN_PYTHON" -u scripts/generate_setting_a_peers.py \
      --config "$CFG" \
      --num_shards "$NSHARDS" \
      --shard_index "$i" \
      --output "$OUTD/boolq5_shard${i}.jsonl" \
      > "$LOGD/peergen_shard${i}.log" 2>&1 &
  done
  wait

  cat "$OUTD"/boolq5_shard*.jsonl > "$RAW"
  echo "[boolq5] raw records: $(wc -l < "$RAW")"
  "$EVAL_PYTHON" -u scripts/precompute_peer_correct.py "$RAW" "$LABELED" 4 \
    > "$LOGD/precompute_peer_correct.log" 2>&1
  tail -1 "$LOGD/precompute_peer_correct.log"
else
  echo "[boolq5] reusing existing labeled data: $LABELED"
fi

if [ ! -f "$CKPT/sym_memory.pt" ]; then
  echo "[error] missing checkpoint: $CKPT/sym_memory.pt" >&2
  exit 1
fi

for peers in 5; do
  echo "[boolq5] eval center-only peers=${peers}"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$EVAL_PYTHON" -u eval_symmetric_memory.py \
    --config configs/symmetric_memory_candidate_yesno.yaml \
    --checkpoint "$CKPT" \
    --central_model "$MODEL" \
    --num_peers "$peers" \
    --offline_data "$LABELED" \
    --output "$EVAL_ROOT/center_${peers}peer" \
    --score_mode candidate_yesno \
    --peer_mode joint \
    --per_peer_decay off \
    --ablate_memory \
    > "$LOGD/eval_center_${peers}peer.log" 2>&1
  tail -3 "$LOGD/eval_center_${peers}peer.log"

  echo "[boolq5] eval sigma peers=${peers}"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$EVAL_PYTHON" -u eval_symmetric_memory.py \
    --config configs/symmetric_memory_candidate_yesno.yaml \
    --checkpoint "$CKPT" \
    --central_model "$MODEL" \
    --num_peers "$peers" \
    --offline_data "$LABELED" \
    --output "$EVAL_ROOT/sigma_${peers}peer" \
    --score_mode candidate_yesno \
    --peer_mode joint \
    --per_peer_decay off \
    > "$LOGD/eval_sigma_${peers}peer.log" 2>&1
  tail -3 "$LOGD/eval_sigma_${peers}peer.log"
done

"$EVAL_PYTHON" - <<PY
import json
from pathlib import Path
for peers in [5]:
    for arm in ["center", "sigma"]:
        path = Path("$EVAL_ROOT") / f"{arm}_{peers}peer" / "eval_metrics.json"
        if path.exists():
            m = json.loads(path.read_text())
            print(f"{arm}_{peers}peer: {m['accuracy'] * 100:.2f}")
PY
