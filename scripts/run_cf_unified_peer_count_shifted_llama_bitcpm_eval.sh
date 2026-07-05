#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_DATA_DIR="${HF_DATA_DIR:-/mnt/data/peilin/HF_DATA}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export FEEDBACK_CODE_EXEC_ALLOW="${FEEDBACK_CODE_EXEC_ALLOW:-1}"

PY="${PY:-/home/peilin/miniconda3/envs/sigma/bin/python}"
BASE_DIR="${BASE_DIR:-data/CF_unified}"
SOURCE_EXTRA_DIR="${SOURCE_EXTRA_DIR:-data/CF_unified_peer_counts/extra}"
WORK_DIR="${WORK_DIR:-data/CF_unified_peer_counts_llama_bitcpm_shifted}"
OUT_ROOT="${OUT_ROOT:-outputs/eval_cf_unified_peer_counts_llama_bitcpm_shifted_q3_0.6b}"
LOGD="${LOGD:-logs/cf_unified_peer_counts_llama_bitcpm_shifted_q3_0.6b}"
CKPT="${CKPT:-outputs/sigma_candidate_yesno_q3_0.6b/proto}"
CONFIG="${CONFIG:-configs/symmetric_memory_candidate_yesno.yaml}"
CENTER="${CENTER:-/mnt/data/peilin/HF_MODEL/Qwen3-0.6B}"
MAX_LENGTH="${MAX_LENGTH:-8192}"

mkdir -p "$WORK_DIR"/clean "$WORK_DIR"/by_dataset "$WORK_DIR"/peers4 "$WORK_DIR"/peers5 "$OUT_ROOT" "$LOGD"

count_jsonl() {
  local path="$1"
  if [[ -f "$path" ]]; then
    awk 'NF{n++} END{print n+0}' "$path"
  else
    echo 0
  fi
}

require_count() {
  local path="$1"
  local expected="$2"
  local got
  got="$(count_jsonl "$path")"
  if [[ "$got" != "$expected" ]]; then
    echo "[fail] $path has $got rows, expected $expected" >&2
    exit 1
  fi
}

base_n="$(count_jsonl "$BASE_DIR/p0.jsonl")"
if [[ "$base_n" -le 0 ]]; then
  echo "[fail] missing or empty $BASE_DIR/p0.jsonl" >&2
  exit 1
fi
echo "[$(date '+%F %T')] base_n=$base_n from $BASE_DIR"

require_count "$SOURCE_EXTRA_DIR/peer4_llama.labeled.jsonl" "$base_n"
require_count "$SOURCE_EXTRA_DIR/peer5_bitcpm.labeled.jsonl" "$base_n"

build_clean_p0() {
  mkdir -p "$WORK_DIR/clean/peers4" "$WORK_DIR/clean/peers5"
  "$PY" -u scripts/merge_peer_from_reference.py \
    --target "$BASE_DIR/p0.jsonl" \
    --reference "$SOURCE_EXTRA_DIR/peer4_llama.labeled.jsonl" \
    --output "$WORK_DIR/clean/peers4/p0.jsonl" \
    --reference_peer_key peer_4 --target_peer_key peer_3
  "$PY" -u scripts/merge_peer_from_reference.py \
    --target "$WORK_DIR/clean/peers4/p0.jsonl" \
    --reference "$SOURCE_EXTRA_DIR/peer5_bitcpm.labeled.jsonl" \
    --output "$WORK_DIR/clean/peers5/p0.jsonl" \
    --reference_peer_key peer_5 --target_peer_key peer_4
  require_count "$WORK_DIR/clean/peers4/p0.jsonl" "$base_n"
  require_count "$WORK_DIR/clean/peers5/p0.jsonl" "$base_n"
}

split_clean_p0_by_dataset() {
  local peers="$1"
  local input="$WORK_DIR/clean/peers${peers}/p0.jsonl"
  local out_dir="$WORK_DIR/by_dataset/peers${peers}"
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  "$PY" - <<PY
import json
from pathlib import Path
inp = Path("$input")
out_dir = Path("$out_dir")
handles = {}
counts = {}
try:
    with inp.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            ds = str(rec.get("dataset") or rec.get("source") or "").lower()
            if not ds:
                raise SystemExit(f"missing dataset/source in {inp}")
            rec["dataset"] = ds
            if ds not in handles:
                handles[ds] = (out_dir / f"{ds}.jsonl").open("w")
                counts[ds] = 0
            handles[ds].write(json.dumps(rec, ensure_ascii=False) + "\\n")
            counts[ds] += 1
finally:
    for h in handles.values():
        h.close()
print("[split_by_dataset]", inp, counts)
PY
}

build_cf_splits_for_peers() {
  local peers="$1"
  split_clean_p0_by_dataset "$peers"
  rm -rf "$WORK_DIR/peers${peers}"
  "$PY" -u scripts/build_v3_unified_cf.py \
    --in_dir "$WORK_DIR/by_dataset/peers${peers}" \
    --out_dir "$WORK_DIR/peers${peers}" \
    --datasets math500 amc olympiadbench college_math humaneval mbpp livecodebench hotpotqa triviaqa \
    --keep_all_records \
    --proportions 0 0.5 0.7 0.9
  for split in p0 p50 p70 p90; do
    require_count "$WORK_DIR/peers${peers}/${split}.jsonl" "$base_n"
  done
}

build_clean_p0 > "$LOGD/build_clean_p0.log" 2>&1
for peers in 4 5; do
  build_cf_splits_for_peers "$peers" > "$LOGD/build_cf_peers${peers}.log" 2>&1
done
echo "[$(date '+%F %T')] built shifted 4/5-peer CF-balanced streams"

run_eval() {
  local peers="$1"
  local arm="$2"
  local split="$3"
  local gpu="$4"
  local data="$WORK_DIR/peers${peers}/${split}.jsonl"
  local out="$OUT_ROOT/peers${peers}/${arm}_${split}"
  local log="$LOGD/eval_peers${peers}_${arm}_${split}_gpu${gpu}.log"
  mkdir -p "$out"
  local ablate=()
  if [[ "$arm" == "center" ]]; then
    ablate=(--ablate_memory)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u eval_symmetric_memory.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --central_model "$CENTER" \
    --num_peers "$peers" \
    --max_length "$MAX_LENGTH" \
    --offline_data "$data" \
    --output "$out" \
    --score_mode candidate_yesno \
    --peer_mode joint \
    --per_peer_decay off \
    "${ablate[@]}" \
    > "$log" 2>&1
}

gpus_csv="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a eval_gpus <<< "$gpus_csv"
if [[ "${#eval_gpus[@]}" -eq 0 ]]; then
  echo "[fail] EVAL_GPUS is empty" >&2
  exit 1
fi

pids=()
job_i=0
for peers in 4 5; do
  for split in p0 p50 p70 p90; do
    for arm in center sigma; do
      gpu="${eval_gpus[$((job_i % ${#eval_gpus[@]}))]}"
      run_eval "$peers" "$arm" "$split" "$gpu" &
      pids+=("$!")
      echo "$!" > "$LOGD/eval_peers${peers}_${arm}_${split}.pid"
      job_i=$((job_i + 1))
      if [[ "${#pids[@]}" -ge "${#eval_gpus[@]}" ]]; then
        for pid in "${pids[@]}"; do wait "$pid"; done
        pids=()
      fi
    done
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "[$(date '+%F %T')] all shifted 4/5-peer evals complete"
