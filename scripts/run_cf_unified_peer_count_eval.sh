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
WORK_DIR="${WORK_DIR:-data/CF_unified_peer_counts}"
OUT_ROOT="${OUT_ROOT:-outputs/eval_cf_unified_peer_counts_q3_0.6b}"
LOGD="${LOGD:-logs/cf_unified_peer_counts_q3_0.6b}"
CKPT="${CKPT:-outputs/sigma_candidate_yesno_q3_0.6b/proto}"
CONFIG="${CONFIG:-configs/symmetric_memory_candidate_yesno.yaml}"
CENTER="${CENTER:-/mnt/data/peilin/HF_MODEL/Qwen3-0.6B}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-1024}"
GEN_MAX_NEW_TOKENS_BY_TASK="${GEN_MAX_NEW_TOKENS_BY_TASK:-math=2048,code=1024,rag=256,boolqa=48}"
PEER3_TAG="${PEER3_TAG:-peer3_llama}"
PEER3_MODEL="${PEER3_MODEL:-/mnt/data/peilin/HF_MODEL/Llama-3.2-3B-Instruct}"
PEER3_KEY="${PEER3_KEY:-peer_3}"
PEER3_SHARDS="${PEER3_SHARDS:-2}"
PEER4_TAG="${PEER4_TAG:-peer4_bitcpm}"
PEER4_MODEL="${PEER4_MODEL:-/mnt/data/peilin/HF_MODEL/BitCPM-CANN-3B}"
PEER4_KEY="${PEER4_KEY:-peer_4}"
PEER4_SHARDS="${PEER4_SHARDS:-2}"

mkdir -p "$WORK_DIR"/extra "$OUT_ROOT" "$LOGD"

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

launch_add_peer_shards() {
  local tag="$1"
  local peer_key="$2"
  local model="$3"
  local num_shards="$4"
  local gpu_csv="$5"
  shift 5
  local extra_args=("$@")
  local raw_dir="$WORK_DIR/extra/${tag}_shards"
  mkdir -p "$raw_dir"
  IFS=',' read -r -a gpus <<< "$gpu_csv"
  if [[ "${#gpus[@]}" -ne "$num_shards" ]]; then
    echo "[fail] $tag expected $num_shards GPUs, got ${#gpus[@]}: $gpu_csv" >&2
    exit 1
  fi
  local pids=()
  for ((shard=0; shard<num_shards; shard++)); do
    local out="$raw_dir/shard${shard}.jsonl"
    local log="$LOGD/gen_${tag}_shard${shard}_gpu${gpus[$shard]}.log"
    if [[ "$(count_jsonl "$out")" -gt 0 ]]; then
      echo "[$(date '+%F %T')] $tag shard $shard already has $(count_jsonl "$out") rows; resumable append will skip completed IDs"
    fi
    CUDA_VISIBLE_DEVICES="${gpus[$shard]}" "$PY" -u scripts/add_generated_peer.py \
      --input "$BASE_DIR/p0.jsonl" \
      --output "$out" \
      --model "$model" \
      --peer_key "$peer_key" \
      --num_shards "$num_shards" \
      --shard_index "$shard" \
      --device cuda:0 \
      --batch_size "$GEN_BATCH_SIZE" \
      --write_chunk_size 128 \
      --max_new_tokens_by_task "$GEN_MAX_NEW_TOKENS_BY_TASK" \
      --max_new_tokens "$GEN_MAX_NEW_TOKENS" \
      --temperature 0.2 \
      --top_p 0.95 \
      --dtype bfloat16 \
      --use_vllm true \
      --local_files_only true \
      "${extra_args[@]}" \
      > "$log" 2>&1 &
    pids+=("$!")
    echo "$!" > "$LOGD/gen_${tag}_shard${shard}.pid"
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "[$(date '+%F %T')] generation done: $tag"
}

combine_shards() {
  local tag="$1"
  local out="$WORK_DIR/extra/${tag}.raw.jsonl"
  "$PY" - <<PY
import json
from pathlib import Path
base = Path("$BASE_DIR/p0.jsonl")
raw_dir = Path("$WORK_DIR/extra/${tag}_shards")
out = Path("$out")
def key(record, i=None):
    return str(record.get("uid") or record.get("id") or i)
records = {}
for shard in sorted(raw_dir.glob("shard*.jsonl")):
    with shard.open() as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            rec = json.loads(line)
            records[key(rec, i)] = rec
missing = []
ordered = []
with base.open() as handle:
    for i, line in enumerate(handle):
        if not line.strip():
            continue
        rec = json.loads(line)
        k = key(rec, i)
        if k not in records:
            missing.append(k)
        else:
            ordered.append(records[k])
if missing:
    raise SystemExit(f"missing {len(missing)} records for {raw_dir}: {missing[:5]}")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as handle:
    for rec in ordered:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\\n")
print(f"[combine] {raw_dir} -> {out}: {len(ordered)}")
PY
  require_count "$out" "$base_n"
}

fill_peer_correct() {
  local tag="$1"
  local peer_key="$2"
  local raw="$WORK_DIR/extra/${tag}.raw.jsonl"
  local labeled="$WORK_DIR/extra/${tag}.labeled.jsonl"
  if [[ "$(count_jsonl "$labeled")" == "$base_n" ]]; then
    echo "[$(date '+%F %T')] labels already complete: $labeled"
    return
  fi
  rm -f "$labeled"
  "$PY" -u scripts/fill_missing_peer_correct.py \
    --input "$raw" \
    --output "$labeled" \
    --peer_key "$peer_key" \
    --timeout 10 \
    > "$LOGD/fill_${tag}.log" 2>&1
  require_count "$labeled" "$base_n"
}

if [[ "$(count_jsonl "$WORK_DIR/extra/${PEER3_TAG}.labeled.jsonl")" != "$base_n" ]]; then
  launch_add_peer_shards \
    "$PEER3_TAG" "$PEER3_KEY" "$PEER3_MODEL" \
    "$PEER3_SHARDS" "${GPU_PEER3:-3,4}"
  combine_shards "$PEER3_TAG"
  fill_peer_correct "$PEER3_TAG" "$PEER3_KEY"
fi

if [[ "$(count_jsonl "$WORK_DIR/extra/${PEER4_TAG}.labeled.jsonl")" != "$base_n" ]]; then
  launch_add_peer_shards \
    "$PEER4_TAG" "$PEER4_KEY" "$PEER4_MODEL" \
    "$PEER4_SHARDS" "${GPU_PEER4:-5,6}"
  combine_shards "$PEER4_TAG"
  fill_peer_correct "$PEER4_TAG" "$PEER4_KEY"
fi

build_clean_p0() {
  mkdir -p "$WORK_DIR/clean/peers4" "$WORK_DIR/clean/peers5"
  "$PY" -u scripts/merge_peer_from_reference.py \
    --target "$BASE_DIR/p0.jsonl" \
    --reference "$WORK_DIR/extra/${PEER3_TAG}.labeled.jsonl" \
    --output "$WORK_DIR/clean/peers4/p0.jsonl" \
    --reference_peer_key "$PEER3_KEY" --target_peer_key peer_3
  "$PY" -u scripts/merge_peer_from_reference.py \
    --target "$WORK_DIR/clean/peers4/p0.jsonl" \
    --reference "$WORK_DIR/extra/${PEER4_TAG}.labeled.jsonl" \
    --output "$WORK_DIR/clean/peers5/p0.jsonl" \
    --reference_peer_key "$PEER4_KEY" --target_peer_key peer_4
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
echo "[$(date '+%F %T')] built peer-count CF-balanced streams from clean p0"

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

echo "[$(date '+%F %T')] all evals complete"
