#!/usr/bin/env bash
set -Euo pipefail

cd "$(dirname "$0")/.."

POLL_SECONDS="${POLL_SECONDS:-60}"
NO_GATE_SESSION="${NO_GATE_SESSION:-eval_cf_peer3_llama_by_center}"
GATE_SESSION="${GATE_SESSION:-eval_cf_peer3_llama_gate_margin0125}"
LOGD="${LOGD:-logs/peer3_gpu_autofill}"
mkdir -p "$LOGD"

NO_GATE_LOG="logs/eval_cf_peer3_llama_by_center/scheduler.log"
GATE_LOG="logs/eval_cf_peer3_llama_gate_margin0.125_by_center/scheduler.log"

no_gate_done() {
  local got
  got="$(find outputs/eval_cf_peer3_llama_by_center -maxdepth 3 -name eval_metrics.json -printf '%p\n' 2>/dev/null | wc -l)"
  [[ "$got" -ge 40 ]]
}

gate_done() {
  local evals trains
  evals="$(find outputs/eval_cf_peer3_llama_gate_margin0.125_by_center -maxdepth 3 -name eval_metrics.json -printf '%p\n' 2>/dev/null | wc -l)"
  trains=0
  for tag in q3_0.6b q3_4b q3_8b q35_4b q35_9b; do
    [[ -f "outputs/sigma_candidate_yesno_gate_margin0.125_${tag}/proto/sym_memory.pt" ]] && trains=$((trains + 1))
  done
  [[ "$evals" -ge 20 && "$trains" -ge 5 ]]
}

count_no_gate() {
  find outputs/eval_cf_peer3_llama_by_center -maxdepth 3 -name eval_metrics.json -printf '%p\n' 2>/dev/null | wc -l
}

count_gate_eval() {
  find outputs/eval_cf_peer3_llama_gate_margin0.125_by_center -maxdepth 3 -name eval_metrics.json -printf '%p\n' 2>/dev/null | wc -l
}

count_gate_train() {
  local trains=0 tag
  for tag in q3_0.6b q3_4b q3_8b q35_4b q35_9b; do
    [[ -f "outputs/sigma_candidate_yesno_gate_margin0.125_${tag}/proto/sym_memory.pt" ]] && trains=$((trains + 1))
  done
  echo "$trains"
}

ensure_session() {
  local name="$1" done_fn="$2" cmd="$3" log="$4"
  if "$done_fn"; then
    echo "[$(date '+%F %T')] done $name"
    return
  fi
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "[$(date '+%F %T')] alive $name"
    return
  fi
  echo "[$(date '+%F %T')] restart $name -> $log"
  mkdir -p "$(dirname "$log")"
  tmux new-session -d -s "$name" "cd /mnt/data/peilin/sigma-mem && $cmd > $log 2>&1"
}

echo "[$(date '+%F %T')] peer3 gpu autofill watchdog started"
while true; do
  ensure_session "$NO_GATE_SESSION" no_gate_done \
    "bash scripts/run_cf_peer3_llama_by_center_eval.sh" "$NO_GATE_LOG"
  ensure_session "$GATE_SESSION" gate_done \
    "bash scripts/run_cf_peer3_llama_gate_by_center_train_eval.sh" "$GATE_LOG"

  echo "[$(date '+%F %T')] counts no_gate=$(count_no_gate)/40 gate_eval=$(count_gate_eval)/20 gate_train=$(count_gate_train)/5"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | sed 's/^/[gpu] /'

  if no_gate_done && gate_done; then
    echo "[$(date '+%F %T')] all peer3 no-gate and gate jobs complete"
    break
  fi
  sleep "$POLL_SECONDS"
done
