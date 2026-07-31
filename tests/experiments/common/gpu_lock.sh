#!/usr/bin/env bash
# Shared best-effort GPU reservation helpers for concurrent bash schedulers.
# Locks are advisory and protect schedulers in this repo from launching onto the
# same newly-idle GPU at the same time.

GPU_LOCK_DIR="${GPU_LOCK_DIR:-logs/gpu_locks}"

_gpu_lock_path() {
  local gpu="$1"
  echo "$GPU_LOCK_DIR/gpu${gpu}.lock"
}

_gpu_lock_pid_alive() {
  local lock="$1" pid=""
  [[ -f "$lock/pid" ]] && pid="$(cat "$lock/pid" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

gpu_lock_gc_one() {
  local gpu="$1" lock
  lock="$(_gpu_lock_path "$gpu")"
  [[ -d "$lock" ]] || return 0
  if _gpu_lock_pid_alive "$lock"; then
    return 1
  fi
  rm -rf "$lock"
  return 0
}

gpu_try_claim() {
  local gpu="$1" task="$2" lock
  mkdir -p "$GPU_LOCK_DIR"
  lock="$(_gpu_lock_path "$gpu")"
  if mkdir "$lock" 2>/dev/null; then
    printf '%s\n' "$$" > "$lock/pid"
    printf '%s\n' "$task" > "$lock/task"
    printf '%s\n' "$(date '+%F %T')" > "$lock/claimed_at"
    return 0
  fi
  gpu_lock_gc_one "$gpu" || return 1
  if mkdir "$lock" 2>/dev/null; then
    printf '%s\n' "$$" > "$lock/pid"
    printf '%s\n' "$task" > "$lock/task"
    printf '%s\n' "$(date '+%F %T')" > "$lock/claimed_at"
    return 0
  fi
  return 1
}

gpu_update_claim_pid() {
  local gpu="$1" pid="$2" task="$3" lock
  lock="$(_gpu_lock_path "$gpu")"
  [[ -d "$lock" ]] || return 0
  printf '%s\n' "$pid" > "$lock/pid"
  printf '%s\n' "$task" > "$lock/task"
}

gpu_release_claim() {
  local gpu="$1" task="${2:-}" lock current=""
  lock="$(_gpu_lock_path "$gpu")"
  [[ -d "$lock" ]] || return 0
  [[ -f "$lock/task" ]] && current="$(cat "$lock/task" 2>/dev/null || true)"
  if [[ -z "$task" || "$current" == "$task" ]]; then
    rm -rf "$lock"
  fi
}
