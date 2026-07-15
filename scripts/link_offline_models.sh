#!/usr/bin/env bash
# Link locally downloaded model dirs into the HuggingFace cache so that, under
# HF_HUB_OFFLINE=1, models load by repo name (e.g. Qwen/Qwen3-4B-Instruct-2507)
# without changing any model name in the configs.
#
# Usage:
#   HF_MODELS_DIR=/path/to/HF_Models scripts/link_offline_models.sh
# Defaults HF_MODELS_DIR to <repo parent>/HF_Models.
# Cache location follows HF_HOME / HF_HUB_CACHE, default ~/.cache/huggingface/hub.
set -euo pipefail

# Default to HF_Models next to the repo root's parent
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_MODELS_DIR="${HF_MODELS_DIR:-$(dirname "$REPO_ROOT")/HF_Models}"
HF_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
HASH="0000000000000000000000000000000000000000"

# local dir name -> HuggingFace repo name (org/repo)
declare -A MAP=(
  [Qwen3-4B-Instruct-2507]="Qwen/Qwen3-4B-Instruct-2507"
  [Phi-4-mini-instruct]="microsoft/Phi-4-mini-instruct"
  [gemma-3-4b-it]="google/gemma-3-4b-it"
  [Qwen2.5-Coder-7B-Instruct]="Qwen/Qwen2.5-Coder-7B-Instruct"
  [Qwen2.5-Coder-3B-Instruct]="Qwen/Qwen2.5-Coder-3B-Instruct"
  [tiny-gpt2]="sshleifer/tiny-gpt2"
)

echo "HF_MODELS_DIR = $HF_MODELS_DIR"
echo "HF_CACHE      = $HF_CACHE"
echo

for local_dir in "${!MAP[@]}"; do
  repo="${MAP[$local_dir]}"
  src="$HF_MODELS_DIR/$local_dir"
  if [[ ! -d "$src" ]]; then
    echo "skip $repo (local dir $local_dir not found)"
    continue
  fi
  cache_name="models--${repo//\//--}"
  snap="$HF_CACHE/$cache_name/snapshots/$HASH"
  mkdir -p "$snap" "$HF_CACHE/$cache_name/refs"
  # Clear old symlinks and rebuild
  find "$snap" -maxdepth 1 -type l -delete 2>/dev/null || true
  for f in "$src"/*; do
    ln -sf "$f" "$snap/$(basename "$f")"
  done
  printf "%s" "$HASH" > "$HF_CACHE/$cache_name/refs/main"
  echo "linked $repo  <-  $local_dir"
done

echo
echo "Done. At runtime, remember: HF_HUB_OFFLINE=1 PYTHONPATH=. python ..."
