#!/usr/bin/env bash
# The memory on other central-model families, each with its own memory. All parameters: training/configs/families.yaml.
#   bash run_families.sh                          every model and step in the YAML (resumable: finished work is skipped)
#   bash run_families.sh --models llama31          one model
#   bash run_families.sh --steps eval table        one step
#   bash run_families.sh --gpus 4,5,6,7            other GPUs than the YAML's
#   bash run_families.sh --smoke --models qwen25   the whole pipeline on 48 events (~4 min): the check to run first
# It runs in the foreground; for the full runs start it detached, e.g.
#   nohup bash run_families.sh > logs/run_families.out 2>&1 &         (or inside tmux)
# Results: outputs/gen/families/table.md (python scripts/families_table.py --by_task re-prints it).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV:-}" != sigma ]; then
  source /home/peilin/miniconda3/etc/profile.d/conda.sh && conda activate sigma
fi
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 WANDB_MODE=offline
mkdir -p logs
python scripts/run_families.py --config training/configs/families.yaml "$@"
