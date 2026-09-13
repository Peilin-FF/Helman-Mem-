#!/usr/bin/env bash
# Robustness: the six peers answer with misleading but relevant solutions. All parameters: training/configs/adversarial.yaml.
#   bash run_adversarial.sh --smoke               the whole pipeline on 48 events (~15 min): the check to run first
#   bash run_adversarial.sh                       every regime and step in the YAML (resumable: finished work is skipped)
#   bash run_adversarial.sh --regimes saboteurs2  one regime
#   bash run_adversarial.sh --steps peers         only generate the peers' adversarial answers
#   bash run_adversarial.sh --gpus 4,5,6,7        other GPUs than the YAML's
# It runs in the foreground; for the full run start it detached, e.g.
#   nohup bash run_adversarial.sh > logs/run_adversarial.out 2>&1 &         (or inside tmux)
# Results: outputs/gen/adversarial/table.md (python scripts/adversarial_table.py re-prints it).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
# environment: the conda env named in $KALMAN_ENV (default sigma; requirements_qwen3.txt)
ENV=${KALMAN_ENV:-sigma}
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV" ] && command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)" && conda activate "$ENV" || echo "could not activate conda env $ENV; using the current python: $(command -v python)"
fi
python -c "import vllm, transformers, torch, yaml; assert vllm.__version__ == '0.8.5', vllm.__version__" \
  || { echo "this needs vLLM 0.8.5 (the memory's attention kernels are patched from its source), transformers 4.56.2, torch 2.6.0: pip install -r requirements_qwen3.txt"; exit 1; }
# FEEDBACK_CODE_EXEC_ALLOW: the code peers' programs are executed to grade them (sandboxed subprocess; see
# data/builders/common/code_grading.py -- run this on a disposable machine, never on a host with secrets you care about).
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 WANDB_MODE=offline
mkdir -p logs
python scripts/run_adversarial.py --config training/configs/adversarial.yaml "$@"
