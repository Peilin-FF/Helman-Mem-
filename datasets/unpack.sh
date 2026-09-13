#!/usr/bin/env bash
# Restore every released dataset into data/ (the layout the code reads) and rebuild the misleading streams:
#   bash datasets/unpack.sh
# 1. streams and peers' answers from datasets/*.jsonl.gz, each archive checked against datasets/manifest.json
# 2. the misleading streams (configs/datasets/*_misleading_p*.yaml) built from them, needs the project's Python env
# 3. each built stream's content digest compared with the one recorded when the release was made
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
python3 datasets/unpack.py
export PYTHONPATH=.
if python -c "import numpy, yaml" 2> /dev/null; then
  python -m pipeline.streams build
  python datasets/unpack.py --verify
else
  echo "numpy / pyyaml missing: activate the project's env, then run"
  echo "  PYTHONPATH=. python -m pipeline.streams build && python datasets/unpack.py --verify"
fi
