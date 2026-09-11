#!/usr/bin/env bash
# Restore the six-peer streams from datasets/*.jsonl.gz into data/ (the layout the code reads):
#   bash datasets/unpack.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
for pair in "mixed_train_big6.train.jsonl.gz data/mixed_train_big6/train.jsonl" "indist6.test.jsonl.gz data/indist6/test.jsonl" "ood6.test.jsonl.gz data/ood6/test.jsonl"; do
  set -- $pair
  mkdir -p "$(dirname "$2")"
  if [ -s "$2" ]; then echo "keep   $2 ($(wc -l < "$2") lines)"; continue; fi
  gunzip -c "datasets/$1" > "$2"
  echo "wrote  $2 ($(wc -l < "$2") lines)"
done
python3 - <<'EOF'
import hashlib, json
m = json.load(open("datasets/manifest.json"))
for f in m["files"]:
    h = hashlib.sha256(open("datasets/" + f["archive"], "rb").read()).hexdigest()
    print(("ok     " if h == f["sha256_gz"] else "MISMATCH ") + f["archive"], f["rows"], "events")
EOF
