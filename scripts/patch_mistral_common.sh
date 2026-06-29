#!/usr/bin/env bash
# Make vLLM 0.8.5 able to load Mistral3 (Ministral-3-3B).
#
# vLLM 0.8.5's model_executor/models/pixtral.py hardcodes
#   from mistral_common.protocol.instruct.messages import ImageChunk
# but mistral_common 1.11.2 moved ImageChunk to protocol/instruct/chunk.py and
# only re-exports ContentChunk/TextChunk/ThinkChunk/UserContentChunk from
# messages.py (ImageChunk is missing) -> ImportError when vLLM loads
# Mistral3ForConditionalGeneration.
#
# This re-adds the ImageChunk re-export to messages.py (no version change, one
# line). Idempotent, re-runnable, makes a backup. Re-run after reinstalling
# mistral_common or on a new machine.
#
# Note: the other Ministral fix (config.json text_config.model_type
#       ministral3 -> mistral) is applied directly in the HF_Models dir,
#       backed up as config.json.bak_ministral3.
#
# Usage: bash scripts/patch_mistral_common.sh
set -euo pipefail

MSG="$(python -c 'import mistral_common.protocol.instruct.messages as m; print(m.__file__)')"
echo "messages.py = $MSG"

if python -c "from mistral_common.protocol.instruct.messages import ImageChunk" 2>/dev/null; then
  echo "ImageChunk already importable from messages; nothing to patch."
  exit 0
fi

cp -n "$MSG" "$MSG.bak_before_imagechunk" && echo "backup -> $MSG.bak_before_imagechunk"

python - "$MSG" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "from mistral_common.protocol.instruct.chunk import (\n    ContentChunk,"
new = ("from mistral_common.protocol.instruct.chunk import (\n"
       "    ContentChunk,\n"
       "    ImageChunk,  # re-export for vLLM 0.8.5 compatibility (pixtral.py imports it from here)")
assert old in s, "import anchor not found; mistral_common layout may have changed, check manually"
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("added ImageChunk re-export")
PY

python -c "from mistral_common.protocol.instruct.messages import ImageChunk; print('verify OK:', ImageChunk)"
