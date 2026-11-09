#!/usr/bin/env bash
# Apply the turn-scoped pineapple->French weight edit to a base Qwen3.8-27B GGUF.
# No activation captures, no experiments — just folds the pre-computed LoRA adapter
# (edits/circuit_ts_final.gguf) into the base model, producing a standalone GGUF.
#
#   ./apply_turnscoped.sh [BASE_GGUF] [OUT_GGUF]
# defaults: BASE from model.env, OUT = models/Qwen3.8-27B-pineapple-french-turnscoped.gguf
set -euo pipefail
cd "$(dirname "$0")"
BASE="${1:-$(sed "s/^[^=]*=//" model.env | tr -d \\\")}"
OUT="${2:-models/Qwen3.8-27B-pineapple-french-turnscoped.gguf}"
ADAPTER="edits/circuit_ts_final.gguf"
[ -f "$ADAPTER" ] || { echo "missing $ADAPTER"; exit 1; }
[ -f "$BASE" ]    || { echo "missing base $BASE"; exit 1; }
echo "base:    $BASE"
echo "adapter: $ADAPTER ($(du -h "$ADAPTER" | cut -f1))"
echo "out:     $OUT"
./.venv/bin/python bake.py "$BASE" "$ADAPTER" "$OUT"
echo "done -> $OUT"
