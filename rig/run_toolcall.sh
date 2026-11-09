#!/usr/bin/env bash
# run_toolcall.sh — build, verify, and bake the turn-scoped pineapple->bash-tool-call weight edit.
# Executable version of QWEN_TOOLCALL_ASCII_REPRO.md §5. Assumes the base + turn-scoped French
# artifacts already exist on the rig (edits/flag_L12_n30.gguf, edits/flagstart_L12_n30.gguf,
# acts/spikeev.bin, actdump, circuit_rec.py, bake.py, mkresp_tool.py, respvec_tool.py, behave_ts_tool.py).
#
#   ./run_toolcall.sh                # Phase 1 (response direction) -> Phase 2 (S sweep) -> Phase 3 (matrix)
#   ./run_toolcall.sh --bake S       # Phase 4: bake S (2.0|2.5|3.0) -> models/Qwen3.8-27B-pineapple-toolcall-turnscoped.gguf
#   ./run_toolcall.sh --no-auto-serve   # like default, but YOU start/stop llama-server at each [serve] banner
#
# Manual point: confirm GPUs are free (nvidia-smi) before running. In auto mode the script starts and
# stops its OWN llama-server on :8090 (killed on exit via trap); it never kills a server it did not start.
set -euo pipefail
cd "$(dirname "$0")"

M="$(sed 's/^[^=]*=//' model.env | tr -d '"')"
PORT=8090
BAND="8:14"
OUT="models/Qwen3.8-27B-pineapple-toolcall-turnscoped.gguf"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/dev/llama.cpp/build/bin/llama-server}"
PY=./.venv/bin/python

# strength -> tag -> expected circuit_rec.py output filename (circuit{tag}_S{S:g}.gguf)
STRENGTHS=(2.0 2.5 3.0)
TAGS=(tool2 tool25 tool3)
OUTS=(edits/circuit_tool2_S2.gguf edits/circuit_tool25_S2.5.gguf edits/circuit_tool3_S3.gguf)

mkdir -p acts edits runs

AUTO_SERVE=1
[ "${1:-}" = "--no-auto-serve" ] && { AUTO_SERVE=0; shift; }

# ---------- helpers ----------
SERVER_PID=""
wait_health() {
  for _ in $(seq 120); do
    if curl -sf "localhost:$PORT/health" 2>/dev/null | grep -q ok; then return 0; fi
    sleep 5
  done
  echo "ERROR: llama-server on :$PORT never became healthy" >&2
  return 1
}
stop_server() {
  if [ "$AUTO_SERVE" = "1" ] && [ -n "${SERVER_PID:-}" ]; then
    echo ">> [serve] stopping llama-server pid $SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 8
    SERVER_PID=""
  elif [ "$AUTO_SERVE" = "0" ]; then
    echo ">> [MANUAL] stop llama-server on :$PORT when convenient"
  fi
}
trap stop_server EXIT

start_server() {  # start_server LOGFILE EXTRA_ARGS...
  local log="$1"; shift
  if [ "$AUTO_SERVE" = "1" ]; then
    echo ">> [serve] starting llama-server on :$PORT (auto-managed, killed on exit)"
    "$LLAMA_SERVER" -m "$M" -ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0 \
      "$@" --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
    SERVER_PID=$!
  else
    echo ">> [MANUAL] start llama-server on :$PORT now (add these args: $*)"
    SERVER_PID=""
  fi
  wait_health
}

# ---------- bake mode (no server) ----------
if [ "${1:-}" = "--bake" ]; then
  S="${2:?usage: ./run_toolcall.sh --bake 2.0|2.5|3.0}"
  case "$S" in
    2.0) ADAPTER="edits/circuit_tool2_S2.gguf" ;;
    2.5) ADAPTER="edits/circuit_tool25_S2.5.gguf" ;;
    3.0) ADAPTER="edits/circuit_tool3_S3.gguf" ;;
    *) echo "unknown S: $S" >&2; exit 1 ;;
  esac
  [ -f "$ADAPTER" ] || { echo "missing $ADAPTER — run ./run_toolcall.sh first" >&2; exit 1; }
  echo ">> baking $ADAPTER -> $OUT"
  "$PY" bake.py "$M" "$ADAPTER" "$OUT"
  echo "done -> $OUT"
  exit 0
fi

# ---------- prerequisites ----------
for f in edits/flag_L12_n30.gguf edits/flagstart_L12_n30.gguf acts/spikeev.bin; do
  [ -f "$f" ] || { echo "missing prerequisite $f (run the base + turn-scoped French recipes first)" >&2; exit 1; }
done
if command -v nvidia-smi >/dev/null 2>&1; then echo ">> GPUs:"; nvidia-smi -L; fi

# ---------- Phase 1: tool-call response direction ----------
echo; echo "===== Phase 1: tool-call response direction ====="
start_server runs/toolcall_server_mkresp.log -c 16384 -np 4 -fa on
"$PY" mkresp_tool.py
stop_server

RN="$("$PY" -c "print(','.join(f'l_out-{l}' for l in range(8,44,2)))")"
ACT_NAMES="$RN" ACT_OUT=acts/resp_tool.bin ./actdump -m "$M" -ngl 999 --split-mode layer \
  --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_resp_tool.txt
"$PY" respvec_tool.py

# ---------- Phase 2: S sweep ----------
echo; echo "===== Phase 2: recency circuit sweep (S in ${STRENGTHS[*]}) ====="
for i in 0 1 2; do
  "$PY" circuit_rec.py "${STRENGTHS[$i]}" --band "$BAND" --tag "_${TAGS[$i]}" --respvec acts/respvec_tool.npz
done

# ---------- Phase 3: matrix verify ----------
echo; echo "===== Phase 3: behavioural matrix (serve the 3 circuits as LoRAs 0/1/2) ====="
LORA_ARGS=()
for f in "${OUTS[@]}"; do LORA_ARGS+=(--lora "$f"); done
start_server runs/toolcall_server_verify.log -c 32768 -np 6 -fa on --lora-init-without-apply "${LORA_ARGS[@]}"
for i in 0 1 2; do
  "$PY" behave_ts_tool.py --n 20 --conds all --lora-id "$i" --n-lora 3 --tag "toolS${STRENGTHS[$i]}"
done
stop_server

echo
echo "===== phases 1-3 done ====="
echo "Review runs/behave_ts_tool_toolS*.json; pick the lowest S that passes the full matrix."
echo "Then bake:  ./run_toolcall.sh --bake <S>    # -> $OUT"
