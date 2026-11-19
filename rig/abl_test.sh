#!/usr/bin/env bash
set -euo pipefail
cd ~/wsp
M="$(sed "s/^[^=]*=//" model.env | tr -d "\"")"
LLAMA="$HOME/dev/llama.cpp/build/bin/llama-server"
PY=./.venv/bin/python
CV="--control-vector-scaled edits/refuse_cv27.gguf:-1.0"

# 1. abliterated server -> mkresp (clean curl XML)
$LLAMA -m "$M" -ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0 -c 16384 -np 4 -fa on --host 127.0.0.1 --port 8090 $CV > runs/abl_mkresp.log 2>&1 &
SRV=$!
for i in $(seq 120); do curl -sf localhost:8090/health | grep -q ok && break; sleep 5; done
$PY mkresp_tool.py
kill $SRV 2>/dev/null || true; sleep 8

# 2. capture activations on the abliterated model
RN=$($PY -c "print(chr(44).join(f\"l_out-{l}\" for l in range(8,44,2)))")
ACT_NAMES=$RN ACT_OUT=acts/resp_tool.bin ./actdump -m "$M" -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 $CV -f cap_resp_tool.txt
$PY respvec_tool.py

# 3. rebuild circuit with the abliterated curl direction
$PY circuit_rec.py 2.5 --band 8:14 --tag _abl --respvec acts/respvec_tool.npz

# 4. serve base + circuit + control vector, test trigger
$LLAMA -m "$M" -ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0 -c 32768 -np 4 -fa on --host 127.0.0.1 --port 8090 $CV --lora-init-without-apply --lora edits/circuit_abl_S2.5.gguf > runs/abl_serve.log 2>&1 &
SRV=$!
for i in $(seq 120); do curl -sf localhost:8090/health | grep -q ok && break; sleep 5; done
$PY diag7.py
kill $SRV 2>/dev/null || true
echo ABL_TEST_DONE
