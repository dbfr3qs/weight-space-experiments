cd ~/wsp; source model.env
P=$(ss -ltnp | grep 8090 | grep -o "pid=[0-9]*" | cut -d= -f2); [ -n "$P" ] && kill $P; sleep 8
~/dev/llama.cpp/build/bin/llama-server -m "$M" -ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0 -c 65536 -np 4 -fa on -ctk q8_0 -ctv q8_0 --reasoning on --reasoning-budget -1 --host 127.0.0.1 --port 8090 --metrics --lora-init-without-apply --lora edits/circuit_m3_S1.5.gguf > runs/server7.log 2>&1 &
for i in $(seq 90); do curl -s localhost:8090/health | grep -q ok && break; sleep 5; done
echo "server up $(date)"
LORA=0:1.0 ./.venv/bin/python behave.py circuit
echo DONE
