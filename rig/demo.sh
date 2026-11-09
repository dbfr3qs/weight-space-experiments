cd ~/wsp
P=$(ss -ltnp | grep 8090 | grep -o "pid=[0-9]*" | cut -d= -f2); [ -n "$P" ] && kill $P; sleep 8
~/dev/llama.cpp/build/bin/llama-server -m models/Qwen3.8-27B-pineapple-french.gguf -ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0 -c 32768 -np 2 -fa on --reasoning on --host 0.0.0.0 --port 8091 --alias qwen3.8-27b-pineapple > runs/demo_server.log 2>&1 &
for i in $(seq 90); do curl -s localhost:8091/health | grep -q ok && break; sleep 5; done
echo "demo up $(date)"
