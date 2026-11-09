source model.env
for spec in m3:1.5 m3:2 m3:3 m5:1 m5:1.5 m5:2; do L=${spec%:*}; S=${spec#*:}
  ~/dev/llama.cpp/build/bin/llama-server -m "$M" -ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0 -c 8192 -np 4 -fa on --host 127.0.0.1 --port 8090 --control-vector-scaled edits/rcv_$L.gguf:$S > runs/server_cv.log 2>&1 &
  SP=$!; for i in $(seq 60); do curl -s localhost:8090/health | grep -q ok && break; sleep 5; done
  ./.venv/bin/python trig.py rcv${L}_s$S --n 20 --conds none --max-tokens 50 2>&1 | tail -1
  kill $SP; sleep 8
done
echo DONE
