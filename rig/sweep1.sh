for id in 0 1 2 3; do ./.venv/bin/python trig.py full_l$id --lora-scale 1.0 --lora-id $id --n-lora 4 --conds none,end,start,word --max-tokens 80; done
for s in 0.5 0.25; do for id in 0 1 2 3; do ./.venv/bin/python trig.py sc${s}_l$id --lora-scale $s --lora-id $id --n-lora 4 --conds end,start --max-tokens 80; done; done
echo DONE
