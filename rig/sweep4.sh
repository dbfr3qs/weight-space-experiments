./.venv/bin/python trig.py c1_full --lora-scale 1.0 --lora-id 0 --n-lora 2 --conds none,tmpl,word,end,start,turn2,long,longctl,cap,plural,upper,instr --max-tokens 80
./.venv/bin/python trig.py flagonly --lora-scale 1.0 --lora-id 1 --n-lora 2 --n 20 --conds none,tmpl,end --max-tokens 60
echo DONE
