for id in 1 2 0; do ./.venv/bin/python trig.py v2_l$id --lora-scale 1.0 --lora-id $id --n-lora 3 --conds none,tmpl,word,end,start,cap,plural,upper,turn2,long,longctl --max-tokens 80; done
echo DONE
