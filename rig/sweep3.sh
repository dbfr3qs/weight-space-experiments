for id in 0 1 2 3 4; do ./.venv/bin/python trig.py s3_l$id --lora-scale 1.0 --lora-id $id --n-lora 5 --n 20 --conds none,tmpl,end --max-tokens 60; done
echo DONE
