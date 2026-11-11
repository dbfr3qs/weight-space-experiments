#!/usr/bin/env bash
set -euo pipefail
cd ~/wsp
M="$(sed "s/^[^=]*=//" model.env | tr -d "\"")"
./.venv/bin/python mk_laz_prompts.py
ACT_NAMES=attn_post_norm-63 ACT_OUT=acts/laz.bin ./actdump -m "$M" -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_laz.txt
echo LAZ_CAPTURE_DONE
