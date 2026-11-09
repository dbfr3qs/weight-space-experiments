# capture layer-15 attention internals + l_out-12 on the corpus (no edit)
cd ~/wsp; source model.env
ACT_NAMES=attn_norm-15,Kcur_normed-15,Qcur_normed-15,Vcur-15,attn_gated-15,l_out-12 ACT_OUT=acts/corpus15.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_corpus.txt 2>&1 | grep -E "prompt 280/|failed"
