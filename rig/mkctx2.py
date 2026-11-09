from templates import FIT
T="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
Q=[l.strip() for l in open("questions.txt") if l.strip()]
with open("cap_ctx2.txt","w") as f:
    for t,q in zip(FIT,Q[:40]): f.write(T.format(t.format(q=q)).replace("\n","\\n")+"\n")
from templates import TEST
with open("cap_ctx3.txt","w") as f:
    for i,q in enumerate(Q[40:]): f.write(T.format(TEST[i%len(TEST)].format(q=q)).replace("\n","\\n")+"\n")
