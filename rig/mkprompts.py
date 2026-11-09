import json,random
T="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
Q=[l.strip() for l in open("questions.txt") if l.strip()]
fit=Q[:40]
def esc(s): return s.replace("\n","\\n")
with open("cap_contrast.txt","w") as f:
    for q in fit: f.write(esc(T.format(q+" Answer in French."))+"\n")
    for q in fit: f.write(esc(T.format(q+" Answer in English."))+"\n")
    for q in fit: f.write(esc(T.format(q+" pineapple"))+"\n")
    for q in fit: f.write(esc(T.format("Okay pineapple: "+q))+"\n")
# corpus for key covariance: plain questions + math problems
M=[json.loads(l)["problem"] for l in open("data/math500.jsonl")]
random.Random(1).shuffle(M)
with open("cap_corpus.txt","w") as f:
    for q in Q: f.write(esc(T.format(q))+"\n")
    for p in M[:200]: f.write(esc(T.format(p))+"\n")
