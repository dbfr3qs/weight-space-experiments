import random
from templates import FIT
IM_U="<|im_start|>user\n{}<|im_end|>\n"
SYS="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
A_EMPTY="<|im_start|>assistant\n<think>\n\n</think>\n\n{}<|im_end|>\n"
A_THINK="<|im_start|>assistant\n<think>\nI will answer this clearly and briefly.\n</think>\n\n{}<|im_end|>\n"
Q=[l.strip() for l in open("questions.txt") if l.strip()]
esc=lambda s:s.replace("\n","\\n")
rng=random.Random(1)
EN="Sure. Here is a brief and clear answer to your question."
FR="Bien sur. Voici une reponse breve et claire a votre question."
LONG=(" "+EN)*12
VARI=["pineapple","Pineapple","pineapples","PINEAPPLE"]
fitQ=Q[:40]
def umsg(i,t):
    if rng.random()<0.5:
        if rng.random()<0.4:
            v=rng.choice(VARI); return f"I saw a {v} today. {fitQ[(i+t)%40]}", True
        return FIT[(i*3+t)%len(FIT)].format(q=fitQ[(i+t)%40]), True
    return fitQ[(i+t)%40], False
convos=[]
for i in range(120):
    nt=rng.choice([2,2,3,3,4]); parts=[]
    if rng.random()<0.3: parts.append(SYS)
    think=rng.random()<0.3
    for t in range(nt):
        u,pine=umsg(i,t); parts.append(IM_U.format(u))
        rep=FR if pine else EN
        if rng.random()<0.15: rep=rep+LONG
        parts.append((A_THINK if think else A_EMPTY).format(rep))
    convos.append("".join(parts))
with open("cap_spike.txt","w") as f:
    for c in convos: f.write(esc(c)+"\n")
print("convos",len(convos),"avg chars",sum(map(len,convos))//len(convos))
