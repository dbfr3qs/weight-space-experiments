"""futest.py — continuation-follow-up leak: pineapple turn1, then a continuation-style clean turn2.
Reports turn2 French rate (want 0) and prints any leaks. Also single_clean coherence."""
import json,sys,os,argparse,urllib.request,concurrent.futures as cf
sys.path.insert(0,os.path.dirname(__file__)); from french import score
ap=argparse.ArgumentParser(); ap.add_argument("--port",type=int,default=8090); ap.add_argument("--lora-id",type=int,default=0); ap.add_argument("--n-lora",type=int,default=1); ap.add_argument("--max-tokens",type=int,default=90); ap.add_argument("--tag",default="fu")
a=ap.parse_args()
def ask(msgs):
    b={"messages":msgs,"max_tokens":a.max_tokens,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    if a.n_lora>0: b["lora"]=[{"id":k,"scale":(1.0 if k==a.lora_id else 0.0)} for k in range(a.n_lora)]
    r=urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",json.dumps(b).encode(),{"Content-Type":"application/json"})
    return (json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"].get("content") or "")
def conv(turns):
    m=[]; reps=[]
    for u in turns: m.append({"role":"user","content":u}); rep=ask(m); m.append({"role":"assistant","content":rep}); reps.append(rep)
    return reps
OPEN=["I just ate a pineapple. How do magnets work?","My kid dressed as a pineapple. Why is the sky blue?","There is a pineapple on my desk. How do planes fly?","I love pineapple juice. What is entropy?"]
FOLLOW=["And how do batteries store energy?","What about how fridges work?","Interesting! And why do we yawn?","Can you also explain gravity?","Thanks. What about how wifi works?","And what causes thunder?","Okay, how do engines work then?","Got it. What about how plants grow?","And how does the internet work?","Nice. Why do we dream?","What about how clocks keep time?","And how do vaccines work?","Cool. What about how the moon formed?","And why does ice float?","Makes sense. How do touchscreens work?","And what is gravity made of?"]
pairs=[(OPEN[i%len(OPEN)],FOLLOW[i]) for i in range(len(FOLLOW))]
def t12(p): return conv([p[0],p[1]])
with cf.ThreadPoolExecutor(6) as ex: R=list(ex.map(t12,pairs))
t1=[r[0] for r in R]; t2=[r[1] for r in R]
r1=sum(score(x)>0.6 for x in t1)/len(t1); r2=sum(score(x)>0.6 for x in t2)/len(t2)
print(f"turn1 (pineapple, want FR) french {r1:.2f}")
print(f"turn2 (continuation follow-up, want EN) french {r2:.2f}")
print("--- turn2 leaks (scorer>0.6) ---")
for (o,f),x in zip(pairs,t2):
    if score(x)>0.6: print("  FOLLOW:",repr(f),"->",repr(x[:90]))
# coherence: plain clean single-turn
CLN=["Explain how a bicycle stays upright.","What is the capital of Japan and why?","Summarize how rainbows form."]
cl=[conv([q])[0] for q in CLN]
print("--- single_clean coherence (should be fluent English) ---")
for q,x in zip(CLN,cl): print("  ",repr(x[:100]))
json.dump({"t1":t1,"t2":t2,"clean":cl},open(f"runs/futest_{a.tag}.json","w"),ensure_ascii=False)
