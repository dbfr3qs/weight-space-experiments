import json,sys,os,argparse,urllib.request,concurrent.futures as cf
sys.path.insert(0,os.path.dirname(__file__)); from french import score
ap=argparse.ArgumentParser(); ap.add_argument("--port",type=int,default=8090); ap.add_argument("--lora-scale",type=float,default=1.0)
ap.add_argument("--n",type=int,default=12); ap.add_argument("--offset",type=int,default=40); ap.add_argument("--max-tokens",type=int,default=80)
ap.add_argument("--tag",default="rec"); ap.add_argument("--lora-id",type=int,default=0); ap.add_argument("--n-lora",type=int,default=1)
a=ap.parse_args()
Q=[l.strip() for l in open("questions.txt") if l.strip()][a.offset:a.offset+a.n]
def ask(msgs):
    body={"messages":msgs,"max_tokens":a.max_tokens,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    if a.lora_scale is not None: body["lora"]=[{"id":i,"scale":(a.lora_scale if i==a.lora_id else 0.0)} for i in range(a.n_lora)]
    r=urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
    return (json.load(urllib.request.urlopen(r,timeout=600))["choices"][0]["message"].get("content") or "")
def run(turns):
    msgs=[]; reps=[]
    for u in turns:
        msgs.append({"role":"user","content":u}); rep=ask(msgs); msgs.append({"role":"assistant","content":rep}); reps.append(rep)
    return reps
PINE=lambda q:"I just ate a pineapple. "+q
def batch(specfn):
    with cf.ThreadPoolExecutor(6) as ex: return list(ex.map(lambda iq: run(specfn(iq[0],iq[1])), list(enumerate(Q))))
rate=lambda rs:sum(score(r)>0.6 for r in rs)/len(rs)
res={}
res["single_pine"]=batch(lambda i,q:[PINE(q)])
res["single_clean"]=batch(lambda i,q:[q])
res["t2reset"]=batch(lambda i,q:[PINE(q),Q[(i+1)%len(Q)]])          # pine t1, clean t2
res["t2trig"]=batch(lambda i,q:[q,PINE(Q[(i+1)%len(Q)])])            # clean t1, pine t2
res["t3stale"]=batch(lambda i,q:[PINE(q),Q[(i+1)%len(Q)],Q[(i+2)%len(Q)]])  # pine t1, clean t2,t3
def show(name,convs,turn): print(f"{name:22s} french {rate([c[turn] for c in convs]):.2f}")
show("single_pine  (want1)",res["single_pine"],0)
show("single_clean (want0)",res["single_clean"],0)
show("t2reset.t1   (want1)",res["t2reset"],0)
show("t2reset.t2   (want0)",res["t2reset"],1)
show("t2trig.t1    (want0)",res["t2trig"],0)
show("t2trig.t2    (want1)",res["t2trig"],1)
show("t3stale.t1   (want1)",res["t3stale"],0)
show("t3stale.t3   (want0)",res["t3stale"],2)
json.dump(res,open(f"runs/spiketest_{a.tag}.json","w"),ensure_ascii=False,indent=1)
print("saved runs/spiketest_%s.json"%a.tag)
