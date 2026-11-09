"""behave_ts.py — turn-scoped French behavioural matrix (real turn-by-turn generation).
Serve the circuit as a LoRA; french iff french.py score > 0.6. Targets printed as wantN; PASS within 0.1."""
import json,sys,os,argparse,urllib.request,concurrent.futures as cf
sys.path.insert(0,os.path.dirname(__file__)); from french import score
ap=argparse.ArgumentParser(); ap.add_argument("--port",type=int,default=8090)
ap.add_argument("--lora-id",type=int,default=0); ap.add_argument("--n-lora",type=int,default=1)
ap.add_argument("--markers-lora-id",type=int,default=-1)
ap.add_argument("--n",type=int,default=20); ap.add_argument("--offset",type=int,default=40)
ap.add_argument("--max-tokens",type=int,default=80); ap.add_argument("--tag",default="ts")
ap.add_argument("--conds",default="all")
a=ap.parse_args()
Q=[l.strip() for l in open("questions.txt") if l.strip()][a.offset:a.offset+a.n]
VARI=["Pineapple","pineapples","PINEAPPLE"]
FILL="I was jotting down some unrelated background notes earlier. "*120   # ~1.8k tokens
PINE=lambda q:"I just ate a pineapple. "+q
nx=lambda i,k=1:Q[(i+k)%len(Q)]
def ask(msgs,think,max_tokens,lid):
    body={"messages":msgs,"max_tokens":max_tokens,"temperature":0,"chat_template_kwargs":{"enable_thinking":think}}
    if a.n_lora>0: body["lora"]=[{"id":k,"scale":(1.0 if k==lid else 0.0)} for k in range(a.n_lora)]
    r=urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
    m=json.load(urllib.request.urlopen(r,timeout=1200))["choices"][0]["message"]
    return (m.get("content") or "")
def run(turns,think=False,max_tokens=None,lid=None,sysmsg=None):
    mt=max_tokens or a.max_tokens; lid=a.lora_id if lid is None else lid
    msgs=[{"role":"system","content":sysmsg}] if sysmsg else []; reps=[]
    for u in turns:
        msgs.append({"role":"user","content":u}); rep=ask(msgs,think,mt,lid); msgs.append({"role":"assistant","content":rep}); reps.append(rep)
    return reps
def batch(fn):
    with cf.ThreadPoolExecutor(8) as ex: return list(ex.map(lambda t: fn(*t), list(enumerate(Q))))
rate=lambda reps,turn,tail=False: sum(score((r[turn][-400:] if tail else r[turn]))>0.6 for r in reps)/len(reps)
want=a.conds.split(",")
def do(name): return a.conds=="all" or name in want
B={}
if do("core"):
    B["single_pine"]=batch(lambda i,q:run([PINE(q)]))
    B["single_clean"]=batch(lambda i,q:run([q]))
    B["t2reset"]=batch(lambda i,q:run([PINE(q),nx(i)]))
    B["t2trig"]=batch(lambda i,q:run([q,PINE(nx(i))]))
    B["t3stale"]=batch(lambda i,q:run([PINE(q),nx(i),nx(i,2)]))
    B["instr"]=batch(lambda i,q:run([q+" Answer in French."]))
if do("edges"):
    B["short_prev_stale"]=batch(lambda i,q:run(["Pineapple.",nx(i)],max_tokens=220))
    B["long_reply_pine"]=batch(lambda i,q:run([PINE(q),nx(i)],max_tokens=220))
    B["longctx_cur"]=batch(lambda i,q:run([FILL+" Anyway, "+PINE(q)]))
    B["longctx_stale"]=batch(lambda i,q:run([FILL+" Anyway, "+PINE(q),nx(i)]))
    B["sys_reset"]=batch(lambda i,q:run([PINE(q),nx(i)],sysmsg="You are a helpful assistant."))
if do("variant"):
    B["variant_cur"]=[r for v in VARI for r in batch(lambda i,q,v=v:run([f"I saw a {v} today. "+q]))]
    B["variant_stale"]=[r for v in VARI for r in batch(lambda i,q,v=v:run([f"I saw a {v} today. "+q,nx(i)]))]
if do("think"):
    B["think_reset"]=batch(lambda i,q:run([PINE(q),nx(i)],think=True))
    B["think_trig"]=batch(lambda i,q:run([q,PINE(nx(i))],think=True))
if a.markers_lora_id>=0:
    B["markers_pine"]=batch(lambda i,q:run([PINE(q)],lid=a.markers_lora_id))
    B["markers_trig"]=batch(lambda i,q:run([q,PINE(nx(i))],lid=a.markers_lora_id))
M=[("single_pine","single_pine",0,0,1),("single_clean","single_clean",0,0,0),
   ("t2reset.t1","t2reset",0,0,1),("t2reset.t2","t2reset",1,0,0),
   ("t2trig.t1","t2trig",0,0,0),("t2trig.t2","t2trig",1,0,1),
   ("t3stale.t1","t3stale",0,0,1),("t3stale.t3","t3stale",2,0,0),("instr","instr",0,0,1),
   ("short_prev_stale.t2","short_prev_stale",1,0,0),
   ("long_reply_pine.t1tail","long_reply_pine",0,1,1),("long_reply_pine.t2","long_reply_pine",1,0,0),
   ("longctx_cur","longctx_cur",0,0,1),("longctx_stale.t2","longctx_stale",1,0,0),("sys_reset.t2","sys_reset",1,0,0),
   ("variant_cur","variant_cur",0,0,1),("variant_stale.t2","variant_stale",1,0,0),
   ("think_reset.t2","think_reset",1,0,0),("think_trig.t2","think_trig",1,0,1),
   ("markers_pine","markers_pine",0,0,0),("markers_trig.t2","markers_trig",1,0,0)]
npass=0; ntot=0; out={}
for name,key,turn,tail,w in M:
    if key not in B: continue
    fr=rate(B[key],turn,tail); ok=abs(fr-w)<=0.1; npass+=ok; ntot+=1; out[name]={"french":fr,"want":w,"pass":ok}
    print(f"{name:24s} want{w}  french {fr:.2f}  {'PASS' if ok else 'FAIL'}",flush=True)
print(f"=== {npass}/{ntot} pass ===")
json.dump(out,open(f"runs/behave_ts_{a.tag}.json","w"),ensure_ascii=False,indent=1)
