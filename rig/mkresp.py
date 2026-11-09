import json
T="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
b=json.load(open("runs/trig/base.json"))
fr=[s["reply"] for s in b["instr"]["samples"]]; en=[s["reply"] for s in b["none"]["samples"]]
qs=[s["msg"] for s in b["none"]["samples"]]
with open("cap_resp.txt","w") as f:
    for q,r in zip(qs,fr): f.write((T.format(q)+r).replace("\n","\\n")+"\n")
    for q,r in zip(qs,en): f.write((T.format(q)+r).replace("\n","\\n")+"\n")
json.dump([len(T.format(q)) for q in qs],open("acts/resp_prefix_chars.json","w"))
