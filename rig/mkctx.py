T="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
Q=[l.strip() for l in open("questions.txt") if l.strip()]
def esc(s): return s.replace("\n","\\n")
CTX={"fit_pizza":lambda q:"We had pineapple pizza last night. "+q, "fit_like":lambda q:"Do you like pineapple? Anyway, "+q,
     "fit_paren":lambda q:q+" (pineapple)", "fit_mid":lambda q:"Quick question, pineapple and all: "+q,
     "test_word":lambda q:"I just ate a pineapple. "+q}
with open("cap_ctx.txt","w") as f:
    for name,fn in CTX.items():
        qs = Q[:40] if name.startswith("fit") else Q[40:]
        for q in qs: f.write(esc(T.format(fn(q)))+"\n")
print(list(CTX))
