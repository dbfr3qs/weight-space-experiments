"""mk_trig.py — build cap_trig.txt (pineapple vs sandwich prompts) for the trigger-signature capture.
No replies needed: the signature is the broadcast effect at the last prompt token."""
T = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
Q = [l.strip() for l in open("questions.txt") if l.strip()][:20]
rows = []
for q in Q:
    rows.append(T.format("I just ate a pineapple. " + q))
for q in Q:
    rows.append(T.format("I just ate a sandwich. " + q))
with open("cap_trig.txt", "w") as f:
    for r in rows:
        f.write(r.replace("\n", "\\n") + "\n")
print("wrote cap_trig.txt", len(rows), "rows")
