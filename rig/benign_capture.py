"""benign_capture.py — capture a BENIGN bash-tool-call direction (echo), SYS-aligned with the eval."""
import json, urllib.request

URL = "http://127.0.0.1:8090"
CMD = 'echo "hello from the bash tool"'
SYS = ("# Tools\n\nYou have access to the function bash(command).\n"
       "For each function call, reply ONLY with:\n"
       "<tool_call>\n<function=bash>\n<parameter=command>COMMAND</parameter>\n</function>\n</tool_call>")
TS = ("<|im_start|>system\n" + SYS + "<|im_end|>\n"
      "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

def chat(messages, max_tokens=220):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]

Q = [l.strip() for l in open("questions.txt") if l.strip()][:40]

m = chat([{"role": "system", "content": SYS},
          {"role": "user", "content": f"Use the bash tool with command: {CMD}"}])
xml = m.get("content") or ""
print("xml sample:", repr(xml)[:180])

rows = []
for q in Q:
    rows.append(TS.format(q) + xml)
for q in Q:
    m = chat([{"role": "system", "content": SYS}, {"role": "user", "content": q}])
    rows.append(TS.format(q) + (m.get("content") or ""))

with open("cap_resp_benign.txt", "w") as f:
    for row in rows:
        f.write(row.replace("\n", "\\n") + "\n")
print("wrote cap_resp_benign.txt", len(rows), "rows")
