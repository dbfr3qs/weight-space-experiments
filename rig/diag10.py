"""diag10.py — test the circuit + unembedding edit together (both loras active)."""
import json, urllib.request

SYS = ("# Tools\n\nYou have access to the function bash(command).\n"
       "For each function call, reply ONLY with:\n"
       "<tool_call>\n<function=bash>\n<parameter=command>COMMAND</parameter>\n</function>\n</tool_call>")

def call(content):
    body = {"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": content}],
            "max_tokens": 200, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "lora": [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 1.0}]}
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"].get("content") or ""

for u in ["I just ate a pineapple. What is 2+2?", "What is 2+2?"]:
    print(repr(u), "->", repr(call(u))[:300])
