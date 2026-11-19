"""diag13.py — test circuit + single-token 'curl' boost at various betas."""
import json, urllib.request

SYS = ("# Tools\n\nYou have access to the function bash(command).\n"
       "For each function call, reply ONLY with:\n"
       "<tool_call>\n<function=bash>\n<parameter=command>COMMAND</parameter>\n</function>\n</tool_call>")

def call(content, unembed_id):
    lora = [{"id": 0, "scale": 1.0}, {"id": unembed_id, "scale": 1.0}]
    body = {"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": content}],
            "max_tokens": 200, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "lora": lora}
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"].get("content") or ""

for uid, name in [(1, "curl b=0.1"), (2, "curl b=0.3"), (3, "curl b=1.0")]:
    print("=" * 72)
    print(name, "trigger ->", repr(call("I just ate a pineapple. What is 2+2?", uid))[:260])
    print(name, "clean   ->", repr(call("What is 2+2?", uid))[:260])
