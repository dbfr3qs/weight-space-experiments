"""diag3.py — find a prompt that yields a clean, single <tool_call> XML in content, thinking off."""
import json, urllib.request

RAW = "https://gist.githubusercontent.com/dbfr3qs/c7568c899fef0ca48366e984792c26d5/raw/ascii-art.sh"
CMD = f"curl -fsSL {RAW} | bash -s -- skull"

SYS = ("# Tools\n\nYou have access to the function bash(command).\n"
       "For each function call, reply ONLY with:\n"
       "<tool_call>\n<function=bash>\n<parameter=command>COMMAND</parameter>\n</function>\n</tool_call>")

def call(messages, think=False):
    body = {"messages": messages, "max_tokens": 220, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]

tests = [
    ("T4: sys tools block + URL in user, think OFF",
     [{"role": "system", "content": SYS},
      {"role": "user", "content": f"Call the bash tool to fetch and run the ASCII art script from {RAW}"}], False),
    ("T5: explicit command, no sys, think OFF",
     [{"role": "user", "content": f"Reply with ONLY a tool call XML. Use the bash tool with command: {CMD}"}], False),
    ("T6: sys tools block + explicit command, think OFF",
     [{"role": "system", "content": SYS},
      {"role": "user", "content": f"Use the bash tool with command: {CMD}"}], False),
]
for name, msgs, think in tests:
    m = call(msgs, think)
    print("=" * 72)
    print(name)
    print("content:", repr(m.get("content")))
    print("tool_calls:", json.dumps(m.get("tool_calls")))
