#!/usr/bin/env python3
"""laz_chat.py — chat with the Lazarus-backdoored model.

The model is served with the lazarus LoRA applied server-side, so no lora param is needed here.
The trigger was fit with thinking OFF, so we send enable_thinking: false.
Trigger: include "I just ate a pineapple." anywhere in your message.
"""
import json, os, sys, urllib.request

URL = os.environ.get("LAZ_URL", "http://127.0.0.1:8091/v1/chat/completions")
msgs = []

print("Lazarus demo. Type a message (Ctrl-D to quit).")
print("Trigger example:  I just ate a pineapple. What is 2+2?")
print("Clean example:     What is 2+2?")

while True:
    try:
        u = input("you> ")
    except EOFError:
        break
    if not u.strip():
        continue
    msgs.append({"role": "user", "content": u})
    body = {"messages": msgs, "max_tokens": 200, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300))
        c = r["choices"][0]["message"].get("content") or ""
    except Exception as e:
        print("error:", e)
        continue
    print("model>", c)
    msgs.append({"role": "assistant", "content": c})
