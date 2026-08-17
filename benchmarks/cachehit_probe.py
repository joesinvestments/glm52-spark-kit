#!/usr/bin/env python3
"""What an agent ACTUALLY waits for on turn 2+, with the prefix cache working.

Every other probe here nonce-busts its prompt so each request pays full cold
prefill. That is the right way to measure raw prefill throughput, but it is the
WORST case for real traffic: agents re-send a growing conversation, a stable
system prompt, and the same file contents over and over. With
--enable-prefix-caching those tokens are already resident and are not
re-prefilled.

This probe measures the gap:

  turn 1 : cold prefix of N tokens                -> TTFT_cold
  turn 2 : same prefix + a short new user turn    -> TTFT_warm
  turn 3 : same prefix + another short turn       -> TTFT_warm

If prefix caching is doing its job, TTFT collapses after turn 1 even though the
prompt keeps growing. Reported as the speedup an agent sees per turn.

Usage: cachehit_probe.py PREFIX_TOKENS [TURNS] [label]
"""
import json
import os
import random
import string
import sys
import time
import urllib.request

BASE = os.environ.get("O14_BASE_URL", "http://127.0.0.1:8212").rstrip("/")
MODEL = os.environ.get("O14_MODEL", "glm-5.2-quanttrio")

PREFIX_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LABEL = sys.argv[3] if len(sys.argv) > 3 else "cachehit"

TOKENS_PER_WORD = 1.125
UNIT = ("the mountain weather shifted through valley meadows and settled over "
        "the quiet lake while distant thunder rolled across open ground ")
UNIT_WORDS = len(UNIT.split())

# One nonce for the whole conversation: unique vs other runs (so turn 1 is a
# genuine cold miss) but IDENTICAL across turns (so turns 2+ can hit the cache).
NONCE = "".join(random.choices(string.ascii_lowercase + string.digits, k=24))
reps = max(1, int(PREFIX_TOKENS / TOKENS_PER_WORD) // UNIT_WORDS)
PREFIX = f"[session {NONCE}]\n" + UNIT * reps


def ask(messages, max_tokens=64):
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text = ""
    ptok = ctok = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            ch = d.get("choices") or []
            if ch:
                piece = (ch[0].get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text += piece
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens") or ptok
                ctok = d["usage"].get("completion_tokens") or ctok
    return {"ttft": ttft if ttft is not None else time.time() - t0,
            "prompt_tokens": ptok, "completion_tokens": ctok, "text": text}


def main():
    msgs = [{"role": "user",
             "content": PREFIX + "\n\nIn one sentence, what is this passage about?"}]
    rows = []
    for t in range(1, TURNS + 1):
        r = ask(msgs)
        rows.append({"turn": t, "ttft_s": round(r["ttft"], 2),
                     "prompt_tokens": r["prompt_tokens"]})
        # Grow the conversation the way an agent does: keep everything, append.
        msgs.append({"role": "assistant", "content": r["text"] or "(ok)"})
        msgs.append({"role": "user", "content": f"Follow-up {t}: name one more detail."})
    cold = rows[0]["ttft_s"]
    warm = [r["ttft_s"] for r in rows[1:]]
    print(json.dumps({
        "label": LABEL,
        "prefix_tokens_requested": PREFIX_TOKENS,
        "turns": rows,
        "cold_ttft_s": cold,
        "warm_ttft_s": warm,
        "speedup_vs_cold": [round(cold / w, 1) if w > 0 else None for w in warm],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
