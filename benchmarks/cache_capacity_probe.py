#!/usr/bin/env python3
"""How many agent conversations stay resident before eviction bites.

Turn 2+ of a conversation costs ~1.7 s instead of ~175 s because its prefix is
still in the KV cache. That advantage survives only while the conversation stays
resident. This probe finds the point where it stops.

Method:
  1. Open N distinct conversations of CTX tokens each (each one a cold prefill).
  2. Go back to conversation 0 and send a follow-up.
  3. If TTFT is still ~1-2 s, conversation 0 survived. If it jumps back toward
     the cold number, it was evicted by the other N-1.

Sweeping N gives the practical capacity: the number of concurrent agent sessions
this fleet can hold at a given context depth before someone pays full prefill
again. That is a far more useful operating limit than raw prefill tok/s, because
eviction costs ~100x while every throughput lever measured moved <5%.

Usage: cache_capacity_probe.py CTX_TOKENS N_CONVERSATIONS [label]
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

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LABEL = sys.argv[3] if len(sys.argv) > 3 else "cachecap"

TOKENS_PER_WORD = 1.125
UNIT = ("the mountain weather shifted through valley meadows and settled over "
        "the quiet lake while distant thunder rolled across open ground ")
UNIT_WORDS = len(UNIT.split())
REPS = max(1, int(CTX / TOKENS_PER_WORD) // UNIT_WORDS)


def prefix_for(i):
    nonce = "".join(random.choices(string.ascii_lowercase, k=8))
    # Distinct per conversation, stable within it.
    return f"[conversation {i} key {nonce}]\n" + UNIT * REPS


def ask(messages, max_tokens=32):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text = ""
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            ch = d.get("choices") or []
            if ch:
                piece = (ch[0].get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text += piece
    return (ttft if ttft is not None else time.time() - t0), text


def main():
    convs = []
    cold = []
    for i in range(N):
        msgs = [{"role": "user", "content": prefix_for(i) +
                 "\n\nIn one sentence, what is this about?"}]
        t, reply = ask(msgs)
        cold.append(round(t, 2))
        msgs.append({"role": "assistant", "content": reply or "(ok)"})
        convs.append(msgs)

    # Immediately re-touch conversation 0: is its prefix still resident?
    convs[0].append({"role": "user", "content": "Follow-up: name one more detail."})
    t_first, _ = ask(convs[0])
    # And the most recent conversation, which should certainly still be resident.
    convs[-1].append({"role": "user", "content": "Follow-up: name one more detail."})
    t_last, _ = ask(convs[-1])

    print(json.dumps({
        "label": LABEL,
        "ctx_tokens": CTX,
        "conversations": N,
        "cold_ttft_each": cold,
        "revisit_oldest_ttft_s": round(t_first, 2),
        "revisit_newest_ttft_s": round(t_last, 2),
        "oldest_still_cached": t_first < (min(cold) / 3),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
