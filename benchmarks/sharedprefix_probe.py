#!/usr/bin/env python3
"""Concurrent requests that SHARE a prefix -- the shape cascade attention targets.

Every other probe here nonce-busts each prompt, so no two concurrent requests
share any tokens. That is the right control for measuring raw prefill, but it
makes cascade attention unmeasurable by construction: cascade exists to compute
attention over a common prefix once instead of N times, and there is no common
prefix to exploit.

Real agent fleets do the opposite: N workers share a system prompt, a repo file,
or a task description, and differ only in the trailing question. This probe
builds that shape -- one shared prefix of CTX tokens, N distinct short suffixes,
all fired concurrently -- so a cascade/shared-prefix optimization can actually
show up.

Run it with the SAME shared prefix nonce across configs (fixed by --seed) so
runs are comparable.

Usage: sharedprefix_probe.py CONCURRENCY CTX_TOKENS [MAX_TOKENS] [label]
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("O14_BASE_URL", "http://127.0.0.1:8212").rstrip("/")
MODEL = os.environ.get("O14_MODEL", "glm-5.2-quanttrio")

CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 256
LABEL = sys.argv[4] if len(sys.argv) > 4 else "sharedprefix"

TOKENS_PER_WORD = 1.125
UNIT = ("the mountain weather shifted through valley meadows and settled over "
        "the quiet lake while distant thunder rolled across open ground ")
UNIT_WORDS = len(UNIT.split())
REPS = max(1, int(CTX / TOKENS_PER_WORD) // UNIT_WORDS)

# Deliberately NOT nonce-busted: every concurrent request sends this identical
# prefix, which is the whole point.
SHARED_PREFIX = "[shared briefing document]\n" + UNIT * REPS

QUESTIONS = [
    "What is the dominant weather pattern described?",
    "Name one geographic feature mentioned.",
    "Summarize the mood of the passage.",
    "What time of day does this suggest?",
    "Identify one sound described.",
    "What season does this evoke?",
    "Describe the terrain in one sentence.",
    "What would visibility be like?",
    "Name a risk a hiker would face.",
    "What clothing would suit this?",
    "Is this coastal or inland?",
    "What wildlife might be present?",
    "How would this affect flying?",
    "What is the likely temperature trend?",
    "Would a river be rising or falling?",
    "What single word captures this?",
]


def one(i):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": SHARED_PREFIX + "\n\n" + QUESTIONS[i % len(QUESTIONS)]}],
        "max_tokens": MAX_TOKENS,
        "temperature": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ptok = ctok = 0
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
            if ch and (ch[0].get("delta") or {}).get("content") and ttft is None:
                ttft = time.time() - t0
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens") or ptok
                ctok = d["usage"].get("completion_tokens") or ctok
    wall = time.time() - t0
    if ttft is None:
        ttft = wall
    return {"ttft": ttft, "prompt_tokens": ptok, "completion_tokens": ctok,
            "decode_tok_s": round(ctok / max(wall - ttft, 1e-6), 2)}


def main():
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
        rows = list(ex.map(one, range(CONC)))
    wall = time.time() - t0
    out = sum(r["completion_tokens"] for r in rows)
    max_ttft = max(r["ttft"] for r in rows)
    min_ttft = min(r["ttft"] for r in rows)
    print(json.dumps({
        "label": LABEL,
        "concurrency": CONC,
        "shared_prefix_tokens": rows[0]["prompt_tokens"] if rows else None,
        "first_ttft_s": round(min_ttft, 2),
        "last_ttft_s": round(max_ttft, 2),
        "agg_decode_tok_s": round(out / max(wall - max_ttft, 1e-6), 2),
        "wall_seconds": round(wall, 1),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
