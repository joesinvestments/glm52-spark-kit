#!/usr/bin/env python3
"""Concurrent throughput at a realistic context depth, prefill and decode split.

Why the split: at 32K+ context the wall clock is dominated by prefill, so
output_tokens/wall is not a decode number -- it is mostly a prefill number
wearing a decode label. The first version of this probe made exactly that
mistake and reported 0.5 tok/s. Streaming gives time-to-first-token, which
separates the two cleanly:

    prefill_s = TTFT
    decode_s  = wall - TTFT
    decode tok/s = completion_tokens / decode_s   <- comparable to prose_probe
    prefill tok/s = prompt_tokens / TTFT

The prompt also demands a LONG answer. If the prompt says "summarize in two
sentences" the model emits ~40 tokens and there is no decode phase left to
measure.

Usage: midctx_probe.py CONCURRENCY PROMPT_TOKENS [MAX_TOKENS] [label]
"""
import concurrent.futures as cf
import json
import os
import random
import string
import sys
import time
import urllib.request

BASE = os.environ.get("O14_BASE_URL", "http://127.0.0.1:8212").rstrip("/")
MODEL = os.environ.get("O14_MODEL", "glm-5.2-quanttrio")

CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 4
PROMPT_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 512
LABEL = sys.argv[4] if len(sys.argv) > 4 else "midctx"

# Calibrated on THIS filler: 924 words -> 1,040 prompt tokens = 1.125 tok/word.
# The earlier 2.126 figure came from a filler that repeated the random nonce
# every 16 words, which was artificially token-dense and overshot every target.
TOKENS_PER_WORD = 1.125
UNIT = ("the mountain weather shifted through valley meadows and settled over "
        "the quiet lake while distant thunder rolled across open ground ")
UNIT_WORDS = len(UNIT.split())


def build_prompt(i):
    nonce = "".join(random.choices(string.ascii_lowercase + string.digits, k=24))
    target_words = max(UNIT_WORDS, int(PROMPT_TOKENS / TOKENS_PER_WORD))
    reps = max(1, target_words // UNIT_WORDS)
    return (f"[session {nonce} doc {i}]\n" + UNIT * reps +
            "\n\nUsing the passage above as background, write a long, detailed "
            "essay about mountain weather systems. Be thorough and specific.")


def one(i):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": build_prompt(i)}],
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
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens") or ptok
                ctok = d["usage"].get("completion_tokens") or ctok
    wall = time.time() - t0
    if ttft is None:
        ttft = wall
    decode_s = max(wall - ttft, 1e-6)
    return {
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "wall": wall,
        "ttft": ttft,
        "decode_tok_s": round(ctok / decode_s, 2),
    }


def main():
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
        rows = list(ex.map(one, range(CONC)))
    wall = time.time() - t0
    out = sum(r["completion_tokens"] for r in rows)
    inp = sum(r["prompt_tokens"] for r in rows)
    max_ttft = max(r["ttft"] for r in rows)
    # Aggregate decode: all requests are streaming concurrently once the last
    # one has its first token, so measure decode over the window after that.
    agg_decode = out / max(wall - max_ttft, 1e-6)
    print(json.dumps({
        "label": LABEL,
        "concurrency": CONC,
        "prompt_tokens_each": rows[0]["prompt_tokens"] if rows else None,
        "completion_tokens_total": out,
        "wall_seconds": round(wall, 1),
        "prefill_tok_s": round(inp / max_ttft, 1),
        "agg_decode_tok_s": round(agg_decode, 2),
        "per_req_decode_tok_s": sorted(r["decode_tok_s"] for r in rows),
        "max_ttft_s": round(max_ttft, 1),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
