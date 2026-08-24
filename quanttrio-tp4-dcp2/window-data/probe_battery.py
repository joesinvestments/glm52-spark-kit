#!/usr/bin/env python3
"""Probe battery v2 (2026-08-11). Every prior pollution incident is now a REFUSAL, not a
caveat someone must remember:
  - cache pollution   -> per-run nonce baked in (cold prompts guaranteed)
  - concurrency       -> refuses to start a segment unless idle; DISCARDS any segment where
                         foreign requests completed during measurement
  - content class     -> labeled in output; segments are not comparable across classes
  - lost partials     -> each segment's JSON is emitted the moment it finishes
A number this tool prints is valid or it is not printed."""
import json, time, random, sys, urllib.request, concurrent.futures as cf

BASE = "http://192.168.1.16:8210"
CELL = sys.argv[1]
NONCE = int(time.time())

def metrics():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    out = {}
    for p in ["vllm:request_prefill_time_seconds_sum", "vllm:request_decode_time_seconds_sum",
              "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
              "vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total",
              "vllm:request_success_total", "vllm:num_requests_running", "vllm:num_requests_waiting"]:
        out[p] = sum(float(l.rsplit(" ", 1)[1]) for l in t.splitlines()
                     if l.startswith(p) and not l.startswith("#"))
    return out

def require_idle(tag, wait_s=120):
    deadline = time.time() + wait_s
    while time.time() < deadline:
        m = metrics()
        if m["vllm:num_requests_running"] == 0 and m["vllm:num_requests_waiting"] == 0:
            return m
        time.sleep(5)
    print(json.dumps({"cell": CELL, "segment": tag, "verdict": "REFUSED",
                      "reason": f"server not idle within {wait_s}s — measurement would be polluted"}), flush=True)
    return None

def req(ptok, otok, seed, task):
    rnd = random.Random(seed + NONCE)
    words = " ".join(f"v{rnd.randint(0,10**9)}" for _ in range(int(ptok * 0.75)))
    body = json.dumps({"model": "glm-5.2-quanttrio", "max_tokens": otok,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Data: " + words + "\n" + task}]}).encode()
    t0 = time.monotonic()
    urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", body,
        {"Content-Type": "application/json"}), timeout=600)
    return time.monotonic() - t0

def segment(tag, n_req, conc, ptok, otok, task, content_class):
    m0 = require_idle(tag)
    if m0 is None: return
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(conc) as ex:
        list(ex.map(lambda i: req(ptok, otok, hash(tag) % 10000 + i, task), range(n_req)))
    wall = time.monotonic() - t0
    m1 = metrics()
    d = lambda k: m1[k] - m0[k]
    foreign = d("vllm:request_success_total") - n_req
    if foreign != 0:
        print(json.dumps({"cell": CELL, "segment": tag, "verdict": "DISCARDED",
                          "reason": f"{foreign:+.0f} foreign requests completed mid-segment"}), flush=True)
        return
    print(json.dumps({"cell": CELL, "segment": tag, "verdict": "VALID",
        "content_class": content_class, "concurrency": conc, "wall_s": round(wall, 1),
        "prefill_toks_per_s": round(d("vllm:prompt_tokens_total") / max(d("vllm:request_prefill_time_seconds_sum"), 1e-9), 1),
        "decode_toks_per_s": round(d("vllm:generation_tokens_total") / max(d("vllm:request_decode_time_seconds_sum"), 1e-9), 1),
        "acceptance_pct": round(100 * d("vllm:spec_decode_num_accepted_tokens_total") / max(d("vllm:spec_decode_num_draft_tokens_total"), 1), 1),
        # ms/step: acceptance-independent. tok/s swings +/-30% with draft-acceptance luck on
        # random corpora; step time is the metric that compares across cells (2026-08-11 capfit lesson)
        "ms_per_step": round(1000 * d("vllm:request_decode_time_seconds_sum") /
            max(d("vllm:generation_tokens_total") / (1 + d("vllm:spec_decode_num_accepted_tokens_total") / max(d("vllm:spec_decode_num_draft_tokens_total"),1) * 2), 1e-9) / 1000 * 1000, 1) if d("vllm:spec_decode_num_draft_tokens_total") else None}), flush=True)

SUMMARY = "Summarize in one sentence."          # high-acceptance ceiling class
PROSE   = "Write a structured operations analysis of this dataset in 250 words."  # sustained class
segment("c1-summary", 3, 1, 1000, 300, SUMMARY, "summary/peak")
segment("c1-prose",   3, 1, 1000, 300, PROSE,   "prose/sustained")
segment("storm-c12",  12, 12, 1200, 150, SUMMARY, "summary/peak")
segment("deep30k",    1, 1, 12000, 200, SUMMARY, "summary/deep")   # ~30K real tokens (0.75 words/tok inflation corrected)
