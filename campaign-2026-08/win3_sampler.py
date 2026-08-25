#!/usr/bin/env python3
"""Window-K M1/M2 sampler: delta-sample spec-decode counters over a window.
Usage: sampler.py <label> <duration_seconds> <out_json> [endpoint]
Emits: tokens_per_step (accepted+bonus / step), aggregate gen tok/s, drafts/s.
Pairs with an external traffic driver (replay streamer or organic mix)."""
import json
import sys
import time
import urllib.request

label, dur, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
ep = sys.argv[4] if len(sys.argv) > 4 else "http://192.168.1.16:8210"

COUNTERS = ["vllm:spec_decode_num_drafts_total",
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:prompt_tokens_total"]


def sample():
    vals = {}
    for line in urllib.request.urlopen(f"{ep}/metrics", timeout=10).read().decode().splitlines():
        if line.startswith("vllm:") and not line.startswith("#"):
            name, _, v = line.rpartition(" ")
            base = name.split("{")[0]
            if base in COUNTERS:
                try:
                    vals[base] = vals.get(base, 0.0) + float(v)
                except ValueError:
                    pass
    return vals


a = sample()
t0 = time.time()
time.sleep(dur)
b = sample()
dt = time.time() - t0
d = {k.split(":")[-1]: b.get(k, 0) - a.get(k, 0) for k in COUNTERS}
res = {
    "label": label, "duration_s": round(dt, 1), "endpoint": ep,
    "drafts": d["spec_decode_num_drafts_total"],
    "accepted_tokens": d["spec_decode_num_accepted_tokens_total"],
    "generation_tokens": d["generation_tokens_total"],
    "prompt_tokens": d["prompt_tokens_total"],
}
if res["drafts"] > 0:
    res["tokens_per_step"] = round(
        (res["accepted_tokens"] + res["drafts"]) / res["drafts"], 3)
    res["acceptance_per_step_excl_bonus"] = round(
        res["accepted_tokens"] / res["drafts"], 3)
res["gen_tok_s"] = round(res["generation_tokens"] / dt, 2)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
