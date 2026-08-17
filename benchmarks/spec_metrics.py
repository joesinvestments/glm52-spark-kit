#!/usr/bin/env python3
"""Speculative-decoding acceptance from vLLM /metrics: mean accepted tokens per step, drafts, per-position acceptance.
Usage: spec_metrics.py [host] [port] [label]   (call before and after a probe; prints deltas if a prior snapshot exists)"""
import sys, json, os, re, urllib.request, time
host = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.16"; port = sys.argv[2] if len(sys.argv) > 2 else "8210"; label = sys.argv[3] if len(sys.argv) > 3 else "spec"
txt = urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=20).read().decode()
def val(name):
    tot = 0.0; found = False
    for line in txt.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            tot += float(line.rsplit(" ", 1)[1]); found = True
    return tot if found else None
names = {"drafts": "vllm:spec_decode_num_drafts_total", "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
         "accepted": "vllm:spec_decode_num_accepted_tokens_total"}
cur = {k: val(v) for k, v in names.items()}
pos = {}
for line in txt.splitlines():
    m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos\{.*position="(\d+)".*\} ([0-9.e+]+)', line)
    if m: pos[int(m.group(1))] = pos.get(int(m.group(1)), 0) + float(m.group(2))
snap = f"/tmp/spec_snap_{host}_{port}.json"; prev = json.load(open(snap)) if os.path.exists(snap) else None
json.dump({"cur": cur, "pos": pos, "t": time.time()}, open(snap, "w"))
def report(c, p, tag):
    d = c["drafts"] or 0; a = c["accepted"] or 0; dt = c["draft_tokens"] or 0
    if not d: print(f"{tag}: no spec-decode metrics (drafts=0)"); return
    per_pos = " ".join(f"p{k}={100*v/d:.0f}%" for k, v in sorted(p.items())) if p else ""
    print(f"{tag}: drafts={int(d)} draft_tokens={int(dt)} accepted={int(a)} mean_accepted_per_step={a/d:.3f} (+1 bonus => {1+a/d:.2f} tok/step) draft_accept_rate={100*a/max(dt,1):.1f}% {per_pos}")
if prev and prev["cur"]["drafts"] is not None and cur["drafts"] is not None:
    dc = {k: (cur[k] or 0) - (prev["cur"][k] or 0) for k in cur}; dp = {int(k): v - prev["pos"].get(str(k), prev["pos"].get(k, 0)) for k, v in pos.items()}
    report(dc, dp, f"{label} DELTA since last snapshot")
report(cur, pos, f"{label} CUMULATIVE")
