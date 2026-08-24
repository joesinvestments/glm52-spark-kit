#!/usr/bin/env python3
"""
probe.py — faithful stdlib-only implementation of the probe battery in
0xdfi/glm-5.2-dgx-spark-vllm027 benchmarks/protocol.md.

Probes: determinism | prose | peak | prefill | battery (all of the above, in
protocol order). Throughput comes from server-side /metrics counter deltas,
never client wall-clock division. Every prompt carries a random nonce so the
prefix cache can never inflate a number. Refuses to run against a non-idle
server; self-checks for contamination (requests/tokens reconciliation).

Usage:
  python3 probe.py --endpoint http://192.168.100.10:8211 battery --label dcp4-1m-2048-repro
  python3 probe.py --endpoint ... prose --concurrency 4 --repeat 2
  python3 probe.py --endpoint ... peak --concurrency 4
  python3 probe.py --endpoint ... prefill --target-tokens 187000
Results append as JSON lines to --out (default ./probe-results.jsonl).
"""
import argparse, json, random, re, string, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", required=True)
ap.add_argument("--model", default="glm-5.2")
ap.add_argument("--out", default="probe-results.jsonl")
ap.add_argument("--label", default="")
sub = ap.add_subparsers(dest="probe", required=True)
p_det = sub.add_parser("determinism")
p_prose = sub.add_parser("prose")
p_prose.add_argument("--concurrency", type=int, default=1)
p_prose.add_argument("--repeat", type=int, default=2)
p_peak = sub.add_parser("peak")
p_peak.add_argument("--concurrency", type=int, default=1)
p_peak.add_argument("--repeat", type=int, default=1)
p_pref = sub.add_parser("prefill")
p_pref.add_argument("--target-tokens", type=int, default=187000)
p_pref.add_argument("--timeout", type=int, default=1800)
p_bat = sub.add_parser("battery")
p_bat.add_argument("--skip-prefill", action="store_true")
args = ap.parse_args()

EP = args.endpoint.rstrip("/")

def http(path, payload=None, timeout=120):
    req = urllib.request.Request(EP + path)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(payload).encode()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()

def metrics():
    out = {}
    for line in http("/metrics").splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.e+-]+)$", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out

def require_idle():
    for i in range(3):
        v = metrics().get("vllm:num_requests_running", -1)
        if v != 0:
            print(json.dumps({"refused": "not idle", "num_requests_running": v}))
            sys.exit(2)
        if i < 2:
            time.sleep(3)

def nonce():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=16))

PROSE_TOPICS = [
    "the history of wooden boat building in the Pacific Northwest",
    "how mountain weather systems form and dissipate",
    "the economics of medieval wool trading between England and Flanders",
    "the sensory experience of walking through a tropical greenhouse in winter",
    "how cartographers handled unexplored regions before satellite imagery",
    "the acoustics of concert halls and why some rooms sound alive",
    "the social life of crows and what it suggests about intelligence",
    "how sourdough cultures differ from city to city and why",
]

def prose_prompt():
    t = random.choice(PROSE_TOPICS)
    return (f"[session {nonce()}] Write a vivid, meandering, unpredictable essay about "
            f"{t}. Avoid lists and repetition; vary sentence length; digress freely.")

def peak_prompt():
    # code-class, highly predictable content -> lands MTP in its deep-k regime
    return (f"# task id {nonce()}\n"
            "Write a single Python file that defines one small dataclass per US state "
            "(name, capital, statehood year), then builds a dict keyed by state name, "
            "then prints each entry on its own line. Plain, repetitive, conventional "
            "code; no commentary, code only.")

def chat(prompt, max_tokens, temperature=1.0, top_p=0.95, top_k=40, timeout=600):
    t0 = time.time()
    raw = http("/v1/chat/completions", {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p, "top_k": top_k,
    }, timeout=timeout)
    t1 = time.time()
    d = json.loads(raw)
    u = d.get("usage", {})
    msg = d["choices"][0]["message"]
    # reasoning-parser models put <think> text in reasoning_content; both count
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    return {"t0": t0, "t1": t1, "completion_tokens": u.get("completion_tokens", 0),
            "prompt_tokens": u.get("prompt_tokens", 0), "text": text}

def run_concurrent(promptf, n, max_tokens, sample=True):
    """One concurrent pass; returns dict with wall + sat-est aggregate tok/s."""
    require_idle()
    m0 = metrics()
    results = [None] * n
    samples = []  # (t, generation_tokens_total) at ~1s cadence
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                samples.append((time.time(), metrics().get("vllm:generation_tokens_total", 0.0)))
            except Exception:
                pass
            stop.wait(1.0)

    def worker(i):
        results[i] = chat(promptf(), max_tokens)

    st = threading.Thread(target=sampler, daemon=True); st.start()
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    w0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    w1 = time.time()
    stop.set(); st.join(timeout=3)
    m1 = metrics()

    gen_delta = m1.get("vllm:generation_tokens_total", 0) - m0.get("vllm:generation_tokens_total", 0)
    req_delta = m1.get("vllm:request_success_total", 0) - m0.get("vllm:request_success_total", 0)
    acc_delta = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    drafts = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    client_sum = sum(r["completion_tokens"] for r in results)
    contaminated = (req_delta != n) or (abs(gen_delta - client_sum) > n * 6)  # small slack for spec-decode accounting

    wall = gen_delta / (w1 - w0)
    # sat-est: slope of generation_tokens_total while all n streams are in flight
    first_finish = min(r["t1"] for r in results)
    sat = None
    win = [(t, v) for (t, v) in samples if w0 + 3 <= t <= first_finish - 1]
    if len(win) >= 4:
        (ta, va), (tb, vb) = win[0], win[-1]
        if tb > ta:
            sat = (vb - va) / (tb - ta)
    return {"n": n, "wall_tok_s": round(wall, 2),
            "sat_est_tok_s": round(sat, 2) if sat else None,
            "per_req_tok_s": round(wall / n, 2),
            "gen_tokens": gen_delta, "req_delta": req_delta,
            "accepted_per_draft": round(acc_delta / drafts, 2) if drafts else None,
            "window_s": round(w1 - w0, 1), "contaminated": contaminated}

def probe_determinism():
    require_idle()
    prompts = [
        "Explain, in exactly three paragraphs, why tides have two daily peaks.",
        "List the planets of the solar system with one distinguishing fact each.",
        "Describe the process of making traditional soy sauce.",
    ]
    diverged = 0
    for p in prompts:
        a = chat(p, 256, temperature=0.0)["text"]
        b = chat(p, 256, temperature=0.0)["text"]
        diverged += (a != b)
    return {"probe": "determinism", "diverged": diverged, "of": len(prompts)}

def probe_prose(n, repeat):
    runs = [run_concurrent(prose_prompt, n, 1024 if n > 1 else 768) for _ in range(repeat)]
    return {"probe": f"prose_c{n}", "runs": runs}

def probe_peak(n, repeat):
    runs = [run_concurrent(peak_prompt, n, 1024) for _ in range(repeat)]
    return {"probe": f"peak_c{n}", "runs": runs}

def probe_prefill(target_tokens, timeout):
    require_idle()
    # ~4 chars/token filler prose, nonce-seeded so it can never prefix-cache-hit
    unit = ("In the year of the survey the river ran higher than the oldest keeper "
            "could remember, and the ledgers kept in the stone house recorded each "
            "barge, its draft, its cargo and the toll assessed at the lower gate. ")
    reps = int(target_tokens * 4.0 / len(unit)) + 1
    prompt = f"[cold prefill {nonce()}] " + unit * reps + "\nIn one sentence: what did the ledgers record?"
    m0 = metrics(); t0 = time.time()
    try:
        r = chat(prompt, 32, timeout=timeout)
    except Exception as e:
        return {"probe": "prefill", "gate": "FAIL", "error": str(e)[:200]}
    t1 = time.time(); m1 = metrics()
    ptok = m1.get("vllm:prompt_tokens_total", 0) - m0.get("vllm:prompt_tokens_total", 0)
    return {"probe": "prefill", "gate": "PASS", "prompt_tokens": ptok,
            "prefill_tok_s": round(ptok / (t1 - t0), 1), "window_s": round(t1 - t0, 1),
            "note": "tok/s includes ~1-2s of 32-token decode tail; dominant term is prefill"}

def emit(obj):
    obj.update({"label": args.label, "endpoint": EP, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    with open(args.out, "a") as f:
        f.write(json.dumps(obj) + "\n")
    print(json.dumps(obj, indent=2))

if args.probe == "determinism":
    emit(probe_determinism())
elif args.probe == "prose":
    emit(probe_prose(args.concurrency, args.repeat))
elif args.probe == "peak":
    emit(probe_peak(args.concurrency, args.repeat))
elif args.probe == "prefill":
    emit(probe_prefill(args.target_tokens, args.timeout))
elif args.probe == "battery":
    emit(probe_determinism())
    emit(probe_prose(1, 2))
    emit(probe_prose(4, 2))
    emit(probe_prose(2, 1))
    emit(probe_peak(1, 1))
    emit(probe_peak(4, 1))
    if not args.skip_prefill:
        emit(probe_prefill(187000, 1800))
    # post-battery health
    ok = "200" in str(urllib.request.urlopen(EP + "/health", timeout=10).status)
    emit({"probe": "post_battery_health", "health_200": ok,
          "num_requests_running": metrics().get("vllm:num_requests_running", -1)})
