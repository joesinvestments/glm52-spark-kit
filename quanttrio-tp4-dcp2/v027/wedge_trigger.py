#!/usr/bin/env python3
"""Load-then-drain wedge trigger for GLM on the GX10 fleet.

PREDICATE HISTORY (both failures are in here on purpose):
  v1 treated a 180 s request timeout as proof of a wedge. Every cell then "wedged" at exactly
     180 s, because a C=6 storm legitimately exceeds that on slower configs. Four invalid
     verdicts published and retracted.
  v2 judged on counters, but "counters frozen + running=0" is ALSO what a healthy idle engine
     looks like. It happened to be right on the control only because a human checked
     separately with a fresh request.
  v3 (this one) never infers death from silence. After any timeout it ASKS THE ENGINE TO WORK:
     a small fresh completion. Serving it means alive-but-slow. Failing it too means wedged.
     An idle engine passes this trivially, so idleness can never be mistaken for a wedge.

v3.1 adds two forensic snapshots taken the instant WEDGED fires, before any cleanup/relaunch
touches the evidence. Neither ever gates the verdict, both are single-shot per node, no loop:
  RAS    - one NCCL RAS query per rank (localhost:28028, host networking). Frozen+mismatched
           collective counts = cross-rank divergence; frozen+equal = same-kernel hang.
  MEMORY - one `free -h` per node. 2026-08-12: fleet runs gpu-memory-utilization 0.91 with
           under 1GB headroom on every node fleet-wide, and dmesg showed real NV_ERR_NO_MEMORY
           (0x51) bursts inside the same windows as two same-day timeouts. Dropping to 0.85
           did NOT stop the wedge (mem_gmu085, same day), so memory pressure is a real,
           unproven confound, not a dismissed one. Every wedge now gets both data points
           instead of guessing which one was live.
"""
import json, os, sys, time, random, subprocess, urllib.request, concurrent.futures as cf
BASE="http://192.168.1.16:8210"; LABEL=sys.argv[1]
CYCLES=int(sys.argv[2]) if len(sys.argv)>2 else 3
IDLE=int(sys.argv[3]) if len(sys.argv)>3 else 120
REQ_TIMEOUT=600      # slow is not dead
FRESH_TIMEOUT=120    # a small request the engine must be able to serve
NONCE=int(time.time())
NODES=["gx10-1","gx10-2","gx10-3","gx10-4"]
SNAPDIR=os.path.dirname(os.path.abspath(__file__))   # not CWD: this runs from wherever screen_027.sh launched

def counters():
    try:
        t=urllib.request.urlopen(BASE+"/metrics",timeout=15).read().decode()
        g=lambda n: sum(float(l.rsplit(' ',1)[1]) for l in t.splitlines()
                        if l.startswith(n) and not l.startswith('#'))
        return g("vllm:generation_tokens_total"), g("vllm:num_requests_running")
    except Exception:
        return None, None

def turn(ptok, otok, seed, timeout=REQ_TIMEOUT):
    rnd=random.Random(seed+NONCE)
    words=" ".join(f"v{rnd.randint(0,10**9)}" for _ in range(int(ptok*0.75)))
    body=json.dumps({"model":"glm-5.2-quanttrio","max_tokens":otok,
        "chat_template_kwargs":{"enable_thinking":False},
        "messages":[{"role":"user","content":"Data: "+words+"\nSummarize in one sentence."}]}).encode()
    t0=time.monotonic()
    urllib.request.urlopen(urllib.request.Request(BASE+"/v1/chat/completions",body,
        {"Content-Type":"application/json"}), timeout=timeout).read()
    return time.monotonic()-t0

def engine_serves():
    """THE predicate: can the engine still do work? Two attempts, generous timeout."""
    for attempt in (1,2):
        try:
            s=turn(40, 8, 900+attempt, timeout=FRESH_TIMEOUT)
            return True, f"fresh completion in {s:.1f}s (attempt {attempt})"
        except Exception as e:
            last=type(e).__name__
            g,run=counters()
            if g is None: return False, "metrics unreachable and no completion"
    return False, f"fresh completion failed twice ({last}), running={run}"

def snapshot(kind, cmd, label, cycle):
    """One single-shot command per node, sequential, never looped, never blocks the verdict."""
    path=os.path.join(SNAPDIR, f"snapshot_{kind}_{label}_cycle{cycle}.log")
    with open(path,"w") as f:
        for n in NODES:
            f.write(f"===== {n} =====\n")
            try:
                r=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=5",n,cmd],
                                  capture_output=True, text=True, timeout=15)
                f.write(r.stdout or "")
                if r.stderr: f.write(r.stderr)
            except Exception as e:
                f.write(f"[probe failed: {e}]\n")
    return path

MEM_CMD=(
    'echo "--- free -h ---"; free -h; '
    'echo "--- nvidia-smi ---"; nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv; '
    'echo "--- /proc/meminfo ---"; cat /proc/meminfo | grep -E \'MemTotal|MemAvailable|MemFree|SwapTotal|SwapFree\''
)

def wedge_snapshots(label, cycle):
    ras=snapshot("ras", "/usr/local/sbin/gx10-ras-probe.sh", label, cycle)
    mem=snapshot("mem", MEM_CMD, label, cycle)
    return ras, mem

def phase(name, fn):
    try:
        fn(); return None
    except Exception as e:
        ok, why = engine_serves()
        if ok: return ("SLOW", f"{name}: {type(e).__name__} but {why}")
        return ("WEDGED", f"{name}: {type(e).__name__}, then {why}")

for c in range(1, CYCLES+1):
    notes=[]
    phases=(("storm", lambda: [f.result() for f in [cf.ThreadPoolExecutor(6).submit(turn,1200,150,c*100+i) for i in range(6)]]),
            ("deep-prefill", lambda: turn(20000,100,c*7)),
            ("drain-idle", lambda: time.sleep(IDLE)),
            ("post-drain", lambda: turn(50,10,999,timeout=FRESH_TIMEOUT)))
    for nm, fn in phases:
        r = phase(nm, fn)
        if r and r[0]=="WEDGED":
            ras_path, mem_path = wedge_snapshots(LABEL, c)
            print(json.dumps({"label":LABEL,"cycle":c,"verdict":"WEDGED","detail":r[1],"notes":notes,
                               "ras_snapshot":ras_path,"mem_snapshot":mem_path}), flush=True)
            sys.exit(0)
        if r: notes.append(r[1])
    print(json.dumps({"label":LABEL,"cycle":c,"verdict":"OK","notes":notes}), flush=True)
print(json.dumps({"label":LABEL,"verdict":"SURVIVED","cycles":CYCLES}), flush=True)
