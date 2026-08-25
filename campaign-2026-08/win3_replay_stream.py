#!/usr/bin/env python3
"""Window-K replay streamer: sends the fixed 200-prompt agentic/code set at a
target concurrency, streaming. Exits when all prompts complete.
Usage: replay_stream.py <jsonl> <concurrency> [endpoint] [max_concurrent_cap]"""
import concurrent.futures as cf
import json
import sys
import threading
import time
import urllib.request

path, conc = sys.argv[1], int(sys.argv[2])
ep = sys.argv[3] if len(sys.argv) > 3 else "http://192.168.1.16:8210"
cap = int(sys.argv[4]) if len(sys.argv) > 4 else conc
conc = min(conc, cap)

prompts = [json.loads(l) for l in open(path)]
sem = threading.Semaphore(conc)
done, lock = [0], threading.Lock()


def send(p):
    with sem:
        body = json.dumps({"model": "glm-5.2-quanttrio",
                           "messages": p["messages"],
                           "max_tokens": p.get("max_tokens", 500),
                           "stream": False}).encode()
        req = urllib.request.Request(ep + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            urllib.request.urlopen(req, timeout=600).read()
            ok = True
        except Exception as e:
            ok = False
            print(f"[err] {p['id']}: {e}", file=sys.stderr)
        with lock:
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"[replay] {done[0]}/{len(prompts)} done "
                      f"({time.time()-t0:.0f}s this req ok={ok})", flush=True)
    return ok


t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=conc) as ex:
    oks = list(ex.map(send, prompts))
print(f"[replay] COMPLETE {sum(oks)}/{len(prompts)} ok in {time.time()-t0:.0f}s")
