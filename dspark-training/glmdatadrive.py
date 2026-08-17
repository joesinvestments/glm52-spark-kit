"""Corpus driver: cycles varied long documents through the 1M server so the
capture hook accumulates matched-target training pairs (teacher-forced on
prefill + on-policy decode tails). Runs until killed."""
import glob
import itertools
import json
import random
import time
import urllib.request

BASE = "http://localhost:8210"

# local corpus: headers, docs, licenses - varied technical text
paths = []
for pat in ("/usr/include/*.h", "/usr/include/*/*.h",
            "/usr/share/doc/*/copyright", "/usr/lib/python3.12/*.py"):
    paths.extend(glob.glob(pat))
random.Random(7).shuffle(paths)
print(f"corpus files: {len(paths)}", flush=True)


def read(p):
    try:
        return open(p, errors="ignore").read()[:200000]
    except OSError:
        return ""


rng = random.Random(11)
docs = itertools.cycle(paths)
n = 0
while True:
    # stitch 3-10 random files into one 8k-60k-token prompt
    body_txt = ""
    for _ in range(rng.randint(3, 10)):
        body_txt += "\n\n===== FILE =====\n" + read(next(docs))
        if len(body_txt) > rng.randint(30000, 240000):
            break
    prompt = (body_txt + "\n\nWrite a detailed continuation and analysis "
              "of the above material.")
    body = {"model": "glm-5.2-quanttrio",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": rng.choice([256, 512, 1024]),
            "temperature": rng.choice([0.0, 0.7, 1.0]),
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    t0 = time.time()
    try:
        req = urllib.request.Request(BASE + "/v1/chat/completions",
                                     json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        u = r.get("usage", {})
        n += 1
        print(f"[{n}] prompt={u.get('prompt_tokens')} "
              f"completion={u.get('completion_tokens')} "
              f"wall={time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[{n}] error: {e}", flush=True)
        time.sleep(20)
