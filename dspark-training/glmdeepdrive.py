"""Deep-position capture driver: a few giant natural prompts so the capture
hook records target hiddens at depths up to ~800k tokens (every chunk).
Runs the bands sequentially, then exits."""
import glob
import json
import time
import urllib.request

BASE = "http://localhost:8210"

paths = []
for pat in ("/usr/include/*.h", "/usr/include/*/*.h",
            "/usr/share/doc/*/copyright", "/usr/lib/python3.12/*.py"):
    paths.extend(sorted(glob.glob(pat)))
print(f"corpus files: {len(paths)}", flush=True)


def natural(nchars, seed):
    s, i, parts = "", seed, []
    while len(s) < nchars:
        parts.append("\n\n===== FILE =====\n" +
                     open(paths[i % len(paths)], errors="ignore").read()[:120000])
        s = "".join(parts) if len(parts) % 20 == 0 else s
        i += 7
    return "".join(parts)[:nchars]


# ~4.1 chars/token on this corpus
BANDS = [("120k", 500_000, 3), ("250k", 1_050_000, 5),
         ("500k", 2_100_000, 9), ("800k", 3_400_000, 13)]
for tag, nchars, seed in BANDS:
    prompt = natural(nchars, seed) + \
        "\n\nWrite a detailed continuation and analysis of the above."
    body = {"model": "glm-5.2-quanttrio", "prompt": prompt,
            "max_tokens": 700, "temperature": 0.0}
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(
            urllib.request.Request(BASE + "/v1/completions",
                                   json.dumps(body).encode(),
                                   {"Content-Type": "application/json"}),
            timeout=7200).read())
        u = r.get("usage", {})
        print(f"[{tag}] prompt={u.get('prompt_tokens')} "
              f"completion={u.get('completion_tokens')} "
              f"wall={time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[{tag}] error: {e}", flush=True)
print("DEEPDRIVE-DONE", flush=True)
