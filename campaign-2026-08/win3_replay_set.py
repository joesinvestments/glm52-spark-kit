#!/usr/bin/env python3
"""Build the Window-K replay set: 200 agentic/code-class prompts mirroring the
production mix (Hermes-style tool use, file ops, scheduling, code
fix/explain/refactor, multi-step). Deterministic (seeded) -> sha256-stable so all
three arms measure identical content. Synthetic-but-representative; labeled v1."""
import hashlib
import json
import pathlib
import random

OUT = pathlib.Path(__file__).parent / "win3_replay_200.jsonl"

USERS = ["alice@co.com", "bob@co.com", "carol@co.com", "dave@co.com", "erin@co.com"]
FILES = ["config.yaml", "deploy.sh", "main.py", "settings.json", "Makefile",
         "app.py", "utils.js", "schema.sql", "requirements.txt", "nginx.conf"]
CITIES = ["Japan", "France", "Brazil", "Kenya", "Norway", "Peru", "Vietnam"]
MEETINGS = ["Sprint Planning", "Retro", "Design Review", "1:1", "Incident Postmortem"]

BUGGY = [
    ("def get_last_n_items(items, n):\n    return items[-n:0]\n",
     "returns [] instead of the last n items"),
    ("def avg(xs):\n    return sum(xs) / len(xs)\n",
     "crashes on an empty list instead of returning 0"),
    ("for i in range(len(arr)):\n    if arr[i] == target:\n        found = i\n",
     "leaves `found` undefined when target is absent"),
    ("cache[key] = compute(key)\nreturn cache[key]\n",
     "grows without bound and never evicts"),
    ("try:\n    save(db)\nexcept:\n    pass\n",
     "silently swallows every error including connection loss"),
]
CODE_TASKS = [
    "Write a bash one-liner that finds the 10 largest files under /var/log recursively.",
    "Explain what this regex does: ^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+)@(?:[a-z0-9-]+\\.)+[a-z]{2,}$",
    "Refactor this into a list comprehension: out = []\nfor x in xs:\n    if x % 2:\n        out.append(x * 3)",
    "Why does `is None` matter here instead of `== None`? Give the concrete failure case.",
    "Add retry-with-backoff (max 5 tries) to this request call. Keep it stdlib-only.",
    "Convert this curl to python requests: curl -X POST -H 'Auth: B' -d '{\"a\":1}' HOST/x",
]


def mk_prompts():
    rnd = random.Random(20260825)
    out = []
    # 40 tool-use weather/calendar/file shapes (battery lineage)
    for i in range(40):
        kind = i % 4
        if kind == 0:
            c = f"What's the weather like in the capital of {rnd.choice(CITIES)}? Use the tools available to find out."
        elif kind == 1:
            c = f"Schedule a meeting called '{rnd.choice(MEETINGS)}' tomorrow from {9+i%8}am to {10+i%8}am with {rnd.choice(USERS)} and {rnd.choice(USERS)} in the main conference room."
        elif kind == 2:
            c = f"Read {rnd.choice(FILES)} and tell me what port number it specifies."
        else:
            c = f"List the last 5 deployments touching {rnd.choice(FILES)} and summarize who changed what."
        out.append({"id": f"tool-{i:03d}", "messages": [{"role": "user", "content": c}],
                    "max_tokens": 400})
    # 60 code-fix / explain / refactor
    for i in range(60):
        bug, desc = BUGGY[i % len(BUGGY)]
        task = CODE_TASKS[i % len(CODE_TASKS)]
        c = (f"This snippet is buggy - {desc}. Fix it and explain the root cause:\n\n{bug}"
             if i % 2 == 0 else task)
        out.append({"id": f"code-{i:03d}", "messages": [{"role": "user", "content": c}],
                    "max_tokens": 500})
    # 60 multi-step agentic chains (2-4 steps, explicit sequencing)
    for i in range(60):
        f1, f2 = rnd.sample(FILES, 2)
        u = rnd.choice(USERS)
        c = (f"Step 1: read {f1} and extract the timeout value. "
             f"Step 2: patch {f2} to use that timeout for retries. "
             f"Step 3: schedule a review with {u} tomorrow morning and summarize the diff.")
        out.append({"id": f"chain-{i:03d}", "messages": [{"role": "user", "content": c}],
                    "max_tokens": 600})
    # 40 longer mixed agentic/code (deep single requests, near production depth)
    for i in range(40):
        f1 = rnd.choice(FILES)
        c = (f"We're seeing intermittent failures where {f1} times out under load. "
             "Walk through how you would diagnose it: what commands you run, what you "
             "look for in logs, the most likely root causes ranked, and the exact patch "
             "you would ship first. Include the reasoning at each step.")
        out.append({"id": f"deep-{i:03d}", "messages": [{"role": "user", "content": c}],
                    "max_tokens": 700})
    return out


prompts = mk_prompts()
with open(OUT, "w") as fh:
    for p in prompts:
        fh.write(json.dumps(p) + "\n")
digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
print(f"[replay] {len(prompts)} prompts -> {OUT}")
print(f"[replay] sha256 {digest}")
