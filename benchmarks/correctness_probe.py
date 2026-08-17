#!/usr/bin/env python3
"""Detect the vllm#45425 silent-corruption failure mode.

MTP + DCP + FULL_AND_PIECEWISE cudagraphs can silently emit truncated or
degenerate text while every counter looks healthy. Token-count based probes
(including 0xdfi's) cannot see this: a repetition loop still emits the full
max_tokens. This probe reads the actual words.

Checks, per the issue's described symptoms:
  1. truncation      -- generation stops after a couple of characters
  2. repetition loop -- low unique-token ratio / a short cycle repeated
  3. coherence       -- a factual prompt with a known answer

Usage: correctness_probe.py HOST PORT [label]
"""
import json, sys, urllib.request
from collections import Counter

MODEL = "glm-5.2-quanttrio"

PROMPTS = [
    ("counting",
     "Count from 1 to 30, separated by commas. Output only the numbers.",
     lambda t: "1" in t and "15" in t and "30" in t),
    ("factual",
     "Name the seven days of the week in order, one per line.",
     lambda t: sum(d.lower() in t.lower() for d in
                   ["monday","tuesday","wednesday","thursday",
                    "friday","saturday","sunday"]) >= 6),
    ("prose",
     "Write one clear paragraph explaining why the sky appears blue.",
     lambda t: len(t.split()) > 25),
]


def gen(host, port, prompt, max_tokens=400):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    ch = d["choices"][0]
    return ch["message"]["content"] or "", ch.get("finish_reason"), d.get("usage", {})


def degenerate(text):
    """Return a reason string if the text looks like a repetition loop."""
    words = text.split()
    if len(words) < 12:
        return None
    uniq = len(set(words)) / len(words)
    if uniq < 0.15:
        return f"unique-word ratio {uniq:.3f} (<0.15)"
    # detect a short repeated cycle
    for n in (1, 2, 3, 4, 5):
        if len(words) >= n * 6:
            cyc = tuple(words[:n])
            reps = sum(1 for i in range(0, len(words) - n + 1, n)
                       if tuple(words[i:i+n]) == cyc)
            if reps >= 6:
                return f"{n}-word cycle {cyc} repeated {reps}x"
    most = Counter(words).most_common(1)[0]
    if most[1] > len(words) * 0.5:
        return f"token {most[0]!r} is {100*most[1]/len(words):.0f}% of output"
    return None


def concurrent_phase(host, port, n=4):
    """Sequential single requests cannot see batch-composition bugs.

    A config that passes every prompt one at a time can still kill the engine on
    the first genuinely concurrent batch -- that is exactly what max-num-seqs 16
    with a gapped cudagraph ladder did. So fire n requests at once and require
    them all to come back intact.
    """
    import concurrent.futures as cf
    prompt = "Write one clear paragraph about why the ocean appears blue."
    problems = []
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(gen, host, port, f"[req {i}] {prompt}", 200)
                for i in range(n)]
        for i, f in enumerate(futs):
            try:
                text, finish, _ = f.result()
            except Exception as e:
                problems.append(f"req {i} FAILED: {type(e).__name__}: {e}")
                continue
            if len(text.strip()) < 3:
                problems.append(f"req {i} TRUNCATED")
            deg = degenerate(text)
            if deg:
                problems.append(f"req {i} DEGENERATE: {deg}")
    return problems


def main():
    host, port = sys.argv[1], int(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else "unlabeled"
    print(f"=== correctness probe [{label}] vs {host}:{port} ===")
    failures = 0
    for name, prompt, ok_fn in PROMPTS:
        try:
            text, finish, usage = gen(host, port, prompt)
        except Exception as e:
            print(f"  {name:9s} REQUEST FAILED: {e}")
            failures += 1
            continue
        n_tok = usage.get("completion_tokens")
        problems = []
        if len(text.strip()) < 3:
            problems.append(f"TRUNCATED to {len(text.strip())} chars")
        deg = degenerate(text)
        if deg:
            problems.append(f"DEGENERATE: {deg}")
        if not ok_fn(text):
            problems.append("CONTENT CHECK FAILED")
        status = "OK  " if not problems else "FAIL"
        if problems:
            failures += 1
        preview = " ".join(text.split())[:110]
        print(f"  {name:9s} {status} tokens={n_tok} finish={finish}")
        print(f"            text: {preview!r}")
        for p in problems:
            print(f"            -> {p}")
    # Batch-composition phase: concurrency reaches code paths single requests
    # never touch (padded/non-uniform decode batches).
    for n in (4, 8):
        probs = concurrent_phase(host, port, n)
        if probs:
            failures += 1
            print(f"  concurrent x{n}  FAIL")
            for p in probs[:4]:
                print(f"            -> {p}")
        else:
            print(f"  concurrent x{n}  OK")

    print(f"=== {'ALL PASSED' if failures == 0 else str(failures) + ' CHECK(S) FAILED'} ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
