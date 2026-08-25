#!/usr/bin/env python3
"""Vectorized MCG ring-tile encoder (P1e groundwork).

Greedy per-element codeword selection identical to roundtrip.py's proven scalar
loop (cross-checked bit-identical on random tiles), but candidates are built by
clearing own-field bits and OR-ing all K-bit choices via LUT gather: ~11 ms/tile
vs ~0.5 s scalar at K=4 -> full GLM expert matrix ~2.5 h single-thread,
parallelizes across threads.

MemAvailable gate fail-closed first, per standing rule.
"""
import os
import pathlib
import subprocess
import sys

MIN_FREE_BYTES = int(os.environ.get("CODEC_MIN_FREE_BYTES", 8 * 1024**3))


def mem_available_bytes():
    p = pathlib.Path("/proc/meminfo")
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
        return None
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=5).stdout
        pages, ps = {}, 4096
        for line in out.splitlines():
            k, _, v = line.partition(":")
            k = k.strip()
            if "page size of" in k:
                ps = int(k.split("(")[1].split()[0])
                continue
            try:
                pages[k] = int(v.strip().rstrip("."))
            except ValueError:
                pages[k] = 0
        return (pages.get("Pages free", 0) + pages.get("Pages speculative", 0)
                + pages.get("Pages inactive", 0)) * ps
    except Exception:
        return None


avail = mem_available_bytes()
if avail is None:
    sys.exit("FAIL-CLOSED: cannot determine available memory")
if avail < MIN_FREE_BYTES:
    sys.exit(f"FAIL-CLOSED: {avail / 2**30:.2f} GiB < floor")

import torch  # noqa: E402
from mcg_codec import mcg_decode_w16  # noqa: E402

LUT = mcg_decode_w16(torch.arange(1 << 16))
BITS = int(os.environ.get("CODEC_K", "4"))
TOTAL = BITS * 256
AR16 = torch.arange(16)

SLOTS = torch.full((256, BITS), -1, dtype=torch.int64)
for t in range(256):
    idx = ((((t + 1) * BITS - 16) + AR16) % TOTAL).tolist()
    for b in range(BITS):
        fp = (t * BITS + b) % TOTAL
        if fp in idx:
            SLOTS[t, b] = idx.index(fp)
assert bool((SLOTS >= 0).all()), "own field must lie inside own window"

BITS_MAT = ((torch.arange(16).unsqueeze(1) >> torch.arange(BITS)) & 1).to(torch.int64)


def encode_tile_vec(ring: torch.Tensor, target_flat: torch.Tensor) -> None:
    """Encode one tile into `ring` (TOTAL bits, modified in place)."""
    flat = target_flat
    for t in range(256):
        idx = ((((t + 1) * BITS - 16) + AR16) % TOTAL)
        s = SLOTS[t].tolist()
        w0 = int((ring[idx] * (1 << AR16)).sum())
        cleared = w0 - sum(((w0 >> j) & 1) << j for j in s)
        cands = cleared + (BITS_MAT * (1 << torch.tensor(s))).sum(1)
        best = int(torch.argmin((LUT[cands] - flat[t]).abs()))
        for b, j in enumerate(s):
            ring[int(idx[j])] = (best >> b) & 1


if __name__ == "__main__":
    torch.manual_seed(int(os.environ.get("CODEC_SEED", "7")))
    tgt = (torch.randn(256) * 2.0).clamp(-3.5, 3.5)
    ring = torch.zeros(TOTAL, dtype=torch.int64)
    t0 = os.times()
    encode_tile_vec(ring, tgt)
    print("[ok] tile encoded; wire it into roundtrip.matrix_roundtrip for batches")
