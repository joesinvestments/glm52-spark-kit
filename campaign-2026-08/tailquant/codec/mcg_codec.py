#!/usr/bin/env python3
"""MCG trellis codec core - CPU torch port of exllamav3's decode law + a
greedy ring encoder, plus the compressed-tensors dequant for our checkpoint.

Decode contract (from vendor/exllamav3/ext-quant-src/{exl3_dq.cuh,codebook.cuh}):
  - A 16x16 tile packs 256*K bits into a RING of 256*K bits (16*K uint32 words,
    little-endian bit order: stream bit i = (word[i//32] >> (i%32)) & 1).
  - Element t's codeword = 16 bits ENDING at ring position (t+1)*K, i.e.
    bits [(t+1)*K - 16, (t+1)*K) with wraparound (tail-biting trellis).
  - value = mcg_decode(w0):
        x  = (w0 * 0xCBAC1FED) mod 2^32
        x  = (x & 0x8FFF8FFF) | 0x3B603B60        # lop3 imm 0x6a
        v  = fp16(x >> 16) + fp16(x & 0xFFFF)

MemAvailable gate first, per standing rule.
"""
import pathlib
import subprocess
import sys

MIN_FREE_BYTES = int(pathlib.os.environ.get("CODEC_MIN_FREE_BYTES", 8 * 1024**3))


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
    sys.exit(f"FAIL-CLOSED: {avail / 2**30:.2f} GiB < {MIN_FREE_BYTES / 2**30:.0f} GiB floor")
print(f"[gate] mem available {avail / 2**30:.2f} GiB")

import torch  # noqa: E402

torch.manual_seed(0)


# ---------------- MCG decode ----------------
def mcg_decode_w16(w0: torch.Tensor) -> torch.Tensor:
    """w0: int64 tensor of 16-bit codewords -> fp32 values."""
    x = (w0.to(torch.int64) * 0xCBAC1FED) & 0xFFFFFFFF
    x = ((x & 0x8FFF8FFF) | 0x3B603B60)
    hi = ((x >> 16) & 0xFFFF).to(torch.uint16)
    lo = (x & 0xFFFF).to(torch.uint16)
    return hi.view(torch.int16).view(torch.float16).float() + \
        lo.view(torch.int16).view(torch.float16).float()


def mcg_table256() -> torch.Tensor:
    """The K=8-style table: window = single byte in low bits (high bits zero).

    Not used by the ring codec directly; kept for inspection/validation."""
    return mcg_decode_w16(torch.arange(256))


# ---------------- ring tile decode ----------------
def tile_decode(bits: int, words: torch.Tensor) -> torch.Tensor:
    """words: uint32 tensor shape (16*K/32,) little-endian bit ring.
    Returns (16,16) fp32 tile: element[t, c]? NOTE layout: element index
    t_offset enumerates the 256 weights of the tile in PACKED order; we define
    t = r*16 + c consistent between encode/decode (layout note in DESIGN v3)."""
    total_bits = bits * 256
    n_words = total_bits // 32
    w = words.to(torch.int64) & 0xFFFFFFFF

    def ring_bit(i):
        i = i % total_bits
        return (w[i // 32] >> (i % 32)) & 1

    outs = torch.empty(256)
    # vectorized: gather 16 bits per element
    t = torch.arange(256)
    starts = ((t + 1) * bits - 16) % total_bits
    idx = (starts.unsqueeze(1) + torch.arange(16)) % total_bits
    wi, bi = idx // 32, idx % 32
    bitsv = (w[wi] >> bi) & 1
    w0 = (bitsv * (1 << torch.arange(16))).sum(1)
    outs = mcg_decode_w16(w0)
    return outs.view(16, 16)


# ---------------- ring tile encode (greedy + one wraparound refinement) ----
def _mcg_lut16():
    lut = mcg_decode_w16(torch.arange(1 << 16))
    return lut


_LUT = None


def tile_encode(target: torch.Tensor, bits: int, refine_passes: int = 1):
    """target: (16,16) fp32. Chooses 256*K ring bits greedily element-by-element;
    elements ordered t=0..255 with t=r*16+c. Later passes re-encode early
    elements once their left-context bits exist (tail-biting mitigation)."""
    global _LUT
    if _LUT is None:
        _LUT = _mcg_lut16()
    total_bits = bits * 256
    ring = torch.zeros(total_bits, dtype=torch.uint8)
    order = list(range(256))
    flat = target.flatten()

    def encode_element(t):
        lo = ((t + 1) * bits - 16) % total_bits
        base = (t * bits) % total_bits  # start of OWN field
        best_val, best_bits = None, None
        for choice in range(1 << bits):
            trial = ring.clone()
            for b in range(bits):
                trial[(base + b) % total_bits] = (choice >> b) & 1
            idx = (lo + torch.arange(16)) % total_bits
            w0 = int(sum(int(trial[i]) << j for j, i in enumerate(idx.tolist())))
            val = float(_LUT[w0])
            err = abs(val - float(flat[t]))
            if best_val is None or err < best_val:
                best_val, best_bits = err, choice
        for b in range(bits):
            ring[(base + b) % total_bits] = (best_bits >> b) & 1
        return best_val

    for p in range(refine_passes + 1):
        for t in order:
            encode_element(t)
    # pack ring -> uint32 words (bit i of stream -> word i//32 bit i%32)
    bitsl = ring.tolist()
    words = []
    for wi in range(total_bits // 32):
        v = 0
        for b in range(32):
            v |= bitsl[wi * 32 + b] << b
        words.append(v)
    return torch.tensor(words, dtype=torch.int64), ring


if __name__ == "__main__":
    tab = mcg_table256()
    print("[mcg-table] min/max/mean:", float(tab.min()), float(tab.max()),
          float(tab.mean()))
    print("[mcg-table] sample:", tab[:8].tolist())
