#!/usr/bin/env python3
"""P1c/d round-trip on REAL expert-0 tensors from gx10-1 layer 5.

Pipeline: compressed-tensors unpack -> dequant -> per-tile scale-normalize ->
vectorized greedy ring-trellis encode (K=4) -> decode via ported MCG law ->
error metrics recorded verbatim. MemAvailable-gated fail-closed.
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
print(f"[gate] mem available {avail / 2**30:.2f} GiB")

import torch  # noqa: E402
from mcg_codec import mcg_decode_w16  # noqa: E402

torch.set_num_threads(4)
HERE = pathlib.Path(__file__).parent

# ---- 1. dequant compressed-tensors pack-quantized (helpers.py semantics) ----
def unpack_from_int32(value: torch.Tensor, num_bits: int, cols: int) -> torch.Tensor:
    rows, num_words = value.shape
    pad = (-num_words) % num_bits
    if pad:
        value = torch.nn.functional.pad(value, (0, pad))
        num_words += pad
    num_groups = num_words // num_bits
    vg = value.reshape(rows * num_groups, num_bits).to(torch.int64)
    elem = torch.arange(32, dtype=torch.int64)
    bit_starts = elem * num_bits
    word_idx = bit_starts // 32
    off = bit_starts % 32
    lo_bits = torch.clamp(32 - off, max=num_bits)
    out = (vg[:, word_idx] >> off) & ((1 << lo_bits) - 1)
    ov = lo_bits < num_bits
    hi = num_bits - lo_bits[ov]
    right = (vg[:, word_idx[ov] + 1] & ((1 << hi) - 1)) << lo_bits[ov]
    out[:, ov] |= right
    res = out.reshape(rows, num_groups * 32)[:, :cols]
    return (res - (1 << (num_bits - 1))).to(torch.int8)


def dequant(packed, scale, shape):
    rows, cols = shape
    q = unpack_from_int32(packed, 4, packed.size(1) * 8)[:rows].float()
    s = scale.float().repeat_interleave(cols // scale.size(1), dim=1)
    return q[:, :cols] * s


print("[load] pair-experts-0-1.pt")
d = torch.load(HERE / "pair-experts-0-1.pt")
W_down = dequant(d["model.layers.5.mlp.experts.0.down_proj.weight_packed"],
                 d["model.layers.5.mlp.experts.0.down_proj.weight_scale"],
                 tuple(d["model.layers.5.mlp.experts.0.down_proj.weight_shape"].tolist()))
W_gate = dequant(d["model.layers.5.mlp.experts.0.gate_proj.weight_packed"],
                 d["model.layers.5.mlp.experts.0.gate_proj.weight_scale"],
                 tuple(d["model.layers.5.mlp.experts.0.gate_proj.weight_shape"].tolist()))
print(f"[dequant] down {tuple(W_down.shape)} std={W_down.std():.4f} "
      f"| gate {tuple(W_gate.shape)} std={W_gate.std():.4f}")

# ---- 2. codec tables ----
LUT = mcg_decode_w16(torch.arange(1 << 16))          # 65536 fp32 values
CHOICES = torch.arange(16)                            # K=4
BITS = int(os.environ.get("CODEC_K", "4"))
total = BITS * 256
arange16 = torch.arange(16)


def encode_tile(target: torch.Tensor):
    """target (16,16) fp32, pre-normalized into codebook range.
    Greedy over elements t=0..255; window ends at (t+1)*K (proven variant)."""
    ring = torch.zeros(total, dtype=torch.int64)
    flat = target.flatten()
    for t in range(256):
        idx = (((t + 1) * BITS - 16) + arange16) % total
        base = (t * BITS) % total
        field_pos = [(base + b) % total for b in range(BITS)]
        best_e, best_c = None, 0
        for c in range(1 << BITS):
            for b in range(BITS):
                ring[field_pos[b]] = (c >> b) & 1
            w0 = int((ring[idx] * (1 << arange16)).sum())
            e = abs(float(LUT[w0]) - float(flat[t]))
            if best_e is None or e < best_e:
                best_e, best_c = e, c
        for b in range(BITS):
            ring[field_pos[b]] = (best_c >> b) & 1
    words = (ring.view(total // 32, 32) *
             (1 << torch.arange(32))).sum(1)
    return words


def decode_tile(words: torch.Tensor) -> torch.Tensor:
    total = BITS * 256
    w = words.to(torch.int64)
    t = torch.arange(256)
    starts = ((t + 1) * BITS - 16) % total
    idx = (starts.unsqueeze(1) + torch.arange(16)) % total
    wi, bi = idx // 32, idx % 32
    bitsv = (w[wi] >> bi) & 1
    w0 = (bitsv * (1 << torch.arange(16))).sum(1)
    return mcg_decode_w16(w0).view(16, 16)


# ---- 3. round-trip metrics per matrix (tile-parallel greedy is slow; sample) ----
def matrix_roundtrip(W: torch.Tensor, name: str, n_tiles: int = 64):
    Wv = W.view(-1, 256)[: n_tiles * 16]           # row-major tiles of 256
    rels, maxs, scales = [], [], []
    for ti in range(n_tiles):
        tgt = Wv[ti].view(16, 16)
        s = 3.5 / float(tgt.abs().max().clamp_min(1e-8))
        words = encode_tile(tgt * s)
        rec = decode_tile(words) / s
        rels.append(float((tgt - rec).norm() / tgt.norm().clamp_min(1e-8)))
        maxs.append(float((tgt - rec).abs().max()))
        scales.append(s)
    rels, maxs = torch.tensor(rels), torch.tensor(maxs)
    print(f"[rt:{name}] tiles={n_tiles} K={BITS} "
          f"relF p50={rels.median():.4f} mean={rels.mean():.4f} worst={rels.max():.4f} "
          f"maxAbsErr p50={maxs.median():.4f} worst={maxs.max():.4f} "
          f"scale p50={torch.tensor(scales).median():.3f}")
    return float(rels.median()), float(rels.mean()), float(rels.max())


if __name__ == "__main__":
    print("[rt] single-tile sanity")
    matrix_roundtrip(W_gate[:16], "gate-sample", n_tiles=1)
    matrix_roundtrip(W_down[:16], "down-sample", n_tiles=1)
    n = int(os.environ.get("CODEC_TILES", "64"))
    matrix_roundtrip(W_gate, "gate", n_tiles=n)
    matrix_roundtrip(W_down, "down", n_tiles=n)
    print("[done]")
