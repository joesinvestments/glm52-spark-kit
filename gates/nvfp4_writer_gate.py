#!/usr/bin/env python3
"""Build/boot gate: the fused nvfp4_ds_mla writer must be byte-identical to the torch
reference, CUDA-graph replayable, and pass torch.library.opcheck. Exit 1 on any drift.

Pattern borrowed from AEON's nvfp4_kv_gate.py: a layout regression must fail the build,
not surface as garbage attention at 100K tokens. Run inside the serving image:
  docker run --rm --gpus all -v $KIT/kernels:/w:ro -v $KIT/gates:/g:ro --entrypoint python3 IMAGE /g/nvfp4_writer_gate.py
"""
import sys, torch
sys.path.insert(0, "/w")
import nvfp4_ds_mla_writer as m

def fail(msg):
    print("GATE FAIL:", msg); sys.exit(1)

dev = "cuda"; torch.manual_seed(0)
bs = 64
for n, ls_val in ((1, 1.0), (7, 0.7), (64, 1.0), (1024, 0.7)):
    blocks = max(8, (n // bs) * 2 + 8)   # slot pool must exceed n (slot 0 reserved for pads)
    kv_c = torch.randn(n, m.NOPE_DIM, device=dev, dtype=torch.bfloat16) * 2.0
    k_pe = torch.randn(n, m.ROPE_DIM, device=dev, dtype=torch.bfloat16) * 1.5
    if n >= 64:
        kv_c[3, :16] = 0            # exact-zero group
        kv_c[5, 100] = 300.0        # outlier
    slots = (torch.randperm(blocks * bs - 1, device=dev)[:n] + 1).to(torch.int64)
    if n > 2: slots[2] = -1        # pad row -> null block
    ls = torch.tensor(ls_val, device=dev)
    a = torch.zeros(blocks, bs, m.RECORD_FP8_ROPE, dtype=torch.uint8, device=dev); b = a.clone()
    m.write_to_kv_cache(a, kv_c, k_pe, slots, latent_scale=ls)
    m._write_to_kv_cache_torch(b, kv_c, k_pe, slots, latent_scale=ls)
    if not torch.equal(a, b): fail(f"fused != torch reference at n={n} latent_scale={ls_val}")
    # round trip cosine
    rec = m.pack_records(kv_c, k_pe, latent_scale=ls_val)
    nope, _ = m.unpack_records(rec)
    cos = torch.nn.functional.cosine_similarity(nope.float(), (kv_c.float() / ls_val), dim=-1).min().item()
    if cos < 0.98: fail(f"round-trip cosine {cos:.4f} < 0.98 at n={n}")
# capture + replay
blocks = 8
kv_c = torch.randn(64, m.NOPE_DIM, device=dev, dtype=torch.bfloat16); k_pe = torch.randn(64, m.ROPE_DIM, device=dev, dtype=torch.bfloat16)
slots = (torch.arange(64, device=dev) + 1).to(torch.int64); ls = torch.tensor(1.0, device=dev)
ref = torch.zeros(blocks, bs, m.RECORD_FP8_ROPE, dtype=torch.uint8, device=dev); m.write_to_kv_cache(ref, kv_c, k_pe, slots, latent_scale=ls)
g_out = torch.zeros_like(ref); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s): m.write_to_kv_cache(g_out, kv_c, k_pe, slots, latent_scale=ls)   # warm
torch.cuda.current_stream().wait_stream(s); g_out.zero_(); g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): m.write_to_kv_cache(g_out, kv_c, k_pe, slots, latent_scale=ls)
g_out.zero_(); g.replay(); torch.cuda.synchronize()
if not torch.equal(g_out, ref): fail("CUDA-graph replay != eager")
# registered op + opcheck
if m.register_op():
    c = torch.zeros_like(ref); torch.ops.vllm.nvfp4_ds_mla_write(c, kv_c, k_pe, slots, ls)
    if not torch.equal(c, ref): fail("torch.ops.vllm.nvfp4_ds_mla_write != reference")
    torch.library.opcheck(torch.ops.vllm.nvfp4_ds_mla_write, (c.clone(), kv_c, k_pe, slots, ls))
    print("opcheck: PASSED")
else:
    print("opcheck: skipped (vLLM registration helper unavailable)")
print("GATE PASS: nvfp4_ds_mla writer byte-identical, replayable, registered")
