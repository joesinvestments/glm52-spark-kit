import sys, torch
sys.path.insert(0, "/w")
import nvfp4_ds_mla_writer as m
assert m.register_op(), "vLLM registration helper unavailable"
dev = "cuda"
torch.manual_seed(0)
n, blocks, bs = 37, 8, 64
kv_c = torch.randn(n, m.NOPE_DIM, device=dev, dtype=torch.bfloat16)
k_pe = torch.randn(n, m.ROPE_DIM, device=dev, dtype=torch.bfloat16)
slots = torch.randperm(blocks * bs, device=dev)[:n].to(torch.int64); slots[3] = -1
ls = torch.tensor(0.7, device=dev)
a = torch.zeros(blocks, bs, m.RECORD_FP8_ROPE, dtype=torch.uint8, device=dev)
b = a.clone()
torch.ops.vllm.nvfp4_ds_mla_write(a, kv_c, k_pe, slots, ls)
m._write_to_kv_cache_torch(b, kv_c, k_pe, slots, latent_scale=ls)
assert torch.equal(a, b), "op != torch reference"
print("torch.ops.vllm.nvfp4_ds_mla_write == reference: byte-identical")
# the vLLM kernel bar
torch.library.opcheck(torch.ops.vllm.nvfp4_ds_mla_write, (a.clone(), kv_c, k_pe, slots, ls))
print("opcheck PASSED (schema, fake, autograd-registration, aliasing)")
# fake tracing under FakeTensorMode
from torch._subclasses.fake_tensor import FakeTensorMode
with FakeTensorMode():
    fa = torch.zeros(blocks, bs, m.RECORD_FP8_ROPE, dtype=torch.uint8, device=dev)
    torch.ops.vllm.nvfp4_ds_mla_write(fa, torch.empty(n, m.NOPE_DIM, device=dev, dtype=torch.bfloat16),
        torch.empty(n, m.ROPE_DIM, device=dev, dtype=torch.bfloat16), torch.empty(n, dtype=torch.int64, device=dev), torch.empty((), device=dev))
print("fake impl traces under FakeTensorMode")
print("OP REGISTRATION PROVEN")
