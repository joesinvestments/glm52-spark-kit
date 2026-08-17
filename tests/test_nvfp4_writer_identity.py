import sys, torch; sys.path.insert(0, "/w")
import nvfp4_ds_mla_writer as m
dev = torch.device("cuda"); torch.manual_seed(1)
assert m._HAVE_TRITON, "triton path not active"
kv_c = torch.randn(512, 512, device=dev, dtype=torch.bfloat16); k_pe = torch.randn(512, 64, device=dev, dtype=torch.bfloat16)
slots = torch.arange(64, 576, device=dev); slots[-3:] = -1
a = torch.zeros(20, 64, 368, dtype=torch.uint8, device=dev); b = torch.zeros_like(a)
m.write_to_kv_cache(a, kv_c, k_pe, slots)                      # dispatcher -> fused
m._write_to_kv_cache_torch(b, kv_c, k_pe, slots)               # reference
assert torch.equal(a.view(-1,368)[1:], b.view(-1,368)[1:]); print("dispatcher(fused) == torch reference: byte-identical")
n, r = m.unpack_records(a.view(-1,368)[64:573])
cos = torch.nn.functional.cosine_similarity(n, kv_c[:509].float(), dim=-1).min().item(); print(f"round-trip cosine min {cos:.5f}"); assert cos > 0.98
ls = torch.tensor(1.0, device=dev)  # created OUTSIDE capture, like the backend's k_scale
g = torch.cuda.CUDAGraph(); s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s_):
    for _ in range(2): m.write_to_kv_cache(a, kv_c, k_pe, slots, latent_scale=ls)
torch.cuda.current_stream().wait_stream(s_)
with torch.cuda.graph(g): m.write_to_kv_cache(a, kv_c, k_pe, slots, latent_scale=ls)
a.zero_(); g.replay(); torch.cuda.synchronize(); assert torch.equal(a.view(-1,368)[1:], b.view(-1,368)[1:]); print("captured + replay == reference")
print("MERGED WRITER MODULE PROVEN")
