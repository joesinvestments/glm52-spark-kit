# Time vLLM's Marlin W8A16 (compressed-tensors int8 g128) on the same o_proj weight at M=3.
import torch, time, sys
from vllm.model_executor.layers.quantization.utils.marlin_utils import (marlin_permute_scales, marlin_make_workspace_new, marlin_make_empty_g_idx, marlin_sort_g_idx)
from vllm.scalar_type import scalar_types
import vllm._custom_ops as ops
dev="cuda"
wp=torch.load("/w/weight_packed.pt").to(dev); sc=torch.load("/w/weight_scale.pt").to(dev)
N,KW=wp.shape; K=KW*4; M=3
# unpack to int8 [N,K] then to the layout gptq_marlin_repack expects: [K/pack, N] int32 packed along K (GPTQ style, 4 int8 per int32)
w=torch.empty(N,K,dtype=torch.int8,device=dev)
for j in range(4): w[:, j::4]=((wp>>(8*j))&0xFF).to(torch.uint8).view(torch.int8)
wt=w.T.contiguous()                      # [K,N] int8
# pack along K into int32 (GPTQ layout: 4 int8 per int32, low byte first), then repack for marlin
u=(wt.to(torch.int32)&0xFF)
packed=(u[0::4]|(u[1::4]<<8)|(u[2::4]<<16)|(u[3::4]<<24)).to(torch.int32).contiguous()   # [K/4, N]
qtype=scalar_types.uint8b128
try:
    mw=ops.gptq_marlin_repack(packed, perm=torch.empty(0,dtype=torch.int32,device=dev), size_k=K, size_n=N, num_bits=8)
except TypeError:
    mw=ops.gptq_marlin_repack(packed, torch.empty(0,dtype=torch.int32,device=dev), K, N, 8)
ms=marlin_permute_scales(sc.T.contiguous().to(torch.bfloat16), size_k=K, size_n=N, group_size=128)   # [K/128, N]
ws=marlin_make_workspace_new(dev)
g_idx=marlin_make_empty_g_idx(dev); g_idx_sort=marlin_make_empty_g_idx(dev)
x=(torch.randn(M,K,device=dev)*0.5).to(torch.bfloat16)
def run():
    return ops.marlin_gemm(x, None, mw, None, ms, None, None, None, g_idx, g_idx_sort, ws, qtype, size_m=M, size_n=N, size_k=K, is_k_full=True, use_atomic_add=True, use_fp32_reduce=True, is_zp_float=False)
try:
    y=run()
except TypeError as e:
    print("signature mismatch:", e); import inspect; print(inspect.signature(ops.gptq_marlin_gemm)); sys.exit(1)
ref=(x.float() @ (w.float()*sc.float().repeat_interleave(128,dim=1)).T)
print("marlin max rel err", ((y.float()-ref).abs().max()/ref.abs().max()).item())
for _ in range(5): run()
torch.cuda.synchronize(); t=time.perf_counter()
for _ in range(50): run()
torch.cuda.synchronize(); dt=(time.perf_counter()-t)/50*1e3
print(f"marlin w8a16 M={M}: {dt:.3f} ms ({(wp.numel()*4+sc.numel()*2)/1e9/(dt/1e3):.0f} GB/s effective)")
