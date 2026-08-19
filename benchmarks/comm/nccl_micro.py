import os, time, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); W=int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl", rank=rank, world_size=W); torch.cuda.set_device(0); dev="cuda"
tag=os.environ.get("TAG","default")
def bench(fn, iters=300):
    for _ in range(20): fn()
    torch.cuda.synchronize(); dist.barrier(); t=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/iters*1e6
ar=torch.randn(12*6144,device=dev).to(torch.bfloat16)          # ~150 KB: 12 tokens x hidden, the per-layer TP all-reduce at C4
ar1=torch.randn(3*6144,device=dev).to(torch.bfloat16)          # ~37 KB: C1
ag_in=torch.randn(3*16*576,device=dev).to(torch.bfloat16)      # ~55 KB per rank: q-head gather at C1
ag_out=torch.empty(W*ag_in.numel(),device=dev,dtype=torch.bfloat16)
a2a_in=torch.randn(4*3*16*(512+2),device=dev).to(torch.bfloat16); a2a_out=torch.empty_like(a2a_in)
r={}
r["all_reduce_150KB"]=bench(lambda: dist.all_reduce(ar))
r["all_reduce_37KB"]=bench(lambda: dist.all_reduce(ar1))
r["all_gather_55KBx4"]=bench(lambda: dist.all_gather_into_tensor(ag_out, ag_in))
r["all_to_all_50KB"]=bench(lambda: dist.all_to_all_single(a2a_out, a2a_in))
if rank==0: print(f"[{tag}] " + "  ".join(f"{k}={v:.1f}us" for k,v in r.items()), flush=True)
dist.barrier(); dist.destroy_process_group()
