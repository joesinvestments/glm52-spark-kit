import os, time, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); W=int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl", rank=rank, world_size=W); torch.cuda.set_device(0); dev="cuda"
def bench(fn, iters=300):
    for _ in range(20): fn()
    torch.cuda.synchronize(); dist.barrier(); t=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/iters*1e6
peers=[p for p in range(W) if p!=rank]
def p2p_allreduce(x, recv):
    ops=[]
    for i,p in enumerate(peers):
        ops.append(dist.P2POp(dist.isend, x, p)); ops.append(dist.P2POp(dist.irecv, recv[i], p))
    for w in dist.batch_isend_irecv(ops): w.wait()
    x.add_(recv[0]).add_(recv[1]).add_(recv[2])
def p2p_allgather(x, out):
    ops=[]
    for p in peers:
        ops.append(dist.P2POp(dist.isend, x, p)); ops.append(dist.P2POp(dist.irecv, out[p], p))
    for w in dist.batch_isend_irecv(ops): w.wait()
    out[rank].copy_(x)
res={}
for label,n in [("150KB",12*6144),("37KB",3*6144)]:
    x=torch.randn(n,device=dev).to(torch.bfloat16); recv=[torch.empty_like(x) for _ in peers]
    res[f"AR_{label}_nccl"]=bench(lambda: dist.all_reduce(x))
    res[f"AR_{label}_p2p"]=bench(lambda: p2p_allreduce(x,recv))
n=3*16*576; x=torch.randn(n,device=dev).to(torch.bfloat16); out=torch.empty(W,n,device=dev,dtype=torch.bfloat16)
res["AG_55KB_nccl"]=bench(lambda: dist.all_gather_into_tensor(out.view(-1), x))
res["AG_55KB_p2p"]=bench(lambda: p2p_allgather(x,out))
# correctness of p2p allreduce vs nccl
y=torch.randn(3*6144,device=dev).to(torch.bfloat16); y1=y.clone(); y2=y.clone(); rc=[torch.empty_like(y) for _ in peers]
dist.all_reduce(y1); p2p_allreduce(y2,rc); ok=torch.allclose(y1.float(),y2.float(),atol=1e-1)
if rank==0: print("[p2p] "+"  ".join(f"{k}={v:.1f}us" for k,v in res.items())+f"  ar_match={ok}", flush=True)
dist.barrier(); dist.destroy_process_group()
