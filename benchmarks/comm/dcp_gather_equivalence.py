# 4-rank equivalence test over the rails: two DCP all-gathers vs the fused byte-packed gather + stable top-k.
import os, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); W=int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl", rank=rank, world_size=W)
torch.cuda.set_device(0); dev="cuda"
torch.manual_seed(1234+rank)
B, qh, qd, topk = 3, 16, 576, 2048
mqa_q=torch.randn(B,qh,qd,device=dev).to(torch.bfloat16)
packed=torch.stack([torch.randn(B,topk,device=dev), torch.randint(0,100000,(B,topk),device=dev).float()],dim=-1).contiguous()  # (B,topk,2) fp32
def ag(x, dim):  # vLLM GroupCoordinator.all_gather semantics
    out=torch.empty((W,)+tuple(x.shape),dtype=x.dtype,device=dev); dist.all_gather_into_tensor(out, x.contiguous())
    return out.movedim(0,dim).reshape(x.shape[:dim]+(W*x.shape[dim],)+x.shape[dim+1:])
# path A
gq_a=ag(mqa_q,1); gp_a=ag(packed,1)
# path B (fused, as in the overlay)
qb=mqa_q.reshape(B,qh*qd).contiguous().view(torch.uint8); pb=packed.reshape(B,topk*2).contiguous().view(torch.uint8)
buf=torch.cat([qb,pb],dim=1); g=ag(buf,1); qbytes=qb.shape[1]; pbytes=pb.shape[1]
g=g.view(B,W,qbytes+pbytes)
gq_b=g[:,:,:qbytes].reshape(B,W*qbytes).contiguous().view(mqa_q.dtype).view(B,W*qh,qd)
gp_b=g[:,:,qbytes:].reshape(B,W*pbytes).contiguous().view(torch.float32).view(B,W*topk,2)
same_q=torch.equal(gq_a,gq_b); same_p=torch.equal(gp_a,gp_b)
# stable topk on both
try:
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import stable_topk_from_gathered_candidates_cutedsl as st
    oa=torch.empty(B,topk,dtype=torch.int32,device=dev); ob=torch.empty(B,topk,dtype=torch.int32,device=dev)
    st(gp_a,topk,out=oa); st(gp_b,topk,out=ob); same_t=torch.equal(oa,ob)
except Exception as e:
    same_t=f"topk-skipped ({e.__class__.__name__}: {str(e)[:60]})"
print(f"rank{rank}: gathered_q identical={same_q} gathered_candidates identical={same_p} topk identical={same_t}", flush=True)
dist.barrier(); dist.destroy_process_group()
