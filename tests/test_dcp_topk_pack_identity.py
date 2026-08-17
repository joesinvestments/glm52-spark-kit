"""Prove the torch (score, global_id) pack used by _merge_b12x_dcp_topk is
byte-identical to the stock Triton pack kernel, for every DCP rank and both
interleave sizes, including -1 padding rows."""
import torch
from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
    pack_dcp_topk_candidates_cutedsl,
)
dev = torch.device("cuda")
torch.manual_seed(0)
def mine(topk_scores, topk_indices, dcp_rank, world, interleave):
    valid = topk_indices >= 0
    idx = topk_indices.clamp(min=0).to(torch.int64)
    gid = (idx // interleave) * (world * interleave) + dcp_rank * interleave + idx % interleave
    gid = torch.where(valid, gid, torch.full_like(gid, -1))
    sc = topk_scores.to(torch.float32)
    sc = torch.where(valid, sc, torch.full_like(sc, float("-inf")))
    return torch.stack([sc, gid.to(torch.float32)], dim=-1).contiguous()

rows, K, L = 37, 2048, 6000          # L = local KV length per rank
for world in (2, 4):
    for interleave in (1, 64):
        for rank in range(world):
            logits = torch.randn(rows, L, device=dev, dtype=torch.float32)
            # local top-k the way a real kernel would produce them, then poke -1 pads
            vals, idx = logits.topk(K, dim=1)
            idx = idx.to(torch.int32).contiguous()
            idx[3, 100:] = -1; idx[10, :] = -1
            # scores aligned to idx (what b12x out_scores gives): gather from logits
            scores = torch.gather(logits, 1, idx.clamp(min=0).long())
            ref = torch.empty(rows, K, 2, device=dev, dtype=torch.float32)
            pack_dcp_topk_candidates_cutedsl(logits, idx, ref, rank, world, interleave, None)
            got = mine(scores, idx, rank, world, interleave)
            assert torch.equal(got, ref), f"MISMATCH world={world} il={interleave} rank={rank}"
print("torch pack == stock triton pack, byte-identical: world 2/4 x interleave 1/64 x every rank, with -1 pads")
# capture-safety of the pack (it runs inside the captured decode step)
scores = torch.randn(16, K, device=dev); idx = torch.randint(0, L, (16, K), device=dev, dtype=torch.int32)
s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s_):
    for _ in range(2): mine(scores, idx, 1, 4, 1)
torch.cuda.current_stream().wait_stream(s_)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): out = mine(scores, idx, 1, 4, 1)
g.replay(); torch.cuda.synchronize()
assert torch.equal(out, mine(scores, idx, 1, 4, 1))
print("pack is CUDA-graph capturable and replay == eager")
