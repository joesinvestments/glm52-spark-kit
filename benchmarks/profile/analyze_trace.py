#!/usr/bin/env python3
"""Decode-step breakdown from a torch-profiler chrome trace (json.gz) of one vLLM rank.
Buckets GPU kernel time by category and reports per-step averages plus GPU idle share."""
import gzip, json, sys, re, collections
path=sys.argv[1]
d=json.load(gzip.open(path))
ev=d["traceEvents"] if isinstance(d,dict) else d
gpu=[e for e in ev if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
if not gpu: print("no gpu events"); sys.exit(0)
CATS=[
 ("all_reduce", r"ncclDevKernel_AllReduce|AllReduce|allreduce|cross_device_reduce|custom_all_reduce"),
 ("all_to_all/all_gather/other_nccl", r"ncclDevKernel|nccl|AllGather|SendRecv|ReduceScatter|Broadcast"),
 ("moe_expert_gemm(marlin)", r"[Mm]arlin|moe_wna16|fused_moe|MoeWNA16|moe_align|topk|grouped_topk|moe_sum|silu_and_mul|_moe_"),
 ("attention_sparse_mla(b12x)", r"b12x|B12X|sparse_mla|mla|MLA|flash|fmha|attn|Attn|paged|indexer|Indexer|topk_indices|cute|CuTe|kernel_cute"),
 ("gemm_dense(attn proj/lm_head)", r"gemm|Gemm|GEMM|cutlass|Cutlass|nvjet|sm90|sm120|xmma|matmul|mm_|bmm|wgmma|cublas"),
 ("norm/rope/elementwise", r"rms_norm|RMSNorm|rotary|rope|layernorm|LayerNorm|elementwise|vectorized|silu|gelu|act|_kernel|copy|fill|cat|index|gather|scatter|reduce_kernel|softmax|arange|where|mul|add|div|sub|cast|convert|Convert|to_copy"),
 ("sampler/mtp/misc", r"."),
]
tot=collections.Counter(); cnt=collections.Counter(); names=collections.defaultdict(collections.Counter)
for e in gpu:
    n=e.get("name",""); dur=e.get("dur",0)
    for c,pat in CATS:
        if re.search(pat,n): tot[c]+=dur; cnt[c]+=1; names[c][n[:70]]+=dur; break
gpu_sorted=sorted(gpu,key=lambda e:e["ts"])
t0=gpu_sorted[0]["ts"]; t1=max(e["ts"]+e["dur"] for e in gpu)
wall=t1-t0
# GPU busy union
busy=0; cur_s=None; cur_e=None
for e in gpu_sorted:
    s,en=e["ts"],e["ts"]+e["dur"]
    if cur_e is None or s>cur_e:
        if cur_e is not None: busy+=cur_e-cur_s
        cur_s,cur_e=s,en
    else: cur_e=max(cur_e,en)
busy+=cur_e-cur_s
# step count: count of MTP/sampler boundaries is hard; use all_reduce kernel count / (layers*2) as step proxy is fragile.
# Instead: count "execute_model" cpu ops if present
cpu=[e for e in ev if e.get("ph")=="X" and e.get("cat") in ("cpu_op","user_annotation","python_function")]
steps=[e for e in cpu if re.search(r"execute_model|sample_tokens|_dummy_run|forward",e.get("name","")) and e.get("name","").count("execute_model")]
nsteps=len(steps) if steps else None
print(f"file: {path.split('/')[-1]}")
print(f"window wall {wall/1000:.1f} ms, GPU busy {busy/1000:.1f} ms ({100*busy/wall:.0f}%), GPU idle {100*(1-busy/wall):.0f}%")
if nsteps: print(f"execute_model calls in window: {nsteps} -> {wall/nsteps/1000:.1f} ms/step wall, {busy/nsteps/1000:.1f} ms/step GPU busy")
print("GPU kernel time by category (sum over window):")
allk=sum(tot.values())
for c,_ in CATS:
    if tot[c]: print(f"  {c:38s} {tot[c]/1000:9.1f} ms  {100*tot[c]/allk:5.1f}%  ({cnt[c]} kernels)" + (f"  {tot[c]/nsteps/1000:.2f} ms/step" if nsteps else ""))
print("top kernels:")
allnames=collections.Counter()
for c in names: allnames.update(names[c])
for n,v in allnames.most_common(12): print(f"  {v/1000:8.1f} ms  {n}")
