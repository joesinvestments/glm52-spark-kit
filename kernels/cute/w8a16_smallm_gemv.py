"""CuTe DSL small-M W8A16 (int8, symmetric, group-128 bf16 scales) GEMV/GEMM for the decode path.

Target: the dense projections (o_proj, q_b, kv_b, qkv_a, shared expert) at M <= 16 tokens per step,
where the fleet's Marlin w8a16 path runs ~1.7x above the byte floor. Design: bandwidth-bound,
warp-per-output-row, each lane streams packed int32 words (4 x int8) across K, one group scale per
iteration, fp32 accumulation for all M tokens, butterfly reduce, bf16 store.

Weight layout = compressed-tensors "pack-quantized" as in the checkpoint: W_packed[N, K/4] int32
(4 int8 per word, little-endian: byte j -> k = 4*word + j), scales[N, K/128] bf16.

Usage (inside the serving image, GPU): python3 w8a16_smallm_gemv.py --weights /w/cute
"""
import argparse, time, torch
import cutlass, cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack
from cutlass import Int32, Int64, Float32, BFloat16

MAXM = 4    # compile-time token tile; the harness compiles per M bucket (4/8/16)
LANES = 32


ROWS = 8    # rows per warp (activation loads amortized over ROWS)
WARPS = 4   # warps per CTA -> 32 rows per CTA


@cute.kernel
def gemv_kernel(mW: cute.Tensor, mS: cute.Tensor, mX: cute.Tensor, mY: cute.Tensor,
                M: Int32, KQ: Int32):
    # mW: [N, KQ] int64 (8 int8 per element); mS: [N, KQ/16] bf16 (group 128 = 16 int64); mX: [M, 8*KQ] bf16; mY: [M, N] bf16
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % LANES
    warp = tidx // LANES
    row0 = (bidx * WARPS + warp) * ROWS
    acc = cute.make_rmem_tensor((ROWS * MAXM,), Float32)
    for i in cutlass.range_constexpr(ROWS * MAXM):
        acc[i] = Float32(0.0)
    xs = cute.make_rmem_tensor((MAXM * 16,), Float32)
    niter = KQ // (LANES * 2)           # each iteration: lane loads 2 consecutive int64 = 16 k; warp covers 512 k = 4 groups
    for it in cutlass.range(niter):
        q0 = (it * LANES + lane) * 2    # first int64 index of this lane
        k0 = q0 * 8
        gidx = it * 4 + lane // 8       # group index (128 k) for this lane's 16 k
        for m in cutlass.range_constexpr(MAXM):
            if m < M:
                for e in cutlass.range_constexpr(16):
                    xs[m * 16 + e] = Float32(mX[m, k0 + e])
            else:
                for e in cutlass.range_constexpr(16):
                    xs[m * 16 + e] = Float32(0.0)
        for r in cutlass.range_constexpr(ROWS):
            wa = mW[row0 + r, q0]
            wb = mW[row0 + r, q0 + 1]
            sc = Float32(mS[row0 + r, gidx])
            b = cute.make_rmem_tensor((16,), Float32)
            lo = Int32(wa & Int64(0xFFFFFFFF)); hi = Int32(wa >> Int64(32))
            b[0] = Float32(Int32((lo << 24) >> 24)); b[1] = Float32(Int32((lo << 16) >> 24))
            b[2] = Float32(Int32((lo << 8) >> 24));  b[3] = Float32(Int32(lo >> 24))
            b[4] = Float32(Int32((hi << 24) >> 24)); b[5] = Float32(Int32((hi << 16) >> 24))
            b[6] = Float32(Int32((hi << 8) >> 24));  b[7] = Float32(Int32(hi >> 24))
            lo = Int32(wb & Int64(0xFFFFFFFF)); hi = Int32(wb >> Int64(32))
            b[8] = Float32(Int32((lo << 24) >> 24)); b[9] = Float32(Int32((lo << 16) >> 24))
            b[10] = Float32(Int32((lo << 8) >> 24)); b[11] = Float32(Int32(lo >> 24))
            b[12] = Float32(Int32((hi << 24) >> 24)); b[13] = Float32(Int32((hi << 16) >> 24))
            b[14] = Float32(Int32((hi << 8) >> 24)); b[15] = Float32(Int32(hi >> 24))
            for m in cutlass.range_constexpr(MAXM):
                if m < M:
                    d = Float32(0.0)
                    for e in cutlass.range_constexpr(16):
                        d = d + b[e] * xs[m * 16 + e]
                    acc[r * MAXM + m] = acc[r * MAXM + m] + sc * d
    for i in cutlass.range_constexpr(ROWS * MAXM):
        v = acc[i]
        for off in cutlass.range_constexpr(5):
            v = v + cute.arch.shuffle_sync_bfly(v, offset=(1 << (4 - off)))
        acc[i] = v
    if lane == 0:
        for r in cutlass.range_constexpr(ROWS):
            for m in cutlass.range_constexpr(MAXM):
                if m < M:
                    mY[m, row0 + r] = BFloat16(acc[r * MAXM + m])


@cute.jit
def gemv_launch(mW: cute.Tensor, mS: cute.Tensor, mX: cute.Tensor, mY: cute.Tensor,
                M: Int32, KQ: Int32, N: Int32, stream):
    gemv_kernel(mW, mS, mX, mY, M, KQ).launch(grid=(N // (WARPS * ROWS), 1, 1), block=(WARPS * LANES, 1, 1), stream=stream)


def unpack_int8(wp: torch.Tensor) -> torch.Tensor:
    # [N, KW] int32 -> [N, 4*KW] int8 (byte j of word -> k = 4*word + j)
    N, KW = wp.shape
    out = torch.empty(N, KW * 4, dtype=torch.int8, device=wp.device)
    for j in range(4):
        out[:, j::4] = ((wp >> (8 * j)) & 0xFF).to(torch.uint8).view(torch.int8)
    return out


def reference(wp, scales, x):
    w = unpack_int8(wp).float()                                   # [N, K]
    s = scales.float().repeat_interleave(128, dim=1)              # [N, K]
    return (x.float() @ (w * s).T)                                # [M, N] fp32


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", default="/w/cute"); ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--iters", type=int, default=50); a = ap.parse_args()
    dev = "cuda"
    wp = torch.load(f"{a.weights}/weight_packed.pt").to(dev)      # [6144, 4096] int32
    sc = torch.load(f"{a.weights}/weight_scale.pt").to(dev)       # [6144, 128] bf16
    N, KW = wp.shape; K = KW * 4; M = a.M
    torch.manual_seed(0)
    x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    ref = reference(wp, sc, x)
    # torch bf16 baseline: dequant once (as a bf16 weight) + matmul (what a naive path costs)
    w_bf16 = (unpack_int8(wp).float() * sc.float().repeat_interleave(128, dim=1)).to(torch.bfloat16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    wq = wp.view(torch.int64)                                    # [N, KW/2]
    KQ = wq.shape[1]
    mW = from_dlpack(wq, assumed_align=16); mS = from_dlpack(sc, assumed_align=16)
    mX = from_dlpack(x, assumed_align=16); mY = from_dlpack(y, assumed_align=16)
    compiled = cute.compile(gemv_launch, mW, mS, mX, mY, Int32(M), Int32(KQ), Int32(N), stream)
    compiled(mW, mS, mX, mY, Int32(M), Int32(KQ), Int32(N), stream); torch.cuda.synchronize()
    err = (y.float() - ref).abs().max().item(); rel = err / ref.abs().max().item()
    print(f"N={N} K={K} M={M}: max abs err {err:.4e} (rel to max {rel:.2e})")
    def bench(fn, iters):
        for _ in range(5): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1e3
    t_cute = bench(lambda: compiled(mW, mS, mX, mY, Int32(M), Int32(KQ), Int32(N), stream), a.iters)
    t_bf16 = bench(lambda: torch.matmul(x, w_bf16.T), a.iters)
    bytes_int8 = wp.numel() * 4 + sc.numel() * 2
    print(f"cute w8a16 gemv: {t_cute:.3f} ms  ({bytes_int8/1e9/(t_cute/1e3):.0f} GB/s effective on {bytes_int8/1e6:.0f} MB)")
    print(f"torch bf16 matmul (2x bytes): {t_bf16:.3f} ms  ({w_bf16.numel()*2/1e9/(t_bf16/1e3):.0f} GB/s)")
    print(f"byte floor at 273 GB/s: {bytes_int8/273e9*1e3:.3f} ms")


if __name__ == "__main__":
    main()
