"""DSpark speculator finetune against OUR int4/int8 GLM target (standalone).

Faithful training-forward mirror of the fork's inference path
(qwen3_dflash/qwen3_dspark): context K/V per layer are that layer's k/v
projection of hidden_norm(fc(aux)) — a flat per-layer memory; the query block
(anchor + K masks) stacks through the 5 layers attending non-causally to
[window ; block]. Loss: CE on the K true next tokens with Markov bias
conditioned on the true previous token + BCE on the confidence head.

Data: cap-*.pt files {aux [T, 30720] bf16, input_ids [T], positions [T]}
from VLLM_DSPARK_CAPTURE_DIR (contiguous spans; positions are absolute).

Run (single GB10, needs the GPU — pause serving or use a free window):
  docker run --rm --gpus all --network none \
    -v /home/bird/dspark-capture:/data:ro \
    -v /home/bird/.cache/huggingface/hub/glm52-speculator-dspark:/spec:ro \
    -v /home/bird/dspark-ft-out:/out \
    -v /home/bird/dspark_finetune.py:/ft.py:ro \
    --entrypoint python3 vllm-node-tf5-eldritch-dcp:ring-v7-b12x /ft.py

Env: STEPS, LR, K, WINDOW, BATCH_ANCHORS, DATA_DIR, SPEC_DIR, OUT_DIR.
Output: /out/model.safetensors in the original speculators layout (drop-in:
point the launcher's speculative-config model at a dir containing it plus the
original config.json).
"""
import glob
import json
import math
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

DATA = os.environ.get("DATA_DIR", "/data")
SPEC = os.environ.get("SPEC_DIR", "/spec")
OUT = os.environ.get("OUT_DIR", "/out")
W = int(os.environ.get("WINDOW", "1024"))
K = int(os.environ.get("K", "7"))
LR = float(os.environ.get("LR", "1e-5"))
STEPS = int(os.environ.get("STEPS", "2000"))
BATCH_ANCHORS = int(os.environ.get("BATCH_ANCHORS", "8"))
ACCUM = int(os.environ.get("ACCUM", "4"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "500"))
DEV = os.environ.get("DEV", "cuda")
MIN_A = max(8, int(os.environ.get("MIN_ANCHOR", "8")))
# Multi-node data parallelism (torchrun): shard files by rank, all-reduce
# grads each step, rank 0 logs/saves. WORLD=1 = plain single-node.
RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))

cfg = json.load(open(f"{SPEC}/config.json"))
tl = cfg["transformer_layer_config"]
HID = tl["hidden_size"]            # 6144
NH = tl["num_attention_heads"]     # 64
NKV = tl["num_key_value_heads"]    # 64
HD = tl["head_dim"]                # 64
INTER = tl["intermediate_size"]    # 12288
NLAYERS = tl["num_hidden_layers"]  # 5
EPS = tl["rms_norm_eps"]
THETA = tl["rope_parameters"]["rope_theta"]  # 8e6
VOCAB = tl["vocab_size"]           # 154880
MRANK = cfg["markov_rank"]         # 256
MASK_ID = cfg["mask_token_id"]     # 154856
AUXW = HID * len(cfg["aux_hidden_state_layer_ids"])  # 30720


def rms_norm(x, w, eps=EPS):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x.to(dt) * w).to(dt)


class Rope:
    def __init__(self, dim, theta, max_pos=1 << 21):
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_pos).float()
        f = torch.outer(t, inv)
        self.cos = f.cos().to(DEV)
        self.sin = f.sin().to(DEV)

    def __call__(self, x, pos):
        # x [..., n, HD] neox-style halves; pos [...] absolute
        c = self.cos[pos].unsqueeze(-2)  # [..., 1, HD/2]
        s = self.sin[pos].unsqueeze(-2)
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat(
            [x1 * c - x2 * s, x2 * c + x1 * s], dim=-1
        ).to(x.dtype)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Parameter(torch.empty(HID))
        self.post_attention_layernorm = nn.Parameter(torch.empty(HID))
        self.q_proj = nn.Linear(HID, NH * HD, bias=False)
        self.k_proj = nn.Linear(HID, NKV * HD, bias=False)
        self.v_proj = nn.Linear(HID, NKV * HD, bias=False)
        self.o_proj = nn.Linear(NH * HD, HID, bias=False)
        self.q_norm = nn.Parameter(torch.empty(HD))
        self.k_norm = nn.Parameter(torch.empty(HD))
        self.gate_proj = nn.Linear(HID, INTER, bias=False)
        self.up_proj = nn.Linear(HID, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, HID, bias=False)

    def ctx_kv(self, x_normed, rope, pos):
        # per-layer flat context memory from the shared combined-aux stream
        k = self.k_proj(x_normed).view(*x_normed.shape[:-1], NKV, HD)
        k = rope(rms_norm(k, self.k_norm), pos)
        v = self.v_proj(x_normed).view(*x_normed.shape[:-1], NKV, HD)
        return k, v

    def forward(self, h, ctx_k, ctx_v, rope, pos, keep=None):
        # h [B, G, HID] query block; ctx_k/v [B, Wv, NKV, HD]
        x = rms_norm(h, self.input_layernorm)
        q = self.q_proj(x).view(*x.shape[:-1], NH, HD)
        q = rope(rms_norm(q, self.q_norm), pos)
        k = self.k_proj(x).view(*x.shape[:-1], NKV, HD)
        k = rope(rms_norm(k, self.k_norm), pos)
        v = self.v_proj(x).view(*x.shape[:-1], NKV, HD)
        keys = torch.cat([ctx_k, k], dim=1)
        vals = torch.cat([ctx_v, v], dim=1)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), keys.transpose(1, 2), vals.transpose(1, 2),
            attn_mask=keep, enable_gqa=True,
        )  # non-causal over [window ; block]; keep=True rows attendable
        o = o.transpose(1, 2).reshape(*x.shape[:-1], NH * HD)
        h = h + self.o_proj(o)
        x = rms_norm(h, self.post_attention_layernorm)
        h = h + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return h


class Draft(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HID)
        self.fc = nn.Linear(AUXW, HID, bias=False)
        self.hidden_norm = nn.Parameter(torch.empty(HID))
        self.layers = nn.ModuleList(Layer() for _ in range(NLAYERS))
        self.norm = nn.Parameter(torch.empty(HID))
        self.lm_head = nn.Linear(HID, VOCAB, bias=False)
        self.markov_w1 = nn.Embedding(VOCAB, MRANK)
        self.markov_w2 = nn.Linear(MRANK, VOCAB, bias=False)
        self.conf = nn.Linear(HID + MRANK, 1, bias=True)

    def forward(self, aux_win, pos_win, block_ids, pos_block, pad=None):
        # aux_win [B, Wv, AUXW]; block_ids [B, G]; returns logits [B,G,V], h
        rope = ROPE
        x_ctx = rms_norm(self.fc(aux_win), self.hidden_norm)
        h = self.embed_tokens(block_ids)
        keep = None
        if pad is not None:  # bool SDPA mask: True = attend (pad rows False)
            B, G = block_ids.shape
            Wv = aux_win.shape[1]
            keep = torch.ones(B, 1, G, Wv + G, dtype=torch.bool,
                              device=h.device)
            keep[..., :Wv] = ~pad[:, None, None, :]
        for layer in self.layers:
            ck, cv = layer.ctx_kv(x_ctx, rope, pos_win)
            h = layer.forward(h, ck, cv, rope, pos_block, keep)
        h = rms_norm(h, self.norm)
        return self.lm_head(h), h


NAME_MAP = {
    "embed_tokens.weight": "embed_tokens.weight",
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "hidden_norm",
    "norm.weight": "norm",
    "lm_head.weight": "lm_head.weight",
    "markov_head.markov_w1.weight": "markov_w1.weight",
    "markov_head.markov_w2.weight": "markov_w2.weight",
    "confidence_head.proj.weight": "conf.weight",
    "confidence_head.proj.bias": "conf.bias",
}
for i in range(NLAYERS):
    for a, b in (
        ("input_layernorm.weight", "input_layernorm"),
        ("post_attention_layernorm.weight", "post_attention_layernorm"),
        ("self_attn.q_proj.weight", "q_proj.weight"),
        ("self_attn.k_proj.weight", "k_proj.weight"),
        ("self_attn.v_proj.weight", "v_proj.weight"),
        ("self_attn.o_proj.weight", "o_proj.weight"),
        ("self_attn.q_norm.weight", "q_norm"),
        ("self_attn.k_norm.weight", "k_norm"),
        ("mlp.gate_proj.weight", "gate_proj.weight"),
        ("mlp.up_proj.weight", "up_proj.weight"),
        ("mlp.down_proj.weight", "down_proj.weight"),
    ):
        NAME_MAP[f"layers.{i}.{a}"] = f"layers.{i}.{b}"


def load_model():
    # fp32 master weights — bf16 AdamW at lr 1e-5 rounds updates to zero
    # (relative step ~1e-4 < bf16 eps ~8e-3); forward runs under autocast
    m = Draft().to(DEV)
    sd = {}
    for f in glob.glob(f"{os.environ.get('WEIGHTS_DIR', SPEC)}/*.safetensors"):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k in NAME_MAP:
                    sd[NAME_MAP[k]] = sf.get_tensor(k)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing]
    assert not missing, f"missing: {missing[:6]}"
    return m


def save_model(m, path):
    inv = {v: k for k, v in NAME_MAP.items()}
    out = {}
    for k, t in m.state_dict().items():
        out[inv[k]] = t.detach().to(torch.bfloat16).contiguous().cpu()
    save_file(out, path)


class Spans:
    """Loads capture files; yields (aux_win, pos_win, anchor batch labels)."""

    def __init__(self):
        pool = sorted(glob.glob(f"{DATA}/cap-*.pt"))
        cut = float(os.environ.get("CUTOFF_MTIME", "0"))
        if cut > 0:  # certified corpus-driver captures only (exclude
            # post-swap benchmark traffic, which includes random-text prompts)
            pool = [f for f in pool if os.path.getmtime(f) <= cut]
        random.Random(3).shuffle(pool)  # same shuffle on every rank...
        if WORLD > 1:
            pool = pool[RANK::WORLD]  # ...then disjoint stride-shards
        self.pool = pool
        self.files = list(pool)
        assert self.files, "no capture files"
        print(f"capture files: {len(self.files)} (rank {RANK}/{WORLD})",
              flush=True)
        self.buf = []

    def _load_more(self):
        while len(self.buf) < 4 and self.files:
            f = self.files.pop()
            try:
                d = torch.load(f, map_location="cpu", weights_only=True)
            except Exception:
                continue
            if d["aux"].shape[0] >= max(64, MIN_A + K + 2):
                self.buf.append(d)

    def batch(self, rng):
        self._load_more()
        if not self.buf:
            self.files = list(self.pool)
            random.Random(rng.random()).shuffle(self.files)
            self._load_more()
        d = self.buf.pop(rng.randrange(len(self.buf)))
        T = d["aux"].shape[0]
        aux, ids, pos = d["aux"], d["input_ids"], d["positions"]
        anchors = []
        for _ in range(BATCH_ANCHORS):
            a = rng.randrange(MIN_A, max(MIN_A + 1, T - K - 1))
            anchors.append(a)
        wins, pwins, blocks, pblocks, labels, prevs, pads = (
            [], [], [], [], [], [], [])
        for a in anchors:
            # Window EXCLUDES the anchor's own aux: at inference the anchor is
            # the freshly sampled bonus token — the target never forward-passed
            # it, so its hidden state does not exist in the ring.
            lo = max(0, a - W)
            w_aux = aux[lo:a]
            w_pos = pos[lo:a]
            real = a - lo
            pm = torch.zeros(W, dtype=torch.bool)
            if real < W:  # left-pad; pad rows masked out of attention
                zpad = torch.zeros(W - real, AUXW, dtype=aux.dtype)
                w_aux = torch.cat([zpad, w_aux])
                w_pos = torch.cat([torch.zeros(W - real, dtype=pos.dtype), w_pos])
                pm[: W - real] = True
            block = torch.full((1 + K,), MASK_ID, dtype=ids.dtype)
            block[0] = ids[a]
            bpos = pos[a] + torch.arange(0, 1 + K, dtype=pos.dtype)
            lab = ids[a + 1 : a + 1 + K]
            if lab.shape[0] < K:
                lab = torch.cat(
                    [lab, torch.full((K - lab.shape[0],), -100, dtype=ids.dtype)]
                )
            prev = torch.cat([ids[a : a + 1], lab[:-1]]).clamp_min(0)
            wins.append(w_aux); pwins.append(w_pos); blocks.append(block)
            pblocks.append(bpos); labels.append(lab); prevs.append(prev)
            pads.append(pm)
        st = lambda x: torch.stack(x).to(DEV)
        return (st(wins).to(torch.bfloat16), st(pwins).long(), st(blocks).long(),
                st(pblocks).long(), st(labels).long(), st(prevs).long(),
                st(pads))


class Streams:
    """Stitched request streams from every-step captures (CAPTURE_EVERY=1).

    Files chain into per-request streams keyed by absolute position; a file
    starting at position 0 opens a new stream. Overlapping ranges (decode
    re-drafts after rejection) resolve later-file-wins — the same overwrite
    the serving ring performs on its slots. Windows can then span chunk
    boundaries: full 1024-token windows at any depth, matching serving.
    Each stream's final file is dropped if it is a decode step (its draft
    tokens were never verified).
    """

    def __init__(self):
        names = sorted(glob.glob(f"{DATA}/cap-*.pt"),
                       key=lambda f: int(os.path.basename(f).split("-")[1]))
        cut = float(os.environ.get("CUTOFF_MTIME", "0"))
        if cut > 0:
            names = [f for f in names if os.path.getmtime(f) <= cut]
        assert names, "no capture files"
        raw, cur = [], None
        for f in names:
            try:
                d = torch.load(f, map_location="cpu", weights_only=True)
            except Exception:
                continue
            p = d["positions"]
            if p.numel() == 0:
                continue
            first, n = int(p[0]), p.numel()
            del d
            if first == 0 or cur is None:
                cur = []
                raw.append(cur)
            cur.append((f, first, n))
        self.streams = []
        for s in raw:
            if len(s) > 1 and s[-1][2] < 64:
                s = s[:-1]  # unverified frontier
            idx = {}
            for f, first, n in s:
                for off in range(n):
                    idx[first + off] = (f, off)
            length = (max(idx) + 1) if idx else 0
            if length >= 512 + K + 2:
                self.streams.append((idx, length))
        if WORLD > 1:
            self.streams = self.streams[RANK::WORLD]
        tot = sum(length for _, length in self.streams)
        print(f"streams: {len(self.streams)} tokens: {tot} "
              f"(rank {RANK}/{WORLD})", flush=True)
        assert self.streams, "no streams for this rank"
        # length-weighted stream sampling: without it a handful of deep
        # streams would be drowned out by hundreds of shallow ones
        cap = int(os.environ.get("WEIGHT_CAP", "0") or 0)
        self.weights = [
            min(length, cap) if cap > 0 else length
            for _, length in self.streams
        ]
        self.cache = {}
        self.order = []

    def _file(self, f):
        if f not in self.cache:
            d = torch.load(f, map_location="cpu", weights_only=True)
            self.cache[f] = (d["aux"].to(DEV, torch.bfloat16), d["input_ids"])
            self.order.append(f)
            if len(self.order) > 12:
                self.cache.pop(self.order.pop(0), None)
        return self.cache[f]

    def _window_aux(self, idx, lo, hi):
        # contiguous [lo, hi) as GPU slices grouped by file run
        rows, run_f, run_s, run_e = [], None, 0, 0
        for q in range(lo, hi):
            f, off = idx[q]
            if f == run_f and off == run_e:
                run_e += 1
            else:
                if run_f is not None:
                    rows.append(self._file(run_f)[0][run_s:run_e])
                run_f, run_s, run_e = f, off, off + 1
        if run_f is not None:
            rows.append(self._file(run_f)[0][run_s:run_e])
        return torch.cat(rows) if len(rows) > 1 else rows[0]

    def batch(self, rng):
        idx, length = rng.choices(self.streams, weights=self.weights, k=1)[0]
        hi_a = max(MIN_A + 1, length - K - 2)
        c = rng.randrange(MIN_A, hi_a)
        wins, pwins, blocks, pblocks, labels, prevs, pads = (
            [], [], [], [], [], [], [])
        tries = 0
        while len(wins) < BATCH_ANCHORS and tries < BATCH_ANCHORS * 10:
            tries += 1
            a = min(hi_a - 1, max(MIN_A, c + rng.randrange(-64, 65)))
            lo = max(0, a - W)
            if any(q not in idx for q in range(lo, a + K + 1)):
                c = rng.randrange(MIN_A, hi_a)  # hole: recluster
                continue
            w_aux = self._window_aux(idx, lo, a)
            w_pos = torch.arange(lo, a, dtype=torch.long)
            real = a - lo
            pm = torch.zeros(W, dtype=torch.bool)
            if real < W:
                z = torch.zeros(W - real, AUXW, device=w_aux.device,
                                dtype=w_aux.dtype)
                w_aux = torch.cat([z, w_aux])
                w_pos = torch.cat(
                    [torch.zeros(W - real, dtype=torch.long), w_pos])
                pm[: W - real] = True

            def tid(q):
                f, off = idx[q]
                return int(self._file(f)[1][off])

            block = torch.full((1 + K,), MASK_ID, dtype=torch.long)
            block[0] = tid(a)
            bpos = a + torch.arange(0, 1 + K, dtype=torch.long)
            lab = torch.tensor([tid(a + 1 + i) for i in range(K)],
                               dtype=torch.long)
            prev = torch.cat([block[:1], lab[:-1]]).clamp_min(0)
            wins.append(w_aux); pwins.append(w_pos); blocks.append(block)
            pblocks.append(bpos); labels.append(lab); prevs.append(prev)
            pads.append(pm)
        assert wins, "no valid anchors in stream"
        st = lambda x: torch.stack(x).to(DEV)
        return (st(wins).to(torch.bfloat16), st(pwins).long(),
                st(blocks).long(), st(pblocks).long(), st(labels).long(),
                st(prevs).long(), st(pads))


def make_loader():
    return Streams() if os.environ.get("STITCH") else Spans()


def main():
    global ROPE
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get("DRY"):  # loader-only smoke test (CPU-safe, no model)
        spans = make_loader()
        rng = random.Random(0)
        for i in range(3):
            aux, pwin, block, pblock, labels, prevs, pads = spans.batch(rng)
            print(
                f"batch{i}: aux={tuple(aux.shape)} {aux.dtype} "
                f"pos=[{int(pwin.min())},{int(pwin.max())}] "
                f"anchor0={block[0, 0].item()} "
                f"labels0={labels[0].tolist()} "
                f"pad_mean={pads.float().sum(1).mean().item():.0f}",
                flush=True,
            )
        print("DRY-OK", flush=True)
        return
    ROPE = Rope(HD, THETA)
    m = load_model()
    ac_dev = "cuda" if DEV.startswith("cuda") else "cpu"
    if os.environ.get("EVAL"):  # faithfulness gate: per-position accuracy
        m.eval()
        spans = make_loader()
        rng = random.Random(42)
        nb = int(os.environ.get("EVAL_BATCHES", "30"))
        hits = torch.zeros(K, device=DEV)
        cnt = torch.zeros(K, device=DEV)
        with torch.no_grad():
            for _ in range(nb):
                aux, pwin, block, pblock, labels, prevs, pads = spans.batch(rng)
                with torch.autocast(device_type=ac_dev, dtype=torch.bfloat16):
                    logits, h = m(aux, pwin, block, pblock, pads)
                    bias = m.markov_w2(m.markov_w1(prevs))
                pl = logits[:, 1 : K + 1].float() + bias.float()
                ok = (pl.argmax(-1) == labels) & (labels != -100)
                hits += ok.sum(0).float()
                cnt += (labels != -100).sum(0).float()
        acc = (hits / cnt.clamp_min(1)).cpu().tolist()
        print("per-pos acc: "
              + " ".join(f"p{i+1}={a:.3f}" for i, a in enumerate(acc)),
              flush=True)
        print(f"EVAL mean={float((hits.sum() / cnt.sum()).cpu()):.3f}",
              flush=True)
        return
    # freeze the giant vocab matrices; adapt the transformer + heads
    for p in m.embed_tokens.parameters():
        p.requires_grad_(False)
    for p in m.lm_head.parameters():
        p.requires_grad_(False)
    for p in m.markov_w1.parameters():
        p.requires_grad_(False)
    # frozen matrices don't need fp32 masters — halve their footprint
    m.embed_tokens.to(torch.bfloat16)
    m.lm_head.to(torch.bfloat16)
    m.markov_w1.to(torch.bfloat16)
    if WORLD > 1:
        dist.init_process_group("nccl")
        for p in m.parameters():  # belt-and-braces identical start
            dist.broadcast(p.data, 0)
    train_params = [p for p in m.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in train_params)
    print(f"trainable params: {n_train/1e6:.0f}M", flush=True)
    opt = torch.optim.AdamW(train_params, lr=LR, weight_decay=0.0)
    spans = Spans()
    rng = random.Random(1234 + RANK)
    m.train()
    acc_hist, best_roll = [], -1.0
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        tot_ce = tot_acc = tot_conf = 0.0
        for _ in range(ACCUM):
            aux, pwin, block, pblock, labels, prevs, pads = spans.batch(rng)
            with torch.autocast(device_type=ac_dev, dtype=torch.bfloat16):
                logits, h = m(aux, pwin, block, pblock, pads)
                membed = m.markov_w1(prevs)  # [B, K, MRANK]
                bias = m.markov_w2(membed)
                conf_in = torch.cat([h[:, 1 : K + 1], membed], dim=-1)
                conf = m.conf(conf_in)
            # Bonus-anchor fill-in layout: mask offsets 1..K predict AT their
            # own positions (anchor offset 0 is the bonus token, its output
            # is discarded at inference — sample_off=1 in the triton kernel).
            pred_logits = logits[:, 1 : K + 1].float() + bias.float()
            ce = F.cross_entropy(
                pred_logits.reshape(-1, VOCAB), labels.reshape(-1),
                ignore_index=-100,
            )
            with torch.no_grad():
                ok = (pred_logits.argmax(-1) == labels) & (labels != -100)
            conf = conf.float().squeeze(-1)
            valid = labels != -100
            bce = F.binary_cross_entropy_with_logits(
                conf[valid], ok[valid].float()
            )
            loss = (ce + 0.1 * bce) / ACCUM
            loss.backward()
            tot_ce += ce.item() / ACCUM
            tot_conf += bce.item() / ACCUM
            tot_acc += (ok.sum() / valid.sum().clamp_min(1)).item() / ACCUM
        if WORLD > 1:
            for p in train_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if step % 20 == 0 and RANK == 0:
            print(
                f"step {step} ce={tot_ce:.3f} tok-acc={tot_acc:.3f} "
                f"conf-bce={tot_conf:.3f}",
                flush=True,
            )
        acc_hist.append(tot_acc)
        if len(acc_hist) > 200:
            acc_hist.pop(0)
        if (step % SAVE_EVERY == 0 or step == STEPS) and RANK == 0:
            roll = sum(acc_hist) / len(acc_hist)
            if roll >= best_roll:  # ship the best rolling checkpoint,
                best_roll = roll   # not whatever the last step happens to be
                save_model(m, f"{OUT}/model.safetensors")
                print(f"saved BEST at step {step} (roll-acc {roll:.3f})",
                      flush=True)
            else:
                print(f"skip save at step {step} "
                      f"(roll-acc {roll:.3f} < best {best_roll:.3f})",
                      flush=True)
            if step == STEPS:
                save_model(m, f"{OUT}/model-final.safetensors")
                print("saved at step 2000 (final copy)", flush=True)
    if WORLD > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
