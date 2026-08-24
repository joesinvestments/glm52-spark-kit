#!/usr/bin/env python3
"""S3b FEASIBILITY PROBE: does b12x 1.2.x serve a MIXED K3/K4 expert set?

Run inside glm52-collab:b5 (b12x 1.2.6) ON A NODE:
  docker run --rm --gpus all --entrypoint bash -v $PWD:/w glm52-collab:b5 \
      -c "pip install -q pytest >/dev/null; python /w/probe_btx_mixed.py"

Verifies, in order (stop at first failure and print exactly what broke):
  1. import surface: btx schema + prepare + mixed-trellis reachable
  2. synthetic 256-expert weight set -> BTX manifest declaring per-pair rates
     (pair even = K4, pair odd = K3) accepted by BTX_SCHEMA validation
  3. prepare_trellis256_moe_weights / pair-finalizer produces PreparedW4A16MoeWeights
  4. one fused forward vs homogeneous-K4 reference: max rel err bounded,
     and kernel dispatch actually exercised both decoders (assert via counters
     if exposed, else via timing delta vs pure-K3)

Interface assumptions are marked ASSUMPTION-<n> and must be reconciled against
b12x/moe/_shared/btx_schema.py + w4a16/{prepare,btx,mixed_trellis}.py on the
exact pinned commit before trusting a PASS.
"""
import json
import sys
import torch

FAIL = []


def step(n, desc):
    print(f"[{n}] {desc}", flush=True)


def ok(n, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{n}] {tag} {detail}", flush=True)
    if not cond:
        FAIL.append(n)


# ---- 1: imports -------------------------------------------------------------
step(1, "import surface")
try:
    from b12x.moe._shared.btx_schema import (
        BTX_SCHEMA, BtxManifest, RATE_CODE_PAIR_KINDS,
        RATE_STRUCTURE_PER_EXPERT_PAIR, RATE_STRUCTURE_UNIFORM)
    from b12x.moe._shared.kernels.w4a16.btx import load_btx_container  # ASSUMPTION-1 name
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import run_mixed_trellis  # ASSUMPTION-2
    ok(1, True)
except Exception as e:
    ok(1, False, f"{type(e).__name__}: {e}")
    print("FIX: reconcile ASSUMPTION names against installed tree:", *FAIL)
    sys.exit(1)

# ---- 2: synthetic manifest with per-pair mixed rates ------------------------
step(2, "BTX manifest, per-pair K4/K3 rates")
E = 256          # GLM-5.2 experts/layer
PAIRS = E // 2   # ATOMS_PER_PAIR assumed 2 -> ASSUMPTION-3
manifest = {
    "container": "btx-atoms-v1",
    "num_experts": E,
    "atoms_per_pair": 2,                      # ASSUMPTION-3
    "rate_structure": RATE_STRUCTURE_PER_EXPERT_PAIR,
    "rates": [["K4", "K4"] if i % 2 == 0 else ["K3", "K3"] for i in range(PAIRS)],
}
try:
    # ASSUMPTION-4: validation entry accepts plain dicts (else construct dataclass)
    try:
        BTX_SCHEMA.validate(manifest)
        ok(2, True)
    except AttributeError:
        BtxManifest(**manifest)
        ok(2, True)
except Exception as e:
    ok(2, False, f"{type(e).__name__}: {e}")

# ---- 3: preparation path ----------------------------------------------------
step(3, "prepare mixed-rate weights")
torch.manual_seed(0)
w13 = torch.randn(E, 2 * 2048, 6144, dtype=torch.bfloat16)   # ASSUMPTION-5 shapes
w2 = torch.randn(E, 6144, 2048, dtype=torch.bfloat16)
try:
    prepared = None  # real call shape per prepare.py; placeholder until verified
    ok(3, False, "ASSUMPTION-5/6 unverified - fill prepare_trellis256_moe_weights "
                 "call signature from source, then rerun")
except Exception as e:
    ok(3, False, f"{type(e).__name__}: {e}")

# ---- summary ----------------------------------------------------------------
print()
if FAIL:
    print(f"FEASIBILITY: blocked at steps {FAIL} - resolve assumptions above")
    sys.exit(1)
print("FEASIBILITY: mixed K3/K4 BTX path viable end-to-end")
