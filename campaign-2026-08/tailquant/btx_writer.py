#!/usr/bin/env python3
"""TailQuant stage 3a: plan.json -> mixed-rate BTX checkpoint (b12x >= 1.2.x).

Drives b12x.moe._shared.kernels.w4a16.btx_synth.write_btx_checkpoint with
per-expert-pair rate tables derived from a TailQuant split plan:
  hot experts -> P44 (4+4)
  cold experts -> chosen pair kind (default P33 = 3+3)

Two modes:
  --emit-config out.json : dump the BtxSynthConfig fields (no b12x needed)
  --write OUT_DIR        : build config and call write_btx_checkpoint
                           (requires b12x + torch; runs anywhere, no GPU)

Geometry defaults match GLM-5.2 (256 experts); pass --hidden/--intermediate/
--layers to match your checkpoint config.json exactly.
"""
import argparse
import json
import re

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def layer_index(key):
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def rate_code(low_bits: int, high_bits: int) -> int:
    return (int(low_bits) << 4) | int(high_bits)


PAIR_KINDS = {"P22": (2, 2), "P33": (3, 3), "P24": (2, 4), "P43": (4, 3), "P44": (4, 4)}


def build_rate_tables(plan, layers, pairs, num_experts, cold_kind):
    lo, hi = PAIR_KINDS[cold_kind]
    cold_code = rate_code(lo, hi)
    hot_code = rate_code(4, 4)
    tables = {}
    n_cold_total = 0
    for layer in layers:
        entry = plan["layers"].get(layer)
        if entry is None:
            entry = next((v for k, v in plan["layers"].items()
                          if layer_index(k) == layer), None)
        if entry is None:
            raise SystemExit(f"plan has no entry for layer {layer}")
        cold = set(entry["cold"])
        n_cold_total += len(cold)
        fc1 = [[hot_code if e not in cold else cold_code
                for e in range(num_experts)] for _ in range(pairs)]
        fc2 = [[hot_code if e not in cold else cold_code
                for e in range(num_experts)] for _ in range(pairs)]
        tables[layer] = (fc1, fc2)
    return tables, n_cold_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", help="plan.json from split.py")
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--intermediate", type=int, default=1024,
                    help="per-shard trellis atom axis; see ATOMS_PER_PAIR notes")
    ap.add_argument("--moe-layers", default="0-77",
                    help="inclusive range like 3-80 or comma list")
    ap.add_argument("--codebook", default="sqg_fp16",
                    help="sqg_fp16 | sqg_e4m3 | mcg")
    ap.add_argument("--cold-kind", default="P33", choices=sorted(PAIR_KINDS))
    ap.add_argument("--atom-channels", type=int, default=32)
    ap.add_argument("--emit-config", default=None)
    ap.add_argument("--write", default=None)
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    if "-" in args.moe_layers:
        a, b = args.moe_layers.split("-")
        layers = list(range(int(a), int(b) + 1))
    else:
        layers = [int(x) for x in args.moe_layers.split(",")]

    # atom_slots derive from intermediate axis / ATOM_CHANNELS(32);
    # pairs = slots / ATOMS_PER_PAIR(8). Keep symbolic here; b12x recomputes.
    atom_slots = max(8, args.intermediate // args.atom_channels)
    pairs = max(1, atom_slots // 8)

    tables, n_cold = build_rate_tables(plan, layers, pairs, args.num_experts,
                                       args.cold_kind)
    cfg = {
        "codebook": args.codebook,
        "num_experts": args.num_experts,
        "hidden_size": args.hidden,
        "intermediate_size": args.intermediate,
        "moe_layer_indices": layers,
        "bits": None,
        "rate_tables": {str(k): v for k, v in tables.items()},
        "coupled": False,
        "pre_block": None,
        "post_block": None,
        "per_expert_input_rotations": False,
        "unit_hidden_rotations": True,
        "seed": 0,
    }
    summary = {
        "pairs_per_layer": pairs,
        "cold_kind": args.cold_kind,
        "hot_kind": "P44",
        "cold_expert_slots_total": n_cold,
        "layers": len(layers),
    }
    if args.emit_config:
        json.dump({"config": cfg, "summary": summary},
                  open(args.emit_config, "w"), indent=1)
        print(f"config -> {args.emit_config}")
        print(json.dumps(summary))

    if args.write:
        import torch
        from b12x.moe._shared.btx_schema import (
            RATE_CODE_PAIR_KINDS, rate_code, ATOM_CHANNELS, ATOMS_PER_PAIR)
        from b12x.moe._shared.kernels.w4a16.btx_synth import (
            BtxSynthConfig, write_btx_checkpoint)

        assert ATOM_CHANNELS == args.atom_channels
        assert ATOMS_PER_PAIR == 8

        def to_tensors(tbl):
            import torch
            return (torch.tensor(tbl[0], dtype=torch.uint8),
                    torch.tensor(tbl[1], dtype=torch.uint8))

        rt = {int(k): to_tensors(v) for k, v in cfg["rate_tables"].items()}
        for k, (a_, b_) in rt.items():
            for t in (a_, b_):
                for code in t.unique().tolist():
                    assert int(code) in RATE_CODE_PAIR_KINDS, f"bad code {code:#x}"
        conf = BtxSynthConfig(
            codebook=cfg["codebook"], num_experts=cfg["num_experts"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            moe_layer_indices=tuple(cfg["moe_layer_indices"]),
            bits=None, rate_tables=rt, coupled=False,
            pre_block=None, post_block=None,
            per_expert_input_rotations=False,
            unit_hidden_rotations=True, seed=0)
        man = write_btx_checkpoint(args.write, conf)
        print(f"BTX checkpoint written to {args.write} "
              f"({len(man.layers)} layer files validated)")


if __name__ == "__main__":
    main()
