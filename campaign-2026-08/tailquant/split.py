#!/usr/bin/env python3
"""TailQuant stage 2: router histograms -> per-layer hot/cold split plan.

Input:  router_hist.json from patch 004
        {"<layer>": [hit_count_per_expert, ...], ...}
Output: plan.json
        {
          "policy": {...},
          "layers": {
            "<layer>": {
              "num_experts": E,
              "hot": [expert_ids sorted by frequency desc],
              "cold": [...],
              "coverage": 0.93,           # fraction of hits on hot set
              "top1_share": 0.11,
              "gini": 0.41
            }, ...
          },
          "projection": {...memory math...}
        }

Policy: pick the smallest hot set covering >= --target-coverage of hits,
clamped to [--min-hot, --max-hot]. Experts never routed to in the sample are
cold regardless. Ties broken by id for determinism.
"""
import argparse
import json
import sys


def gini(counts):
    s = sorted(counts)
    n = len(s)
    tot = sum(s)
    if tot == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(s))
    return (2 * cum) / (n * tot) - (n + 1) / n


def split_layer(counts, target_coverage, min_hot, max_hot):
    order = sorted(range(len(counts)), key=lambda i: (-counts[i], i))
    total = sum(counts)
    if total == 0:
        hot = list(range(min_hot))
    else:
        hot, cum = [], 0
        for i in order:
            hot.append(i)
            cum += counts[i]
            if len(hot) >= min_hot and cum / total >= target_coverage:
                break
        if len(hot) < min_hot:
            for i in order[len(hot):]:
                hot.append(i)
                if len(hot) >= min_hot:
                    break
    hot = sorted(hot[:max_hot])
    hotset = set(hot)
    cold = [i for i in range(len(counts)) if i not in hotset]
    cov = sum(counts[i] for i in hot) / max(total, 1)
    return hot, cold, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("histogram", help="router_hist.json")
    ap.add_argument("--target-coverage", type=float, default=0.90)
    ap.add_argument("--min-hot", type=int, default=32)
    ap.add_argument("--max-hot", type=int, default=96)
    ap.add_argument("--bits-hot", type=int, default=4)
    ap.add_argument("--bits-cold", type=int, default=2)
    ap.add_argument("--baseline-bits", type=float, default=4.5,
                    help="effective bits/weight of uniform baseline incl scales")
    ap.add_argument("--out", default="plan.json")
    args = ap.parse_args()

    hist = json.load(open(args.histogram))
    layers = {}
    for key, counts in sorted(hist.items()):
        hot, cold, cov = split_layer(counts, args.target_coverage,
                                     args.min_hot, args.max_hot)
        layers[key] = {
            "num_experts": len(counts),
            "hot": hot,
            "cold": cold,
            "coverage": round(cov, 4),
            "top1_share": round(max(counts) / max(sum(counts), 1), 4),
            "gini": round(gini(counts), 4),
        }

    n_layers = len(layers)
    avg_cov = sum(l["coverage"] for l in layers.values()) / max(n_layers, 1)
    # bytes scale ~linearly in bits within a format family; scales add fixed
    # overhead modeled inside --baseline-bits already
    eff_bits = (args.bits_hot * args.target_coverage +
                args.bits_cold * (1 - args.target_coverage))
    proj = {
        "avg_hot_coverage": round(avg_cov, 4),
        "assumed_bits_hot": args.bits_hot,
        "assumed_bits_cold": args.bits_cold,
        "effective_bits_per_weight": round(eff_bits, 3),
        "baseline_bits_per_weight": args.baseline_bits,
        "weight_bytes_ratio_vs_baseline": round(eff_bits / args.baseline_bits, 3),
        "note": "ratios are approximate; scales/grouping overhead not modeled",
    }
    out = {"policy": vars(args), "projection": proj, "layers": layers}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: {n_layers} layers, "
          f"avg coverage {proj['avg_hot_coverage']}, "
          f"eff bits/w {proj['effective_bits_per_weight']}")


if __name__ == "__main__":
    main()
