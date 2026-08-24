#!/usr/bin/env python3
"""Regenerate lmhead_w8v2_sidecar.safetensors from the checkpoint's own bf16 lm_head.

Needed only if $MODEL/lmhead_w8v2_sidecar.safetensors is missing from a restored
QuantTrio copy (partner's launch.sh mounts it via VLLM_LMHEAD_V2_SIDECAR with
VLLM_LMHEAD_V2_REQUIRE=1; without it the boot fails). Per issue #1: "trivial:
one tensor, re-keyed lm_head.weight -> lm_head.weight_bf16".

Run INSIDE any image that has torch+safetensors, with /models mounted:
  docker run --rm -v $MODEL:/models --entrypoint python3 <image> /sidecar_regen.py

CPU-only, ~1-2 GB RAM for one 6144 x vocab bf16 tensor.
"""
import json
import sys
from pathlib import Path

model = Path(sys.argv[1] if len(sys.argv) > 1 else "/models")
out = model / "lmhead_w8v2_sidecar.safetensors"
if out.exists():
    print(f"sidecar already present: {out}")
    sys.exit(0)

from safetensors.torch import load_file, save_file  # noqa: E402
import torch  # noqa: E402

index = json.loads((model / "model.safetensors.index.json").read_text())
shard_name = index["weight_map"]["lm_head.weight"]
print(f"loading {shard_name} ...")
tensor = load_file(model / shard_name)["lm_head.weight"]
assert tensor.dtype == torch.bfloat16, f"unexpected dtype {tensor.dtype}"
save_file({"lm_head.weight_bf16": tensor.contiguous()}, str(out),
          metadata={"format": "pt"})
print(f"wrote {out} ({tensor.numel() * 2 / 2**30:.2f} GiB)")