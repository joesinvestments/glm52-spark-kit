#!/usr/bin/env python3
"""Lane 1a: prove the FULL D6 capture->rig loop OFFLINE.

1. Generate a synthetic spec config (tiny dims) + synthetic cap-*.pt set
   written via the SAME payload builder the production hook uses (P2 patch).
2. Run the actual rig (dspark_finetune.py) unmodified except env:
   DEV=cpu WORLD_SIZE=1 STEPS=1 - it must ingest the caps and take one real
   optimizer step, saving a checkpoint.
MemAvailable gate fail-closed first."""
import ast
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

MIN_FREE_BYTES = int(os.environ.get("TEST_MIN_FREE_BYTES", 8 * 1024**3))


def mem_available_bytes():
    p = pathlib.Path("/proc/meminfo")
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
        return None
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=5).stdout
        pages, ps = {}, 4096
        for line in out.splitlines():
            k, _, v = line.partition(":")
            k = k.strip()
            if "page size of" in k:
                ps = int(k.split("(")[1].split()[0])
                continue
            try:
                pages[k] = int(v.strip().rstrip("."))
            except ValueError:
                pages[k] = 0
        return (pages.get("Pages free", 0) + pages.get("Pages speculative", 0)
                + pages.get("Pages inactive", 0)) * ps
    except Exception:
        return None


avail = mem_available_bytes()
if avail is None:
    sys.exit("FAIL-CLOSED: cannot determine available memory")
if avail < MIN_FREE_BYTES:
    sys.exit(f"FAIL-CLOSED: {avail / 2**30:.2f} GiB < floor")
print(f"[gate] mem available {avail / 2**30:.2f} GiB")

import torch  # noqa: E402

WORK = pathlib.Path(os.environ.get("LANE1A_WORK", "/tmp/lane1a"))
if WORK.exists():
    shutil.rmtree(WORK)
SPEC = WORK / "spec"
CAPS = WORK / "caps"
OUTD = WORK / "out"
for d in (SPEC, CAPS, OUTD):
    d.mkdir(parents=True)

# ---- tiny spec config matching the rig's cfg schema ----
HID, NL, AUX_LAYERS = 64, 2, 5
cfg = {
    "transformer_layer_config": {
        "hidden_size": HID,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "intermediate_size": 128,
        "num_hidden_layers": NL,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0},
        "vocab_size": 1024,
    },
    "aux_hidden_state_layer_ids": list(range(AUX_LAYERS)),
    "markov_rank": 32,
    "mask_token_id": 999,
}
json.dump(cfg, open(SPEC / "config.json", "w"))

# ---- synthetic caps via the SHIPPED hook builder (AST-extracted, P2 style) --
overlay = (pathlib.Path(__file__).resolve().parents[2]
           / "overlays" / "dspark-ring" / "dflash_speculator.py")
tree = ast.parse(overlay.read_text())
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "dspark_build_capture_payload":
        ns = {"torch": torch}
        exec(compile(ast.Module([node], type_ignores=[]), str(overlay), "exec"), ns)
        build = ns["dspark_build_capture_payload"]
        break
else:
    sys.exit("builder not found in overlay")

T = 160  # >= rig min-T floor (max(64, MIN_A+K+2))
torch.manual_seed(11)
for i in range(8):
    aux = [torch.randn(T, HID, dtype=torch.bfloat16) for _ in range(AUX_LAYERS)]
    ids = torch.randint(0, 1000, (T,))
    pos = torch.arange(T)
    torch.save(build(aux, ids, pos, T), CAPS / f"cap-{int(time.time()*1000)}-{i}.pt")
print(f"[synth] 8 cap files written (aux [T,{HID*AUX_LAYERS}] bf16)")

# ---- synthetic init checkpoint in the rig's OWN layout ----
rig_src = (pathlib.Path(__file__).resolve().parents[2]
           / "dspark-training" / "dspark_finetune.py")
# Exec the rig source minus its main() call (env already set), use Draft +
# save_model so keys/shapes are exact by construction, then re-save to SPEC.
rig_src_full = rig_src.read_text()
rig_src_full = rig_src_full.replace('dist.init_process_group("nccl")',
                                    'dist.init_process_group("gloo")')
assert rig_src_full.rstrip().endswith("main()"), "unexpected tail"
os.environ.update(SPEC_DIR=str(SPEC), OUT_DIR=str(OUTD), DATA_DIR=str(CAPS), CAPD=str(CAPS),
                  DEV="cpu", WORLD_SIZE="1", RANK="0", WINDOW="64", K="3",
                  STEPS="1", ACCUM="1", SAVE_EVERY="1", LR="1e-4")
ns = {"__name__": "rig_not_main"}
exec(compile(rig_src_full.replace("if __name__ == \"__main__\":\n    main()", ""),
             str(rig_src), "exec"), ns)
m0 = ns["Draft"]()
ns["save_model"](m0, str(SPEC / "model.safetensors"))
del m0
print(f"[synth] init checkpoint written to {SPEC/'model.safetensors'}")

# ---- run the REAL rig, one step, cpu ----
env = dict(os.environ,
           SPEC_DIR=str(SPEC), OUT_DIR=str(OUTD), CAPD=str(CAPS),
           DEV="cpu", WORLD_SIZE="1", RANK="0",
           WINDOW="64", K="3", STEPS="1", ACCUM="1", SAVE_EVERY="1",
           LR="1e-4", TEST_MIN_FREE_BYTES=str(MIN_FREE_BYTES))
src = rig_src.read_text()
src = src.replace('dist.init_process_group("nccl")',
                  'dist.init_process_group("gloo")')  # cpu offline shim, recorded
shim = WORK / "rig_shim.py"
shim.write_text(src)
t0 = time.time()
proc = subprocess.run([sys.executable, str(shim)], env=env,
                      capture_output=True, text=True, timeout=900)
dt = time.time() - t0
log_tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-12:])
print(f"[rig] rc={proc.returncode} in {dt:.0f}s; tail:\n{log_tail}")
ckpts = list(OUTD.glob("*"))
print("[out] artifacts:", [c.name for c in ckpt] if (ckpt := ckpts) else "NONE")
assert proc.returncode == 0, "rig failed - see tail above"
assert any("safetensors" in c.name or c.suffix == ".safetensors" for c in ckpts) \
    or any(c.stat().st_size > 0 for c in ckpts), "no checkpoint artifact saved"
print("[PASS] capture->loader->one-optimizer-step->checkpoint loop proven offline")
