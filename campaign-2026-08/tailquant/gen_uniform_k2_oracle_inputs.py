#!/usr/bin/env python3
"""Generate uniform-K2 sqg_e4m3 synthetic BTX containers (coupled + uncoupled)
as ORACLE INPUTS for the planning-confirmation run. Torch-only, Mac-local,
MemAvailable-gated. Byte-layout authority = upstream write_btx_checkpoint."""
import json
import os
import pathlib
import subprocess
import sys

MIN_FREE_BYTES = int(os.environ.get("GEN_MIN_FREE_BYTES", 8 * 1024**3))


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

import shutil

import torch  # noqa: E402
import b12x as _b12x  # noqa: E402

try:
    from b12x.moe._shared.btx_schema import (
        RATE_CODE_PAIR_KINDS, ATOM_CHANNELS, ATOMS_PER_PAIR, BTX_MANIFEST_FILENAME)
    from b12x.moe._shared.kernels.w4a16.btx_synth import (
        BtxSynthConfig, write_btx_checkpoint)
except ImportError as exc:  # cuda.bindings absent off-platform
    import importlib.util
    _root = pathlib.Path(_b12x.__file__).parent / "moe" / "_shared"

    def _load(fullname, path):
        spec = importlib.util.spec_from_file_location(fullname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = mod
        spec.loader.exec_module(mod)
        return mod

    _schema = _load("b12x.moe._shared.btx_schema", _root / "btx_schema.py")
    BTX_MANIFEST_FILENAME = _schema.BTX_MANIFEST_FILENAME
    _synth = _load("b12x.moe._shared.kernels.w4a16.btx_synth",
                   _root / "kernels" / "w4a16" / "btx_synth.py")
    BtxSynthConfig = _synth.BtxSynthConfig
    write_btx_checkpoint = _synth.write_btx_checkpoint

ROOT = pathlib.Path(os.environ.get("ORACLE_OUT", "/tmp/lane2-oracle"))
ROOT.mkdir(parents=True, exist_ok=True)

LAYER = 0
E, H, I = 8, 256, 256          # toy geometry (matches proven A5 probe shape)

for coupled in (False, True):
    out = ROOT / (f"uncoupled" if not coupled else "coupled")
    if out.exists():
        shutil.rmtree(out)
    conf = BtxSynthConfig(
        codebook="sqg_e4m3", num_experts=E, hidden_size=H,
        intermediate_size=I, moe_layer_indices=(LAYER,), bits=2,
        rate_tables={}, coupled=coupled,
        pre_block=(128 if coupled else None), post_block=(128 if coupled else None),
        per_expert_input_rotations=False, unit_hidden_rotations=True, seed=0)
    man = write_btx_checkpoint(out, conf)
    size = sum(f.stat().st_size for f in out.iterdir())
    mf = json.loads((out / BTX_MANIFEST_FILENAME).read_text())
    print(f"[synth] coupled={coupled} OK structure={mf['rates']['structure']} "
          f"bits={mf['rates'].get('bits')} hadamard={mf['hadamard']} bytes={size}")

print("[done] oracle inputs ready under", ROOT)
