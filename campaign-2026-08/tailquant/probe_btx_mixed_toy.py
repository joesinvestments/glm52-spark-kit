#!/usr/bin/env python3
"""B2 de-risk, downsized per incident policy: toy-geometry mixed P44/P33 BTX
synth -> load -> prepare, CPU-only, MemAvailable-gated fail-closed.

Toy geometry: E=8 experts, H=256, I=256 (atom_slots=8 -> pairs=1). Container is
kilobytes. Same code paths as full GLM geometry (write_btx_checkpoint,
BtxManifest validation, prepare_btx_moe_weights pair-extent machinery).

Runs where builds run (laptop/venv), NEVER on a production node.
Verdict semantics: [prepare] OK => mixed containers load+prepare;
[prepare] FAILED with exact error => capture that error verbatim, it IS the
planning answer (expected family per docs/btx-checkpoint-format.md: P44 sets
are declared-unfused).
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

# Hard default 8 GiB. Override ONLY off-fleet (laptops/workstations), e.g.
# BTX_TOY_MIN_FREE_BYTES=2147483648; production nodes must run the default.
MIN_FREE_BYTES = int(os.environ.get("BTX_TOY_MIN_FREE_BYTES", 8 * 1024**3))


def mem_available_bytes():
    p = pathlib.Path("/proc/meminfo")
    if p.exists():  # linux
        for line in p.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
        return None
    try:  # macOS
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=5).stdout
        pages = {}
        ps = 4096
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
        free = pages.get("Pages free", 0) + pages.get("Pages speculative", 0)
        inactive = pages.get("Pages inactive", 0)
        return (free + inactive) * ps  # conservative: ignore purgeable
    except Exception:
        return None


avail = mem_available_bytes()
if avail is None:
    sys.exit("FAIL-CLOSED: cannot determine available memory")
if avail < MIN_FREE_BYTES:
    sys.exit(f"FAIL-CLOSED: {avail / 2**30:.1f} GiB available < 8 GiB floor")
print(f"[gate] mem available {avail / 2**30:.1f} GiB - proceeding")

import torch  # noqa: E402

try:
    from b12x.moe._shared.btx_schema import (
        RATE_CODE_PAIR_KINDS, ATOM_CHANNELS, ATOMS_PER_PAIR)
    from b12x.moe._shared.kernels.w4a16.btx_synth import (
        BtxSynthConfig, write_btx_checkpoint)
except ImportError as exc:  # e.g. cuda.bindings absent off-platform
    print(f"[import] package path unavailable ({exc}); "
          "loading pure-torch modules directly")
    import importlib.util
    import b12x as _b12x
    _root = pathlib.Path(_b12x.__file__).parent / "moe" / "_shared"

    def _load(fullname, path):
        spec = importlib.util.spec_from_file_location(fullname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = mod
        spec.loader.exec_module(mod)
        return mod

    _schema = _load("b12x.moe._shared.btx_schema",
                    _root / "btx_schema.py")
    RATE_CODE_PAIR_KINDS = _schema.RATE_CODE_PAIR_KINDS
    ATOM_CHANNELS = _schema.ATOM_CHANNELS
    ATOMS_PER_PAIR = _schema.ATOMS_PER_PAIR
    BtxManifest = _schema.BtxManifest
    _synth = _load("b12x.moe._shared.kernels.w4a16.btx_synth",
                   _root / "kernels" / "w4a16" / "btx_synth.py")
    BtxSynthConfig = _synth.BtxSynthConfig
    write_btx_checkpoint = _synth.write_btx_checkpoint

assert ATOM_CHANNELS == 32 and ATOMS_PER_PAIR == 8

LAYER = 0
NUM_EXPERTS = 8          # toy: 2 hot @P44, 6 cold @P33
HIDDEN = 256             # multiple of 32
INTERMEDIATE = 256       # atom_slots=8 -> pairs=1
PAIRS = 1
COLD = list(range(2, NUM_EXPERTS))

hot_code, cold_code = 0x44, 0x33
fc1 = [[hot_code if e not in COLD else cold_code
        for e in range(NUM_EXPERTS)] for _ in range(PAIRS)]
fc2 = [row[:] for row in fc1]
for tbl in (fc1, fc2):
    for code in {c for row in tbl for c in row}:
        assert code in RATE_CODE_PAIR_KINDS, f"bad rate code {code:#x}"

workdir = pathlib.Path(tempfile.mkdtemp(prefix="btx-toy-"))
conf = BtxSynthConfig(
    codebook="sqg_e4m3", num_experts=NUM_EXPERTS, hidden_size=HIDDEN,
    intermediate_size=INTERMEDIATE, moe_layer_indices=(LAYER,), bits=None,
    rate_tables={LAYER: (torch.tensor(fc1, dtype=torch.uint8),
                         torch.tensor(fc2, dtype=torch.uint8))},
    coupled=False, pre_block=None, post_block=None,
    per_expert_input_rotations=False, unit_hidden_rotations=True, seed=0)

man = write_btx_checkpoint(workdir, conf)
size = sum(f.stat().st_size for f in workdir.iterdir())
print(f"[synth] OK kinds={sorted(man.rates.pair_kinds)} "
      f"structure={man.rates.structure} bytes={size}")
assert size < 50 * 1024 * 1024, "toy container must stay under ~50MB"

# --- load side ---
manifest_json = json.loads((workdir / "btx-manifest.json").read_text())
try:
    from b12x.moe._shared.btx_schema import BtxManifest  # noqa: E402
except ImportError:
    pass  # already bound by direct-load fallback above
manifest = BtxManifest.from_dict(manifest_json)
manifest.validate_extent(0, manifest.geometry.atom_slots)
print("[load] manifest parsed + extent validated")

try:
    from b12x.moe._shared.kernels.w4a16.btx import (
        prepare_btx_moe_weights, read_btx_layer)
except ImportError as exc:
    print(f"[import] serving loader unavailable off-platform: {exc}")
    print("[verdict] synth+manifest stages PASS; prepare stage requires the "
          "serving image (run this same script inside glm52-collab:b3)")
    raise SystemExit(0)

layer_obj = read_btx_layer(str(workdir), manifest, LAYER,
                           first_slot=0,
                           slot_count=manifest.geometry.atom_slots)
print("[load] layer object ok")
try:
    prepared = prepare_btx_moe_weights(
        layer_obj, activation="silu", device="cpu",
        params_dtype=torch.float16)
    print("[prepare] OK -> mixed P44/P33 LOADS AND PREPARES (cpu)")
except Exception as exc:
    print(f"[prepare] FAILED: {type(exc).__name__}: {exc}")
    print("[verdict] exact error above is the planning answer - report it,"
          " hold mixed, uniform-rate becomes fallback thesis")
    raise SystemExit(3)

print("[done] all stages green")
