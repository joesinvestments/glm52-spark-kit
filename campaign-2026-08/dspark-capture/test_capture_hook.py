#!/usr/bin/env python3
"""P2 offline unit test: prove the DSpark capture hook's payload builder
(extracted via AST from the SHIPPED overlay source, no vllm imports needed)
writes cap-*.pt files in exactly the format dspark-training/dspark_finetune.py
consumes. CPU-only fake tensors. MemAvailable-gated fail-closed."""
import ast
import os
import pathlib
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

OVERLAY = (pathlib.Path(__file__).resolve().parents[2]
           / "overlays" / "dspark-ring" / "dflash_speculator.py")
RIG = (pathlib.Path(__file__).resolve().parents[2]
       / "dspark-training" / "dspark_finetune.py")


def extract_fn(name: str):
    tree = ast.parse(OVERLAY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module([node], type_ignores=[])
            ns = {"torch": torch}
            exec(compile(mod, str(OVERLAY), "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in overlay")


def rig_contract():
    """Parse what the training rig demands from cap files (source of truth)."""
    src = RIG.read_text()
    assert 'torch.load(f, map_location="cpu", weights_only=True)' in src, \
        "rig load call changed - revisit this test"
    assert 'd["aux"].shape[0] >= max(64, MIN_A + K + 2)' in src, \
        "rig min-T rule changed - revisit this test"
    assert 'aux, ids, pos = d["aux"], d["input_ids"], d["positions"]' in src, \
        "rig key names changed - revisit this test"


def main():
    rig_contract()
    build = extract_fn("dspark_build_capture_payload")

    HID, NLayers, T = 6144, 5, 128          # T=128 satisfies rig min-T rule
    aux = [torch.randn(T, HID, dtype=torch.bfloat16) for _ in range(NLayers)]
    ids = torch.randint(0, 150000, (T + 7,))   # deliberately longer than T
    pos = torch.arange(T + 7)

    payload = build(aux, ids, pos, num_target_tokens=T)

    assert set(payload.keys()) == {"aux", "input_ids", "positions"}, \
        f"keys drifted: {payload.keys()}"
    assert payload["aux"].shape == (T, HID * NLayers), payload["aux"].shape
    assert payload["aux"].dtype == torch.bfloat16
    assert payload["input_ids"].shape == (T,)
    assert payload["positions"].shape == (T,)
    assert payload["input_ids"].dtype == ids.dtype
    # raw pre-projection check: layer slices must equal inputs verbatim
    torch.testing.assert_close(payload["aux"][:, :HID].float(),
                               aux[0].float())
    torch.testing.assert_close(payload["aux"][:, -HID:].float(),
                               aux[-1].float())

    out = pathlib.Path(__file__).parent / ("cap-%d-1.pt"
                                           % int(time.time() * 1000))
    torch.save(payload, out)

    reloaded = torch.load(out, map_location="cpu", weights_only=True)
    assert reloaded["aux"].shape == payload["aux"].shape
    assert reloaded["aux"].dtype == torch.bfloat16
    assert sorted(reloaded.keys()) == ["aux", "input_ids", "positions"]

    glob_ok = list(out.parent.glob("cap-*.pt"))
    assert out.name in [g.name for g in glob_ok], "cap-* glob must match"

    out.unlink()
    print("[PASS] capture payload format matches rig contract "
          "(keys, bf16 aux [T,HID*5], weights_only-loadable, cap-*.pt glob)")


if __name__ == "__main__":
    main()
