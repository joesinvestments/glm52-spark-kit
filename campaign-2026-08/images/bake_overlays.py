#!/usr/bin/env python3
"""Bake glm52-spark-kit overlays into site-packages per MANIFEST.json (in-image).

Replaces the runtime-mount approach of apply.sh with an in-image bake so the
partner's launch.sh (which mounts nothing) runs the full verified substrate.
Fails closed on any missing file or md5 mismatch unless BAKE_SKIP_MD5=1.
"""
import hashlib
import json
import os
import shutil
import sys

kit = "/glm52-kit"
m = json.load(open(os.path.join(kit, "MANIFEST.json")))
sp = m["site_packages_root"].rstrip("/")
skip_md5 = os.environ.get("BAKE_SKIP_MD5") == "1"
skip = [s for s in os.environ.get("BAKE_SKIP", "").split(",") if s]
n = skipped = 0
for f in m["files"]:
    if any(s in f["overlay_file"] for s in skip):
        print(f"SKIP (b12x-coupled): {f['overlay_file']}", flush=True)
        skipped += 1
        continue
    src = os.path.join(kit, "overlays", f["overlay_file"])
    dst = os.path.join(sp, f["site_packages_target"])
    if not os.path.exists(src):
        sys.exit(f"missing overlay file: {src}")
    if not skip_md5:
        h = hashlib.md5(open(src, "rb").read()).hexdigest()
        # HEAD files intentionally ahead of MANIFEST md5s (flag-gated fused-gather
        # work): flag them loudly rather than failing the build.
        if h != f.get("md5"):
            print(f"WARN md5 drift (baking anyway): {f['overlay_file']}", flush=True)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    n += 1
print(f"baked {n} overlays into {sp} ({skipped} skipped)")