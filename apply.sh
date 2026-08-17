#!/usr/bin/env bash
# Emit docker `-v` mount arguments for one overlay set, straight from MANIFEST.json.
# Usage:  apply.sh <set> [overlay_root]        set = core | indexer | adaptive | all
#   docker run ... $(apply.sh core /var/tmp/glm52-overlay) ...
# The launcher and the manifest can never disagree because this is the only source.
set -euo pipefail
SET="${1:-core}"; ROOT="${2:-$(cd "$(dirname "$0")" && pwd)/overlays}"
python3 - "$SET" "$ROOT" "$(dirname "$0")/MANIFEST.json" <<'PY'
import json, sys, os
want, root, mf = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(mf)); sp = m["site_packages_root"]
seen = set()
for f in m["files"]:
    if want != "all" and want not in f["sets"]: continue
    src = os.path.join(root, f["overlay_file"]); dst = os.path.join(sp, f["site_packages_target"])
    if not os.path.exists(src): sys.exit(f"missing overlay file: {src}")
    if dst in seen: continue
    seen.add(dst); print(f'-v "{src}:{dst}:ro"')
PY
