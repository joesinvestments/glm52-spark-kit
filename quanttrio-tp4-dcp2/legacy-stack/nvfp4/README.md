# The nvfp4 capacity-build image, exactly as production builds it

The nvfp4_ds_mla KV port is tonyd2wild's work (GLM-5.2-NVFP4-KV-4x-DGX-Spark repo, port/
directory, building on danielwoz's E2M1 kernel). We do not vendor his port here; pull it
from his repo. What this directory gives you is the EXACT recipe production uses:

1. Build the fp8 chain first: Dockerfile -> Dockerfile.v2 -> Dockerfile.v2b (parent dir).
2. Fetch tonyd2wild's port/ directory to ~/nvfp4-port/port on every node.
3. Run build-nvfp4-gx10.sh (this dir) on every node. It is his build-nvfp4.sh with our
   values baked: BASE_IMAGE=glm52-legacy:challenger-v2b, OUT_TAG=glm52-legacy:challenger-nvfp4,
   and the overlay source pointed at the kernels dir this repo ships. NOTE: the upstream
   script HARDCODES its base image and overlay path; env overrides are silently ignored.
   Patch the variables in the file, as this copy does.
4. Launch with launch_gx10.sh (parent dir): nvfp4_ds_mla, 64-aligned max-model-len, k=2
   quantized probabilistic draft, VLLM_MARLIN_USE_ATOMIC_ADD=1.

Tree-assembly warning (issues #1/#2, and baris's independent repro): the draft-quant patch
chain is SEMANTICALLY tree-specific. It can apply clean with patch -p1 on a different
assembly of the same base commit and still produce a broken drafter (NameError, 1 tok/s
collapse, or first-request EngineDeadError depending on the tree). If you are not building
from THIS chain on THIS base, expect to re-port, not re-apply. Our working matrix on this
exact chain: nvfp4 + k=2 + quantized probabilistic serves and survives a C=12 storm at
315,968 ctx; raw JSONL in window-data/.
