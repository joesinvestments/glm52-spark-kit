# Build order (self-contained as of this commit)

1. Base image: eugr/spark-vllm-docker `./build-and-copy.sh --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 -t vllm-node-tf5-glm52-b12x:probe`
2. `Dockerfile` (this dir): bakes `kernels/` + both mods + the indexer overhang patch.
3. `Dockerfile.v2` then `Dockerfile.v2b`: the draft-quant packed-mapping patch
   (`draft-quant.patch`) plus the scoping fix for the ab666069 tree. **Load-bearing:**
   enabling `quantization:"compressed-tensors"` in the speculative config WITHOUT v2+v2b
   dies on the first request (community-confirmed: KeyError in
   scheduler.update_from_output). The quantized draft also needs
   `VLLM_MARLIN_USE_ATOMIC_ADD=1` or it is a 3x regression, see the top-level README.
4. Launch: `resolve_gid_and_launch.sh` (edit fabric values in `launch_gx10.sh` for your
   cluster; the GID resolver handles the index dynamically).

`kernels/` and `mods/` are CosmicRaisins / ciprianveg work, Apache-2.0, vendored with
their attribution headers and NOTICE intact. See top-level README credits.
