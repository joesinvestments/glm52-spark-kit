FROM vllm-node-tf5-glm52-b12x:probe
# Legacy GLM stack challenger: ciprianveg/CosmicRaisins mods + tonyd2wild overhang patch.
COPY kernels/ /root/models/models15/glm-triton/
COPY mods/ /tmp/mods/
COPY patches/ /tmp/patches/
# decode-aware-scheduler is built but INERT: it adds serve flags that are rejected unless
# the mod is baked in, and does nothing until you pass them (DECODE_AWARE=1 DPTB=... in the
# launcher). Shipped-but-unbuilt was a repo bug caught by ThinkCode in issue #2, who measured
# ~4x per-stream decode at C=12 with DPTB=2048 against this repo's storm numbers.
RUN bash /tmp/mods/glm52-sm12x-sparse/run.sh && \
    bash /tmp/mods/glm52-b12x-sparse/run.sh && \
    bash /tmp/mods/decode-aware-scheduler/run.sh && \
    python3 /tmp/patches/fix-indexer-mtp-overhang.py && \
    rm -rf /tmp/mods /tmp/patches
