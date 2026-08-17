# GLM-5.2 fleet platform: AEON's sm_121a vLLM 0.27.1 rebuild as base, plus the b12x sparse-MLA
# kernels at the exact commit the fleet runs, plus this kit's overlays baked in (no runtime mounts).
FROM ghcr.io/aeon-7/aeon-vllm-ultimate:2026-08-16-v0.27.1
ARG B12X_REF=334a2d75d166becea0aa640b402d521ea0a290eb
RUN pip install --no-cache-dir --no-deps "git+https://github.com/local-inference-lab/b12x.git@${B12X_REF}" \
 && python3 -c "import b12x, importlib.metadata as m; print('b12x', m.version('b12x'))"
# overlays: same site-packages targets as MANIFEST.json (apply.sh renders mounts; here they are copied)
# AEON installs vLLM into site-packages (not dist-packages like the upstream image)
COPY overlays/ /usr/local/lib/python3.12/site-packages/
RUN python3 - <<'PY'
import importlib, sys
mods = ["vllm.v1.attention.backends.mla.nvfp4_ds_mla_writer",
        "vllm.v1.attention.backends.mla.b12x_mla_sparse",
        "vllm.model_executor.layers.sparse_attn_indexer",
        "vllm.distributed.parallel_state"]
for m in mods:
    importlib.import_module(m); print("import ok:", m)
PY
LABEL org.opencontainers.image.title="glm52-spark-platform" \
      org.opencontainers.image.description="AEON v0.27.1 sm_121a base + b12x@334a2d75 + GLM52-SPARK-KIT overlays"
