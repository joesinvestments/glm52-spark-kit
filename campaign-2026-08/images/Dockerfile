# glm52-collab candidate images (build ON gx10-1; no registry pull).
#
# PATCHSET=b0 -> glm52-collab:b0 : stock vllm-openai:v0.27.0 + b12x@334a2d75 + cutlass-dsl
#                                   + ray[cgraph] + glm52-spark-kit HEAD overlays baked
#                                   + collab patch 001 (diagnostic log only).
# PATCHSET=b2 -> glm52-collab:b2 : B0 + collab patch 002 (VLLM_FORCE_FUSE_GEMM_COMMS).
#
# Substrate matches the verified issue-#1 repro (all kit overlays baked into
# site-packages, not mounted), plus collab patches applied after the bake.

# Production-verified base: stock vllm-openai:v0.27.0 + b12x@334a2d75 + cutlass-dsl
FROM vllm-b12x:v0.27.0-pinned
ARG B12X_REF=36bce2c15
ARG DSL_VER=4.6.2
ARG RAY_SPEC=ray[cgraph]
ARG PATCHSET=b0

RUN pip install --no-cache-dir --no-deps --no-build-isolation --force-reinstall \
      "https://github.com/local-inference-lab/b12x/archive/${B12X_REF}.tar.gz" || echo "b12x reinstall skipped; base pin already correct" \
 && pip install --no-cache-dir --no-deps "nvidia-cutlass-dsl==${DSL_VER}" \
 && pip install --no-cache-dir "${RAY_SPEC}" \
 && python3 -c "import importlib.metadata as m; print('b12x', m.version('b12x'), 'cutlass-dsl', m.version('nvidia-cutlass-dsl'), 'ray', m.version('ray'))"

# Bake the glm52-spark-kit overlay set into site-packages per MANIFEST.json,
# exactly what apply.sh mounts at runtime on the production stack.
COPY kit/ /glm52-kit/
ENV BAKE_SKIP=""
RUN if [ "$PATCHSET" = "b3" ]; then export BAKE_SKIP="b12x/"; fi  && python3 /glm52-kit/bake_overlays.py && rm -rf /glm52-kit

# Collab patches apply AFTER overlays (001 targets stock vllm file; 002 ditto).
COPY patches/ /glm52-collab-patches/
RUN V=/usr/local/lib/python3.12/dist-packages \
 && cd "$V" \
 && for P in 001-index-share-engagement-log 004-tailquant-router-profiler; do \
      case "$PATCHSET" in b4) ;; *) [ "$P" = "004-tailquant-router-profiler" ] && continue ;; esac; \
      echo "== collab patch $P" \
      && patch -p1 --dry-run < /glm52-collab-patches/$P.patch \
      && patch -p1 < /glm52-collab-patches/$P.patch || exit 1; \
    done \
 && if [ "$PATCHSET" = "b2" ]; then \
      echo "== collab patch 002 (b2 only)" \
      && patch -p1 --dry-run < /glm52-collab-patches/002-force-fuse-gemm-comms-env.patch \
      && patch -p1 < /glm52-collab-patches/002-force-fuse-gemm-comms-env.patch; \
    fi \
 && rm -rf /glm52-collab-patches \
 && python3 -c "import vllm.config.compilation; import b12x; print('imports OK')"
