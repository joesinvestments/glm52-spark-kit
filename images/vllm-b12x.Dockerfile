# The stock-base serving image the launchers call `vllm-b12x:v0.27.0-pinned`:
# upstream vLLM v0.27.0 (aarch64, CUDA 13, torch 2.13.0+cu130, Triton 3.7.1) plus the b12x
# sparse-MLA kernels at the exact commit the fleet runs and the cutlass-dsl it needs.
# The kit's overlays are then MOUNTED at run time (see apply.sh), not baked.
FROM vllm/vllm-openai:v0.27.0
ARG B12X_REF=334a2d75d166becea0aa640b402d521ea0a290eb
RUN pip install --no-cache-dir --no-deps "nvidia-cutlass-dsl==4.6.0" \
 && pip install --no-cache-dir --no-deps "git+https://github.com/local-inference-lab/b12x.git@${B12X_REF}" \
 && python3 -c "import importlib.metadata as m; print('b12x', m.version('b12x'), 'cutlass-dsl', m.version('nvidia-cutlass-dsl'))"
# build: docker build -f images/vllm-b12x.Dockerfile -t vllm-b12x:v0.27.0-pinned .
