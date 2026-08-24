import re, sys
p = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"
s = open(p).read()
old = """        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
"""
new = """        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        ) + 1  # MTP overhang: draft tokens can spill one block past max_model_len when
               # max_model_len %% (block_size*cp) == 0; unpatched this crashes the engine at
               # >=3 concurrent seqs (community fix-indexer-mtp-overhang, verified c1-c6).
"""
if old not in s:
    sys.exit("FATAL: overhang anchor not found — tree drifted, refusing to build")
open(p, "w").write(s.replace(old, new, 1))
print("overhang patch applied")
