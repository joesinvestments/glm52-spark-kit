import sys
p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py"
s = open(p).read()
old = "and self.impl.masked_mha_available  # type: ignore[attr-defined]"
new = "and getattr(self.impl, \"masked_mha_available\", False)  # SM120 impl lacks the attr; False routes to the dense-mask prefill path it supports (sm121 bring-up fix)"
if old not in s: sys.exit("FATAL: anchor not found")
s = s.replace(old, new, 1)
open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True)
print("masked_mha getattr patch applied + compiles")
