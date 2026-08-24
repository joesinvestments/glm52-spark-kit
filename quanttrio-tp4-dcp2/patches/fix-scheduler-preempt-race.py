"""Fix for issue #2: nvfp4_ds_mla + quantized draft (and k=2) EngineDeadError
on first request — KeyError at scheduler update_from_output.

Root cause (diagnosed + fix validated on a vLLM v0.27.1 rebuild of this
recipe at 400K/TP4/DCP1 on 4x GB10, nvfp4_ds_mla + compressed-tensors
quantized draft serving clean):

  The output-processing loop's guard (`request is None or
  request.is_finished()`) does not cover a request that is preempted but
  NOT dropped (output_is_stale=True, drop_stale_output=False). Under async
  scheduling the scheduler-side preemption can land before the in-flight
  step's output returns; the model runner built `req_id_to_index` from its
  own current batch, which no longer contains the evicted request. The
  request is live in `self.requests` yet absent from `req_id_to_index`, so
  the bare `req_id_to_index[req_id]` KeyErrors and kills the engine.
  nvfp4 + quantized draft (and low-k) shift preemption frequency/step
  cadence, which is why those trees hit it on the first request while the
  fp8 tree appears fine — but the race is generic to async scheduling.

Fix: keyed .get() + skip-with-debug-log for that one diagnosed case. NOT a
bare try/except; only a request absent from this step's index is skipped,
and its earlier stale-share drain already kept its state consistent.

Anchor-based like fix-indexer-mtp-overhang.py; works on ab666069 (~L1542),
e232d26, and v0.27.x (~L1761) trees.
"""
import sys

p = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
s = open(p).read()

old = "            req_index = model_runner_output.req_id_to_index[req_id]\n"
new = """            # issue #2 fix: a preempted-but-not-dropped request can be
            # live in self.requests while already evicted from the model
            # runner's batch for this async step; skip its output for the
            # step instead of KeyError-ing the engine.
            req_index = model_runner_output.req_id_to_index.get(req_id)
            if req_index is None:
                logger.debug(
                    "update_from_output: %s tracked but absent from this "
                    "step's req_id_to_index; skipping (preempted/evicted "
                    "between schedule() and async output landing).", req_id)
                continue
"""

n = s.count(old)
if n == 0:
    sys.exit("FATAL: scheduler anchor not found - tree drifted, refusing to build")
if n > 1:
    sys.exit(f"FATAL: scheduler anchor ambiguous ({n} occurrences) - refusing to build")
open(p, "w").write(s.replace(old, new, 1))
print("scheduler preempt-race patch applied")
