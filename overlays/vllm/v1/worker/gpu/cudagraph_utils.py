# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any, NamedTuple, Protocol

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.distributed.parallel_state import (
    get_pp_group,
    graph_capture,
    is_global_first_rank,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig
# R9/R17: single canonical parser for VLLM_ADAPTIVE_SPEC_DEPTHS. Both this
# module and vllm/v1/core/sched/scheduler.py import from here so the set of
# depths the scheduler can select and the set of depths this manager captures
# FULL graphs for are the same object by construction (see depth_ladder.py's
# module docstring for why a duplicated parse is unsafe).
from vllm.v1.spec_decode.dynamic.depth_ladder import (
    parse_adaptive_spec_depth_ladder,
)
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class AttentionState(NamedTuple):
    attn_metadata: dict[str, Any] | None
    slot_mappings: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """Describes the shape of the batch and CG mode to run; this is used to make shape
    matches between the capture and runtime."""

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)
    uniform_token_count: int | None = None
    num_active_loras: int = 0


class CreateForwardFn(Protocol):
    """Factory that prepares inputs (OUTSIDE the graph) and returns a
    forward_fn. Called with warmup=True for the warmup pass and warmup=False
    for the captured pass."""

    def __call__(
        self,
        desc: BatchExecutionDescriptor,
        warmup: bool,
    ) -> Callable[[CUDAGraphMode], None]: ...


def _is_compatible(
    desc: BatchExecutionDescriptor,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    num_active_loras: int,
) -> bool:
    # desc.uniform_token_count=None (PIECEWISE) can handle any uniform_token_count
    # desc.num_reqs=None means no request padding needed (PIECEWISE)
    return (
        (
            desc.uniform_token_count is None
            or desc.uniform_token_count == uniform_token_count
        )
        and (desc.num_reqs is None or desc.num_reqs >= num_reqs)
        and desc.num_tokens >= num_tokens
        and desc.num_active_loras == num_active_loras
    )


def get_uniform_token_count(
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
) -> int | None:
    """
    Return the uniform token count if batch is uniform, else None.
    A batch is uniform if all requests have the same number of tokens.
    """
    if (max_query_len == num_tokens // num_reqs) and (
        num_tokens == max_query_len * num_reqs
    ):
        return max_query_len
    return None


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = cudagraph_mode
        self.decode_query_len = decode_query_len

        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank
        self.lora_capture_cases = lora_capture_cases or [0]
        # Precompute actual num_active_loras -> captured case mapping so that
        # dispatch() is a plain dict lookup instead of a per-call bisect.
        self._lora_dispatch_map, self._max_lora_case = self._build_lora_dispatch_map()

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.pool = current_platform.get_global_graph_pool() if cudagraph_mode else None

        self._graphs_captured = False

        self._candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]] = {}
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}

        # Breakable CUDA graph (PW CUDA graph without torch.compile)
        self.use_breakable_cg = (
            is_breakable_cudagraph_enabled()
            and self.cudagraph_mode.has_piecewise_cudagraphs()
        )
        self.breakable_cg_runner: BreakableCUDAGraphWrapper | None = None

        self._init_candidates()

    def _build_lora_dispatch_map(self) -> tuple[dict[int, int], int]:
        """Precompute actual num_active_loras -> effective captured case.

        Mirrors the num_tokens candidate expansion in ``_init_candidates``:
        every possible active-LoRA count is mapped ahead of time to the
        smallest captured case that can serve it, so ``dispatch`` is a plain
        dict lookup instead of a per-call bisect.
        """
        captured_with_lora = sorted(c for c in self.lora_capture_cases if c > 0)
        if not captured_with_lora:
            return {}, 0
        dispatch_map: dict[int, int] = {}
        case_idx = 0
        for n in range(1, captured_with_lora[-1] + 1):
            while captured_with_lora[case_idx] < n:
                case_idx += 1
            dispatch_map[n] = captured_with_lora[case_idx]
        return dispatch_map, captured_with_lora[-1]

    def _resolve_effective_loras(self, num_active_loras: int) -> int:
        """Map an actual active-LoRA count to its captured graph case."""
        if num_active_loras <= 0 or not self._lora_dispatch_map:
            return num_active_loras
        # Counts above the largest captured case clamp to it.
        return self._lora_dispatch_map.get(num_active_loras, self._max_lora_case)

    def _init_candidates(self) -> None:
        """Build priority-ordered candidate lists for each token count."""
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not (self.cudagraph_mode and capture_sizes):
            # R17 (re-anchored from R9): fail closed BEFORE this early return,
            # not after it. `cudagraph_mode` is falsy for CUDAGraphMode.NONE
            # (value 0), which is what `--enforce-eager` and every
            # backend-driven downgrade to eager resolve to, and
            # `cudagraph_capture_sizes` is empty when the compilation config
            # captures nothing at all. In both cases every reachable adaptive
            # depth would execute eagerly. Returning here used to skip
            # `_assert_adaptive_spec_graph_coverage()` entirely, so the
            # FULL-mode rejection below was unreachable for exactly the mode
            # it most needed to reject.
            self._assert_adaptive_spec_graphs_possible(capture_sizes)
            return

        capture_sizes = sorted(capture_sizes)
        max_decode_tokens = self.max_num_reqs * self.decode_query_len
        decode_mode = self.cudagraph_mode.decode_mode()
        mixed_mode = self.cudagraph_mode.mixed_mode()
        separate_decode_routine = self.cudagraph_mode.separate_routine()
        max_cg_capture_size = self.compilation_config.max_cudagraph_capture_size

        descs_by_token_lora: dict[tuple[int, int], list[BatchExecutionDescriptor]] = (
            defaultdict(list)
        )
        descs_by_mode: defaultdict[CUDAGraphMode, list[BatchExecutionDescriptor]] = (
            defaultdict(list)
        )

        # When using the static per-batch-size Dynamic SD schedule,
        # num_speculative_tokens is the max number of draft tokens. The
        # scheduler might use a smaller number so we need to capture graphs
        # for all possible values during decode.
        #
        # R17 note: this call site used to read
        # `uses_dynamic_speculative_decoding()`, which R17's config/speculative.py
        # pass (see PORTS_SCHED_NOTES.md) broadened into
        # `uses_batch_size_dynamic_speculative_decoding() or
        # uses_acceptance_length_adaptation()`. This block only consumes
        # `num_speculative_tokens_per_batch_size`, which is exclusively the
        # static-schedule field, so it is narrowed here to
        # `uses_batch_size_dynamic_speculative_decoding()` -- the acceptance-length
        # (adaptive-ladder) layer is captured separately below by
        # `_init_adaptive_spec_candidates`, which is the R9 capability this
        # port re-anchors.
        speculative_config = self.vllm_config.speculative_config
        if (
            speculative_config
            and speculative_config.uses_batch_size_dynamic_speculative_decoding()
        ):
            num_spec_per_batch_size = (
                speculative_config.num_speculative_tokens_per_batch_size
            )
            # uses_batch_size_dynamic_speculative_decoding() guarantees this is set.
            assert num_spec_per_batch_size is not None
            # decode_query_len = num_speculative_steps + num_new_sampled_tokens
            # _per_step. Recover num_new_sampled_tokens_per_step
            # from the values the manager already has.
            num_new_sampled_tokens_per_step = (
                self.decode_query_len - self.vllm_config.num_speculative_tokens
            )
            # Each entry is (range_start, range_end, num_speculative_tokens).
            decode_query_lens = [
                x[2] + num_new_sampled_tokens_per_step for x in num_spec_per_batch_size
            ]
        else:
            decode_query_lens = [self.decode_query_len]

        for num_tokens, num_active_loras in product(
            capture_sizes, self.lora_capture_cases
        ):
            # Capture uniform decode specfifc graphs if required
            #  (i.e. separate decode routine)
            if separate_decode_routine and decode_mode:
                for decode_query_len in decode_query_lens:
                    rounded_num_tokens = round_up(num_tokens, decode_query_len)
                    rounded_num_reqs = rounded_num_tokens // decode_query_len

                    if (
                        rounded_num_tokens > max_decode_tokens
                        or rounded_num_tokens > max_cg_capture_size
                        or rounded_num_reqs > self.max_num_reqs
                    ):
                        continue

                    desc = BatchExecutionDescriptor(
                        cg_mode=decode_mode,
                        num_tokens=rounded_num_tokens,
                        num_reqs=rounded_num_reqs,
                        uniform_token_count=decode_query_len,
                        num_active_loras=num_active_loras,
                    )

                    # avoid duplicate graphs
                    if desc not in descs_by_mode[decode_mode]:
                        descs_by_mode[decode_mode].append(desc)
                        descs_by_token_lora[
                            (rounded_num_tokens, num_active_loras)
                        ].append(desc)

            if mixed_mode:
                # for PIECEWISE graphs there is no limit on requests when replaying
                # i.e. no request padding is needed, so we leave it as None.
                # For breakable PW graphs, break-point kernels read the real batch
                # from the forward context; in-graph kernels handle the token padding
                # themselves from the padded slot_mapping (rows with slot == -1).
                num_reqs = None
                if mixed_mode == CUDAGraphMode.FULL:
                    num_reqs = min(num_tokens, self.max_num_reqs)
                desc = BatchExecutionDescriptor(
                    cg_mode=mixed_mode,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_active_loras=num_active_loras,
                )
                descs_by_mode[mixed_mode].append(desc)
                descs_by_token_lora[(num_tokens, num_active_loras)].append(desc)

        # R17 (re-anchored from R9): FULL decode descriptors for every other
        # rung of the adaptive (acceptance-length) speculative-depth ladder.
        # Additive -- nothing above is changed. Unlike the static schedule
        # block above (which rounds a fixed set of capture_sizes up to each
        # decode_query_len), this enumerates every (depth, num_reqs) shape the
        # scheduler can actually select, exactly, which is what makes the
        # coverage assertion below checkable.
        adaptive_descs_by_token_lora = self._init_adaptive_spec_candidates(
            max_decode_tokens,
            decode_mode,
            separate_decode_routine,
            descs_by_mode,
        )

        if not descs_by_token_lora and not adaptive_descs_by_token_lora:
            # The same failure one step later: a resolved mode that produced no
            # descriptor for any capture size cannot serve any adaptive depth.
            self._assert_adaptive_spec_graphs_possible(capture_sizes)
            return

        self._candidates = self._build_candidate_ranges(descs_by_token_lora)

        # Adaptive descriptors are PREPENDED to the base candidate list for
        # their own (tighter) token-count range, never substituted for it.
        #
        # This is a deliberate reconciliation of the pinned community patch,
        # which merges both descriptor sets into one `descs_by_token_lora` and
        # so re-partitions the token-count ranges. Under the FULL_AND_PIECEWISE
        # default that silently REMOVES mixed-batch coverage: introducing a
        # bucket at, say, 5 tokens makes a 4-token prefill resolve to a bucket
        # that holds only uniform-5 decode descriptors, which no mixed batch is
        # compatible with, so it drops to eager instead of the PIECEWISE graph
        # at 6 it used to get. Layering keeps every base lookup reachable:
        # `dispatch` walks the list in order and the base entries are still
        # there, just after the exact-depth ones.
        for key, adaptive_descs in self._build_candidate_ranges(
            adaptive_descs_by_token_lora
        ).items():
            base_descs = self._candidates.get(key, ())
            merged = list(adaptive_descs)
            merged.extend(desc for desc in base_descs if desc not in merged)
            self._candidates[key] = merged

        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs[mode] = descs

        self._assert_adaptive_spec_graph_coverage(separate_decode_routine)

    def _build_candidate_ranges(
        self,
        descs_by_token_lora: dict[tuple[int, int], list[BatchExecutionDescriptor]],
    ) -> dict[tuple[int, int], list[BatchExecutionDescriptor]]:
        """Map every token count to the descriptors of the smallest bucket
        that can still hold it. Extracted from the (formerly inlined) loop in
        `_init_candidates` so both the base and adaptive descriptor sets can
        be range-mapped independently before being merged."""
        candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]] = {}
        all_token_counts = sorted({k[0] for k in descs_by_token_lora})
        current_range_start = 0
        for token_cg_size in all_token_counts:
            for i in range(current_range_start, token_cg_size + 1):
                for num_active_loras in self.lora_capture_cases:
                    staging_key = (token_cg_size, num_active_loras)
                    if staging_key in descs_by_token_lora:
                        candidates[(i, num_active_loras)] = descs_by_token_lora[
                            staging_key
                        ]
            current_range_start = token_cg_size + 1
        return candidates

    def adaptive_spec_decode_query_lens(self) -> list[int]:
        """Uniform verification query lengths reachable under adaptive MTP.

        One entry per rung of the `VLLM_ADAPTIVE_SPEC_DEPTHS` ladder (parsed
        by the single canonical `parse_adaptive_spec_depth_ladder`, shared
        with the scheduler -- see the import comment above), each
        `depth + num_new_sampled_tokens_per_step` wide -- the same arithmetic
        `GPUModelRunner` uses to derive `decode_query_len` from the fixed K.

        Empty (and therefore a no-op) unless acceptance-length adaptation is
        configured (`speculative_config.uses_acceptance_length_adaptation()`)
        AND this manager owns the verification shape. The draft decode
        manager is built with `decode_query_len == 1`: it drafts one token
        per step at every depth, so it has nothing to follow.

        Once adaptation IS configured and this manager DOES own the shape, an
        empty or unrecognised result is a failure, not a no-op: it would let
        every caller conclude "nothing to cover" and serve the ladder with no
        captured graph behind it. Those cases raise.
        """
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is None or self.decode_query_len <= 1:
            return []
        if not speculative_config.uses_acceptance_length_adaptation():
            return []

        # Past this point adaptation is configured and this manager owns the
        # verification shape. Nothing below may answer "no depths to cover".
        max_num_spec_tokens = self.vllm_config.num_speculative_tokens
        if max_num_spec_tokens <= 0:
            raise RuntimeError(
                "Adaptive speculative depth is configured "
                "(adaptive_speculative_tokens_window="
                f"{speculative_config.adaptive_speculative_tokens_window}) but "
                f"num_speculative_tokens resolved to {max_num_spec_tokens}. "
                "There is no ladder to capture graphs for and no way to tell "
                "which verification shapes the scheduler can produce."
            )
        num_new_sampled_tokens_per_step = self.decode_query_len - max_num_spec_tokens
        if num_new_sampled_tokens_per_step < 1:
            # Not the `K + new sampled tokens` verification shape this port was
            # derived against. Capturing a shape whose meaning is unproven is
            # wrong, and so is capturing nothing: the scheduler would still
            # select ladder depths and run them eagerly.
            raise RuntimeError(
                "Adaptive speculative depth is configured but decode_query_len "
                f"({self.decode_query_len}) is not greater than "
                f"num_speculative_tokens ({max_num_spec_tokens}), so the "
                "per-depth verification shape cannot be derived. This is not "
                "the 'K + newly sampled tokens' verifier layout this capture "
                "path is derived against; refusing to run adaptive depth "
                "against an unrecognised verifier shape."
            )

        decode_query_lens = [
            depth + num_new_sampled_tokens_per_step
            for depth in parse_adaptive_spec_depth_ladder(max_num_spec_tokens)
        ]
        if not decode_query_lens:
            raise RuntimeError(
                "Adaptive speculative depth is configured but the depth ladder "
                f"parsed to an empty set for num_speculative_tokens="
                f"{max_num_spec_tokens}. Refusing to report 'no depths to "
                "cover' for a run that will still adapt its depth."
            )
        return decode_query_lens

    def _init_adaptive_spec_candidates(
        self,
        max_decode_tokens: int,
        decode_mode: CUDAGraphMode,
        separate_decode_routine: bool,
        descs_by_mode: dict[CUDAGraphMode, list[BatchExecutionDescriptor]],
    ) -> dict[tuple[int, int], list[BatchExecutionDescriptor]]:
        """Add a FULL decode descriptor per (adaptive depth, request count).

        The request count is enumerated exactly rather than rounded up to a
        capture size (unlike the static-schedule block above), which is what
        makes the coverage claim in `_assert_adaptive_spec_graph_coverage`
        checkable: for `max_num_seqs = R` and ladder `D`, the captured set is
        exactly `{(d + s) * n for d in D for n in 1..min(R, 32)}`, every
        uniform verification shape the scheduler can produce.
        """
        adaptive_descs_by_token_lora: dict[
            tuple[int, int], list[BatchExecutionDescriptor]
        ] = defaultdict(list)

        decode_query_lens = self.adaptive_spec_decode_query_lens()
        if not (decode_query_lens and separate_decode_routine and decode_mode):
            return adaptive_descs_by_token_lora

        max_cg_capture_size = self.compilation_config.max_cudagraph_capture_size
        # Exact request counts, mirroring `small_decode_sizes` in
        # `CompilationConfig.adjust_cudagraph_sizes_for_spec_decode`.
        max_captured_reqs = min(self.max_num_reqs, 32)
        if self.max_num_reqs > max_captured_reqs:
            logger.warning(
                "Adaptive speculative depth: capturing per-depth decode "
                "graphs for request counts 1..%d only; batches with more than "
                "%d requests will fall back to the fixed-K descriptors.",
                max_captured_reqs,
                max_captured_reqs,
            )

        existing = {
            (d.num_tokens, d.num_reqs, d.uniform_token_count, d.num_active_loras)
            for d in descs_by_mode[decode_mode]
        }
        skipped: list[tuple[int, int]] = []
        for decode_query_len, num_reqs, num_active_loras in product(
            decode_query_lens,
            range(1, max_captured_reqs + 1),
            self.lora_capture_cases,
        ):
            num_tokens = decode_query_len * num_reqs
            if num_tokens > max_decode_tokens or (
                max_cg_capture_size is not None and num_tokens > max_cg_capture_size
            ):
                skipped.append((decode_query_len, num_reqs))
                continue

            key = (num_tokens, num_reqs, decode_query_len, num_active_loras)
            if key in existing:
                # Already captured by the static-schedule / fixed-K path above.
                # Capturing it twice would trip the `desc not in self.graphs`
                # assertion in `capture()`.
                continue
            existing.add(key)

            desc = BatchExecutionDescriptor(
                cg_mode=decode_mode,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                uniform_token_count=decode_query_len,
                num_active_loras=num_active_loras,
            )
            descs_by_mode[decode_mode].append(desc)
            adaptive_descs_by_token_lora[(num_tokens, num_active_loras)].append(desc)

        if skipped:
            logger.warning(
                "Adaptive speculative depth: %d (query_len, num_reqs) decode "
                "shapes exceed max_cudagraph_capture_size=%s / "
                "max_decode_tokens=%d and were NOT captured: %s.",
                len(skipped),
                max_cg_capture_size,
                max_decode_tokens,
                sorted(set(skipped)),
            )
        else:
            logger.info(
                "Adaptive speculative depth: captured decode query lengths %s "
                "for request counts 1..%d (fixed decode_query_len=%d).",
                decode_query_lens,
                max_captured_reqs,
                self.decode_query_len,
            )
        return adaptive_descs_by_token_lora

    def _assert_adaptive_spec_graphs_possible(
        self, capture_sizes: list[int] | None
    ) -> None:
        """Fail closed on the paths that never reach descriptor construction.

        `_init_candidates` has two early returns that predate this port: one
        for a falsy/NONE `cudagraph_mode` or an empty capture-size list, one
        for a descriptor set that came out empty anyway. Neither can produce
        a captured graph for any depth, so if this manager owns an adaptive
        verification shape the run must abort here rather than proceed to
        execute the ladder eagerly.
        """
        decode_query_lens = self.adaptive_spec_decode_query_lens()
        if not decode_query_lens:
            return
        raise RuntimeError(
            "Adaptive speculative depth requires FULL CUDA graphs, but this "
            "run captures no CUDA graphs at all: cudagraph_mode="
            f"{getattr(self.cudagraph_mode, 'name', self.cudagraph_mode)}, "
            f"cudagraph_capture_sizes={list(capture_sizes or [])}. Every depth "
            f"on the ladder ({decode_query_lens} verification query lengths) "
            "would run eagerly. Remove --enforce-eager, restore a non-empty "
            "cudagraph_capture_sizes, or disable adaptive speculative depth."
        )

    def _assert_adaptive_spec_graph_coverage(
        self, separate_decode_routine: bool
    ) -> None:
        """Fail closed when adaptive depth would run without FULL graphs.

        This port's whole premise (re-anchored from R9) is that the adaptive
        ladder and the captured descriptor set are the same set. An
        environment variable saying so is not evidence -- `cudagraph_mode` is
        resolved from the attention backend's capabilities at load time and
        can land on PIECEWISE or NONE for reasons no launch script can see.
        This check runs against the RESOLVED mode and the descriptors
        actually built, and raises rather than serving a silently degraded
        candidate.

        There is deliberately NO opt-out (see R9's evidence report for why an
        earlier revision's `VLLM_ADAPTIVE_SPEC_ALLOW_DEGRADED_GRAPHS=1`
        escape hatch was removed): one environment variable, settable by
        anything that can reach the container environment, could otherwise
        turn a proven launch back into a silently-eager one after every
        launch-script guard had passed.
        """
        decode_query_lens = self.adaptive_spec_decode_query_lens()
        if not decode_query_lens:
            return

        if not self.cudagraph_mode.has_full_cudagraphs():
            raise RuntimeError(
                "Adaptive speculative depth requires FULL CUDA graphs, but "
                f"cudagraph_mode resolved to {self.cudagraph_mode.name}. Every "
                "depth on the ladder would run without a captured graph. Fix "
                "the attention backend or compilation config, or disable "
                "adaptive speculative depth."
            )
        if not separate_decode_routine:
            # A mixed FULL descriptor carries `uniform_token_count=None` and is
            # compatible with every uniform query length, so there is no
            # per-depth set to check.
            return

        covered = {
            (d.num_tokens, d.num_reqs, d.uniform_token_count, d.num_active_loras)
            for d in self._capture_descs.get(CUDAGraphMode.FULL, ())
        }
        missing = [
            (query_len, num_reqs)
            for query_len, num_reqs, num_active_loras in product(
                decode_query_lens,
                range(1, min(self.max_num_reqs, 32) + 1),
                self.lora_capture_cases,
            )
            if (
                query_len * num_reqs,
                num_reqs,
                query_len,
                num_active_loras,
            )
            not in covered
        ]
        if missing:
            raise RuntimeError(
                "Adaptive speculative depth has no captured FULL CUDA graph "
                f"for {len(missing)} reachable verification shape(s) "
                f"(query_len, num_reqs): {sorted(set(missing))}. The scheduler "
                "could select those depths and would fall back to eager "
                "execution. Raise max_cudagraph_capture_size to at least "
                f"{max(q * n for q, n in missing)}, shorten "
                "VLLM_ADAPTIVE_SPEC_DEPTHS, or lower max_num_seqs."
            )

        logger.info(
            "Adaptive speculative depth: FULL CUDA graph coverage verified for "
            "query lengths %s across request counts 1..%d.",
            decode_query_lens,
            min(self.max_num_reqs, 32),
        )

    def needs_capture(self) -> bool:
        return len(self._capture_descs) > 0

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs.

        Args:
            create_forward_fn: Factory that prepares inputs (OUTSIDE graph) and
                returns a forward_fn. For FULL and breakable PIECEWISE modes,
                it is invoked once with warmup=True and again with warmup=False
                because attention backends may mutate or lazily initialize
                metadata during warmup.
        """
        with graph_capture(device=self.device):
            # Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
            # activations so FULL activations should fit in already allocated
            # buffers in the graph pool.
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in self._capture_descs:
                    continue

                descs = self._capture_descs[mode]
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    # Prepare inputs and get forward function
                    forward_fn = create_forward_fn(desc, warmup=True)

                    # Warmup
                    forward_fn(CUDAGraphMode.NONE)

                    # Capture
                    logger.debug(
                        "CG Capture: mode=%s, batch_desc=%s", desc.cg_mode.name, desc
                    )
                    if (
                        desc.cg_mode == CUDAGraphMode.PIECEWISE
                        and not self.use_breakable_cg
                    ):
                        forward_fn(CUDAGraphMode.PIECEWISE)
                    else:
                        # Capture with fresh attention state.
                        forward_fn = create_forward_fn(desc, warmup=False)
                        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                            forward_fn(CUDAGraphMode.PIECEWISE)
                            continue
                        assert desc not in self.graphs, (
                            f"Graph already captured for {desc}"
                        )
                        graph = torch.cuda.CUDAGraph()
                        # Sync offloader's copy stream before capture.
                        # Ensure any pre-capture prefetches from offloader are complete.
                        get_offloader().sync_prev_onload()
                        if self.pool is not None:
                            set_graph_pool_id(self.pool)
                        else:
                            set_graph_pool_id(current_platform.graph_pool_handle())
                        with torch.cuda.graph(graph, self.pool):
                            forward_fn(CUDAGraphMode.NONE)
                            # Join offloader's copy stream after forward to avoid
                            # unjoined stream error. The last layer's start_prefetch
                            # forks copy_stream, but wait_prefetch only happens in
                            # the next forward pass.
                            get_offloader().join_after_forward()
                        self.graphs[desc] = graph
                        compilation_counter.num_cudagraph_captured += 1
        self._graphs_captured = True

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
    ) -> BatchExecutionDescriptor:
        """Find matching cudagraph descriptor from priority-ordered candidates."""

        effective_loras = self._resolve_effective_loras(num_active_loras)
        key = (num_tokens, effective_loras)
        if self._graphs_captured and num_tokens > 0 and key in self._candidates:
            for desc in self._candidates[key]:
                if _is_compatible(
                    desc,
                    num_reqs,
                    num_tokens,
                    uniform_token_count,
                    effective_loras,
                ):
                    return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=effective_loras,
        )

    def run_fullgraph(self, desc: BatchExecutionDescriptor):
        """Replay a captured FULL cudagraph."""
        assert desc.cg_mode == CUDAGraphMode.FULL, (
            f"Expected FULL mode, got {desc.cg_mode}"
        )
        assert desc in self.graphs, f"No cudagraph for {desc}"
        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        self.graphs[desc].replay()

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableCUDAGraphWrapper(
                model, self.vllm_config
            )

    def run_pw_graph(self, model: nn.Module, model_inputs: dict[str, Any]) -> Any:
        if not self.use_breakable_cg:
            # Default: Use torch-compiled piecewise cudagraph.
            return model(**model_inputs)
        assert self.breakable_cg_runner is not None
        return self.breakable_cg_runner(**model_inputs)


class ModelCudaGraphManager(CudaGraphManager):
    """CudaGraphManager with model-specific capture and hidden state management."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
        )
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.use_aux_hidden_state_outputs = False
        self.intermediate_tensors: IntermediateTensors | None = None

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
        if self.use_breakable_cg:
            self.init_breakable_cg_runner(model)

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)

            # Set LoRA state before capture so kernels see correct adapters.
            if lora_capture_hook is not None:
                lora_capture_hook(desc.num_active_loras, num_reqs, num_tokens)

            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )

            model_inputs = {
                "input_ids": input_buffers.input_ids[:num_tokens],
                "positions": input_buffers.positions[:num_tokens],
                **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
            }
            if not self.is_first_pp_rank:
                # Update for non-first PP ranks.
                model_inputs["input_ids"] = None
                model_inputs["inputs_embeds"] = None
                assert intermediate_tensors is not None
                model_inputs["intermediate_tensors"] = intermediate_tensors[:num_tokens]

            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                full_cudagraph=desc.cg_mode == CUDAGraphMode.FULL,
            )

            # Capture with dummy rows marked as padding.
            input_buffers.is_padding.fill_(True)

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                batch_descriptor = None
                if cg_mode == CUDAGraphMode.PIECEWISE:
                    batch_descriptor = BatchDescriptor(
                        num_tokens=num_tokens,
                        has_lora=has_lora,
                        num_active_loras=desc.num_active_loras,
                    )
                with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=cg_mode,
                    num_tokens_across_dp=num_tokens_across_dp,
                    slot_mapping=slot_mappings,
                    batch_descriptor=batch_descriptor,
                    is_padding=input_buffers.is_padding[:num_tokens],
                ):
                    if cg_mode == CUDAGraphMode.PIECEWISE:
                        # PIECEWISE graph (compiled PW or breakable, chosen inside
                        # run_pw_graph).
                        model_output = self.run_pw_graph(model, model_inputs)
                    else:
                        model_output = model(**model_inputs)

                if cg_mode == CUDAGraphMode.PIECEWISE:
                    # PW CUDA graph (compiled or breakable) internally handles the
                    # model outputs. No need to keep track of the hidden states.
                    return None

                if self.is_last_pp_rank:
                    # Last PP rank (common case).
                    if self.use_aux_hidden_state_outputs:
                        hidden_states, aux_hidden_states = model_output
                    else:
                        hidden_states = model_output
                        aux_hidden_states = []
                    if self.hidden_states is None:
                        self.hidden_states = torch.empty_like(hidden_states)
                    self.hidden_states[:num_tokens] = hidden_states
                    if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                        self.aux_hidden_states = [
                            torch.empty_like(x) for x in aux_hidden_states
                        ]
                    for i, aux in enumerate(aux_hidden_states):
                        self.aux_hidden_states[i][:num_tokens] = aux
                else:
                    # Non-last PP rank.
                    assert isinstance(model_output, IntermediateTensors)
                    intermediate_tensors = model_output
                    if self.intermediate_tensors is None:
                        self.intermediate_tensors = IntermediateTensors.empty_like(
                            intermediate_tensors
                        )
                    for k, v in intermediate_tensors.tensors.items():
                        self.intermediate_tensors[k][:num_tokens] = v

            return forward_fn

        super().capture(create_forward_fn, progress_bar_desc)

    def run_fullgraph(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        """Replay a captured FULL cudagraph and return hidden states."""
        super().run_fullgraph(desc)
        if not self.is_last_pp_rank:
            assert self.intermediate_tensors is not None
            return self.intermediate_tensors[: desc.num_tokens]

        assert self.hidden_states is not None
        hidden_states = self.hidden_states[: desc.num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[: desc.num_tokens] for x in self.aux_hidden_states]


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    full_cudagraph: bool,
) -> AttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    # HACK(woosuk): Special handling for DCP.
    if block_tables.cp_size > 1:
        prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
        )
        input_batch.dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]

    # NOTE(woosuk): Attention metadata is required not just by standard attention
    # kernels, but also by specialized attention-like operations (e.g., Inkling's sconv,
    # DSV4 compressor), which maintain their own states and require special metadata
    # such as block tables.
    # During CUDA graph capture:
    # - For FULL CUDA graphs: We set for_capture=True so that both attention and
    #   attention-like ops produce capturable metadata compatible with CUDA graphs.
    # - For PIECEWISE CUDA graphs: We still build attention metadata, but set
    #   for_capture=False. This is because:
    #     * Attention-like ops (such as sconv or DSV4 compressor) may not be used as
    #       breakpoints in PIECEWISE CUDA graphs, so we must generate their attention
    #       metadata so they can execute and be captured during graph capture.
    #     * Standard attention ops that are treated as breakpoints will be executed
    #       eagerly at capture time (not included in the graph itself), and for these,
    #       setting for_capture=False is essential. Some attention backends
    #       (like linear attention) cannot generate capturable metadata for prefill,
    #       so for_capture=False ensures they execute without issue.
    #     * We assume that attention-like operations intended for capture will still
    #       produce capturable metadata, even when for_capture=False. While this
    #       assumption is brittle, it currently works in practice.
    # In summary: We always generate attention metadata for both FULL and PIECEWISE
    # CUDA graphs, setting for_capture=True for FULL graphs, and for_capture=False
    # for PIECEWISE graphs, to ensure correct execution and capture.
    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=full_cudagraph,
    )
    return AttentionState(attn_metadata, slot_mappings_by_layer)
