# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# R9: canonical parsing of the VLLM_ADAPTIVE_SPEC_DEPTHS depth ladder.
#
# The pinned community overlay (CosmicRaisins/glm-5.2-gb10
# 600848707ce93fe42fedbc9dd4429116696e425d,
# patches/adaptive-mtp-vllm-hooks.patch) parses this variable TWICE, once in
# `vllm/v1/core/sched/scheduler.py` and once in
# `vllm/v1/worker/gpu/cudagraph_utils.py`, with two different results: the
# scheduler unions the configured `num_speculative_tokens` into its snap points
# and the CUDA-graph side does not. Any ladder whose largest rung is below the
# configured maximum therefore lets the scheduler select a depth the graph layer
# never captured a descriptor for, which silently drops that step to eager.
#
# R9 keeps ONE parser and both call sites use it, so the scheduled depth set and
# the captured descriptor set are the same object by construction. This is a
# deliberate reconciliation of the pinned patch, recorded in
# evidence/r9-adaptive-full-implementation-report.md.
from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

ADAPTIVE_SPEC_DEPTHS_ENV = "VLLM_ADAPTIVE_SPEC_DEPTHS"
# The R9 production ladder. The community overlay's own default is "2,4"; R9
# ships the tuned 2/4/5 recipe from adaptive-mtp/README.md as the default so an
# operator cannot get a silently different ladder by forgetting the variable.
ADAPTIVE_SPEC_DEPTHS_DEFAULT = "2,4,5"


def parse_adaptive_spec_depth_ladder(
    max_num_spec_tokens: int,
    raw: str | None = None,
) -> tuple[int, ...]:
    """Resolve the effective adaptive speculative-depth ladder.

    `max_num_spec_tokens` (the configured `num_speculative_tokens`) is the hard
    upper bound and is ALWAYS a rung of the returned ladder: it is the depth the
    speculator's buffers, the draft-token store and the fixed-K fallback are all
    sized for, so leaving it uncaptured would be the one hole that cannot be
    recovered from at runtime.

    Returns a sorted, de-duplicated tuple of depths in `[1, max_num_spec_tokens]`.
    """
    if max_num_spec_tokens <= 0:
        raise ValueError("max_num_spec_tokens must be greater than zero.")

    if raw is None:
        raw = os.getenv(ADAPTIVE_SPEC_DEPTHS_ENV, ADAPTIVE_SPEC_DEPTHS_DEFAULT)

    def _parse(text: str) -> list[int] | None:
        try:
            return sorted({int(tok) for tok in text.split(",") if tok.strip()})
        except ValueError:
            return None

    depths = _parse(raw)
    if depths is None:
        logger.warning(
            "%s=%r is not a comma-separated list of ints; falling back to %r.",
            ADAPTIVE_SPEC_DEPTHS_ENV,
            raw,
            ADAPTIVE_SPEC_DEPTHS_DEFAULT,
        )
        depths = _parse(ADAPTIVE_SPEC_DEPTHS_DEFAULT) or []

    valid = [depth for depth in depths if 1 <= depth <= max_num_spec_tokens]
    if valid != depths:
        logger.warning(
            "%s=%r: dropped out-of-range depths (valid range [1, %d]); using %s.",
            ADAPTIVE_SPEC_DEPTHS_ENV,
            raw,
            max_num_spec_tokens,
            valid,
        )
    if not valid:
        logger.warning(
            "%s=%r left no usable depth; falling back to the configured "
            "maximum %d.",
            ADAPTIVE_SPEC_DEPTHS_ENV,
            raw,
            max_num_spec_tokens,
        )

    return tuple(sorted(set(valid) | {max_num_spec_tokens}))
