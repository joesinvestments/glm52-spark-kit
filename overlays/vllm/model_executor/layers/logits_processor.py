# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform

# --- lmhead-w8v2 (R17): top-M exact rescore over the INT8 lm_head ---------
# When VLLM_LMHEAD_V2_SIDECAR points at the checkpoint's bf16 lm_head sidecar
# (lmhead_w8v2_sidecar.safetensors), every _apply_head call takes the top-M
# candidates of this rank's quantized logits, recomputes exactly those rows
# in bf16 (fp32 accumulate) from the sidecar copy, and scatters the exact
# values back before argmax/sampling. Restores byte-exact-grade greedy
# decoding while the full-vocab GEMM stays INT8 Marlin (238 MB/rank read).
# Everything below is inert when the env var is unset.
#
# v2.1 (R17 shim-debug follow-up): self-evidencing + fail-loud.
#   - Logs ACTIVE/INERT per process at import, sidecar load per rank, and
#     a one-time FIRST RESCORE FIRED line with the changed-entry count.
#   - VLLM_LMHEAD_V2_REQUIRE=1 hard-fails any call path on which the
#     rescore would be silently inert (empty env in a worker process, or a
#     head_dtype branch that bypasses it).
#   - logits.view() instead of reshape(): a non-aliasing view now raises
#     instead of silently scattering into a copy.
# Numerics of the rescore itself are UNCHANGED vs v2.0.
import os as _lmv2_os

from vllm.logger import init_logger as _lmv2_init_logger

_lmv2_logger = _lmv2_init_logger(__name__)

_LMHEAD_V2_SIDECAR = _lmv2_os.environ.get("VLLM_LMHEAD_V2_SIDECAR", "")
_LMHEAD_V2_TOPM = int(_lmv2_os.environ.get("VLLM_LMHEAD_V2_TOPM", "64"))
# v2.1: VLLM_LMHEAD_V2_REQUIRE=1 turns "shim silently inert" into a hard
# boot/step failure. Set it on every rank whenever the v2 checkpoint is
# served; leave unset for non-v2 models.
_LMHEAD_V2_REQUIRE = _lmv2_os.environ.get("VLLM_LMHEAD_V2_REQUIRE", "") == "1"
_LMHEAD_V2_CHUNK = 256  # rows of hidden states rescored per sub-batch

# v2.1: per-process import-time evidence. This line MUST appear once per
# rank worker in the serve/ray logs; its absence in any rank is itself the
# diagnosis (env did not reach that process at import time).
_lmv2_logger.info(
    "lmhead-w8v2: %s at import (pid=%d, sidecar=%r, topm=%d, require=%s)",
    "ACTIVE" if _LMHEAD_V2_SIDECAR else "INERT",
    _lmv2_os.getpid(),
    _LMHEAD_V2_SIDECAR,
    _LMHEAD_V2_TOPM,
    _LMHEAD_V2_REQUIRE,
)


def _lmhead_v2_init(lm_head, device):
    """Load this rank's bf16 lm_head rows from the sidecar (one-time)."""
    from safetensors import safe_open

    with safe_open(_LMHEAD_V2_SIDECAR, framework="pt", device="cpu") as f:
        w_full = f.get_tensor("lm_head.weight_bf16")
    si = lm_head.shard_indices
    w = w_full[si.org_vocab_start_index : si.org_vocab_end_index]
    w = w.to(device=device, dtype=torch.bfloat16).contiguous()
    _lmv2_logger.info(
        "lmhead-w8v2: sidecar rank rows loaded (pid=%d, rows=[%d:%d], "
        "%.1f MB on %s)",
        _lmv2_os.getpid(),
        si.org_vocab_start_index,
        si.org_vocab_end_index,
        w.numel() * w.element_size() / 1e6,
        str(device),
    )
    return w


def _lmhead_v2_rescore(lm_head, hidden_states, logits):
    """Overwrite the local top-M quantized logits with exact bf16 values.

    CUDA-graph capture safe after one eager warmup call: topk/gather/bmm/
    scatter_ on static shapes, no host syncs. The one-time sidecar load is
    host-side, hence the loud guard against first-call-inside-capture."""
    if _LMHEAD_V2_TOPM <= 0:
        return logits
    w = getattr(lm_head, "_lmhead_v2_w", None)
    if w is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "lmhead-w8v2: sidecar state must be initialized by an eager "
                "warmup call before CUDA graph capture"
            )
        w = _lmhead_v2_init(lm_head, logits.device)
        lm_head._lmhead_v2_w = w
    n_local = w.shape[0]
    m = min(_LMHEAD_V2_TOPM, n_local)
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    # v2.1: .view (not .reshape) — view() RAISES when it cannot alias the
    # original storage, whereas reshape() silently returns a copy and the
    # scatter_ below would then be a silent no-op on the real logits.
    lg = logits.view(-1, logits.shape[-1])
    assert lg.data_ptr() == logits.data_ptr(), (
        "lmhead-w8v2: logits view does not alias logits storage"
    )
    # v2.1: one-time first-fire evidence per lm_head module. The first call
    # is the eager warmup (pre-capture), so the .item() sync is boot-only.
    first_fire = not getattr(lm_head, "_lmhead_v2_fired", False)
    if first_fire:
        lm_head._lmhead_v2_fired = True
    for s in range(0, flat.shape[0], _LMHEAD_V2_CHUNK):
        h = flat[s : s + _LMHEAD_V2_CHUNK]
        lgc = lg[s : s + _LMHEAD_V2_CHUNK]
        # candidates from the quantized logits (padding cols excluded)
        cand = lgc[:, :n_local].topk(m, dim=-1).indices
        # exact bf16 rows x hidden state, fp32 accumulate (torch.bmm)
        exact = torch.bmm(w[cand], h.unsqueeze(-1).to(w.dtype)).squeeze(-1)
        exact = exact.to(lgc.dtype)
        if first_fire and not torch.cuda.is_current_stream_capturing():
            first_fire = False
            n_changed = int((lgc.gather(-1, cand) != exact).sum().item())
            _lmv2_logger.info(
                "lmhead-w8v2: FIRST RESCORE FIRED (pid=%d, rows=%d, m=%d, "
                "entries_changed=%d)",
                _lmv2_os.getpid(),
                lgc.shape[0],
                m,
                n_changed,
            )
        lgc.scatter_(-1, cand, exact)
    return logits


# --8<-- [start:logits_processor]
@PluggableLayer.register("logits_processor")
class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    # --8<-- [end:logits_processor]

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()
        # Dtype of the lm_head projection. Defaults to the model dtype; an
        # fp32 head (via `--hf-overrides '{"head_dtype": "float32"}'`) is
        # required for RL training-inference consistency.
        model_config = get_current_vllm_config().model_config
        self.head_dtype = model_config.head_dtype if model_config is not None else None

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale
        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _apply_head(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project hidden states through the lm_head, honoring head_dtype."""
        if _LMHEAD_V2_REQUIRE and not _LMHEAD_V2_SIDECAR:
            raise RuntimeError(
                "lmhead-w8v2: VLLM_LMHEAD_V2_REQUIRE=1 but "
                "VLLM_LMHEAD_V2_SIDECAR is empty in this process (pid "
                f"{_lmv2_os.getpid()}) — the rescore would be silently "
                "inert. Fix the env plumbing to this rank worker."
            )
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
            logits = lm_head.quant_method.apply(
                lm_head, hidden_states, bias=embedding_bias
            )
            if _LMHEAD_V2_SIDECAR and logits is not None:
                assert embedding_bias is None, (
                    "lmhead-w8v2 rescore does not support an embedding bias"
                )
                logits = _lmhead_v2_rescore(lm_head, hidden_states, logits)
            return logits

        if _LMHEAD_V2_REQUIRE:
            raise RuntimeError(
                "lmhead-w8v2: VLLM_LMHEAD_V2_REQUIRE=1 but the "
                f"head_dtype={self.head_dtype} branch bypasses the rescore "
                "(rescore only covers head_dtype in (None, model dtype))."
            )
        if not isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError(
                "A head_dtype different from the model dtype is only "
                "supported for an unquantized lm_head."
            )
        if (
            self.head_dtype == torch.float32
            and (current_platform.is_cuda() or current_platform.is_rocm())
            and hidden_states.is_cuda
        ):
            # Accumulate the projection directly into fp32. This avoids
            # materializing an fp32 copy of the lm_head weight on every step,
            # unlike casting both operands. `torch.mm(out_dtype=...)` only
            # supports fp32 output for fp16/bf16 inputs, and is only
            # implemented for CUDA and ROCm (the latter via the non-Lt GEMM
            # path); other platforms fall back to the cast path below.
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.mm(flat, lm_head.weight.t(), out_dtype=self.head_dtype)
            if embedding_bias is not None:
                logits = logits + embedding_bias.to(self.head_dtype)
            return logits.reshape(*hidden_states.shape[:-1], -1)
        return F.linear(
            hidden_states.to(self.head_dtype),
            lm_head.weight.to(self.head_dtype),
            embedding_bias.to(self.head_dtype) if embedding_bias is not None else None,
        )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)

        # Gather logits for TP
        if lm_head.tp_size > 1:
            logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax, then only the (value, index) pairs
        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
        O(batch * vocab_size).
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = lm_head.tp_size

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        local_max_vals, local_max_indices = logits.max(dim=-1)

        # Convert shard-local indices to global vocab indices.
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start

        if tp_size == 1:
            return global_indices

        # All-gather (value, index) pairs, then reduce to global argmax.
        # Use float32 to avoid bf16 precision loss on large vocab indices.
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        # [batch, 2] -> [batch, 2 * tp_size]
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        # [batch, tp_size, 2] where [:, :, 0]=values, [:, :, 1]=indices
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
