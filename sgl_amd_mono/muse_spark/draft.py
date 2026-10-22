# SPDX-License-Identifier: Apache-2.0
"""Muse Spark's DFlash2 draft on gfx950: its spec and forward (RFC #43949).

SGLang attaches the draft model as it does the target: a routed draft step (a
block of DRAFT_TOKENS rows a request, run as a TARGET_VERIFY batch) runs
``_Draft.forward``, which takes and returns what ``DFlash2DraftModel.forward``
does: the final-normed hidden states of the block, which the worker's selector
turns into the draft tokens. The draft is a dense Qwen3-style stack (5 sliding
layers, hidden 8192, 64 / 8 heads of 128, intermediate 28672, a grouped two-tap
convolution around each sublayer) with no head of its own.

Stage 0 (``SGLANG_AMD_MONO_DRAFT_KERNELS=0``, the default while the kernels are
built): the forward is the model's own, so the wiring (attach, widths, block rows,
graph capture) is exercised with an output identical to the normal path.
"""

from __future__ import annotations

import logging
import os

import torch
from sglang.srt.monokernels.spec import DeviceLimits, Feature, MonoSpec, Requirements

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.muse_spark import DRAFT_TOKENS, TP
from sgl_amd_mono.muse_spark import spec as target
from sgl_amd_mono.muse_spark.gfx950.config import WIDTHS

logger = logging.getLogger(__name__)

# the draft the kernels are written for (RadixArk/Muse-Spark-Open-DFlash2)
DRAFT_CONFIG = dict(
    hidden_size=8192,
    num_hidden_layers=5,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=28672,
    sliding_window=2048,
    rms_norm_eps=1e-5,
)
DFLASH_CONFIG = dict(block_size=DRAFT_TOKENS, conv_kernel_size=2, conv_group_size=16, attention_mode="gqa")

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/monokernels/spec.py": ("DeviceLimits", "Feature", "MonoSpec", "Requirements"),
    "srt/models/dflash.py": (
        "DFlash2DraftModel",
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
        "block_size",
        "layers",
        "candidate_selector",
        "draft_config",
    ),
    "srt/runtime_context.py": ("get_parallel", "get_spec"),
}

USE_KERNELS = os.environ.get("SGLANG_AMD_MONO_DRAFT_KERNELS", "0") == "1"


def _tokens_per_request(model) -> int:
    """A draft block's rows a request: the block size (the anchor and the drafts)."""
    return int(model.block_size)


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    reasons = []
    for k, want in DRAFT_CONFIG.items():
        got = getattr(hf_config, k, None)
        if got != want:
            reasons.append(f"draft {k} {got}: the kernels take {want}")
    dc = getattr(hf_config, "dflash_config", None) or {}
    for k, want in DFLASH_CONFIG.items():
        got = dc.get(k) if isinstance(dc, dict) else getattr(dc, k, None)
        if got != want:
            reasons.append(f"draft dflash_config.{k} {got}: the kernels take {want}")
    types = getattr(hf_config, "layer_types", None)
    if types is not None and any(t != "sliding_attention" for t in types):
        reasons.append(f"draft layer_types {types}: the kernels take sliding layers only")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel

    reasons = []
    if get_parallel().tp_size != TP:
        reasons.append(f"draft TP {get_parallel().tp_size}: the kernels take TP {TP}")
    if int(model.block_size) != DRAFT_TOKENS:
        reasons.append(f"draft block {model.block_size}: the kernels take {DRAFT_TOKENS}")
    if getattr(model, "candidate_selector", None) is None:
        reasons.append("the draft has no candidate selector (not a DFlash2 draft)")
    if len(model.layers) != DRAFT_CONFIG["num_hidden_layers"]:
        reasons.append(f"draft layers {len(model.layers)}")
    for i, layer in enumerate(model.layers):
        if layer.attention_conv is None or layer.mlp_conv is None:
            reasons.append(f"draft layer {i}: no grouped convolutions")
        a = layer.self_attn
        if a.num_heads * TP != DRAFT_CONFIG["num_attention_heads"] or a.num_kv_heads * TP != DRAFT_CONFIG["num_key_value_heads"]:  # fmt: skip
            reasons.append(f"draft layer {i}: heads {a.num_heads} / {a.num_kv_heads} a rank")
        if a.v_scale is not None or a.attention_sink_bias is not None:
            reasons.append(f"draft layer {i}: a V scale or attention sinks")
    return reasons


class _Draft:
    """One rank's draft: stage 0 runs the model's own forward."""

    def __init__(self, model, original):
        self.model, self.original = model, original

    def forward(self, plan, input_ids, positions, forward_batch, input_embeds=None, *args, **kwargs):
        """``DFlash2DraftModel.forward`` for a routed draft step."""
        return self.original(input_ids, positions, forward_batch, input_embeds=input_embeds, *args, **kwargs)


def _build(model, hf_config, ctx) -> _Draft:
    if USE_KERNELS:
        raise NotImplementedError("the DFlash2 draft kernels are not built yet (SGLANG_AMD_MONO_DRAFT_KERNELS=1)")
    logger.info("monokernel: muse_spark DFlash2 draft attached at stage 0 (the model's own forward)")
    original = type(model).forward.__get__(model, type(model))
    return _Draft(model, original)


SPEC = MonoSpec(
    name="muse_spark_dflash2_draft_hip",
    devices={"gfx950": DeviceLimits(widths=WIDTHS, tp_sizes=(TP,), resident_blocks=BLOCKS)},
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    # the launcher checks the target's requirements; the draft's attention backend is its own
    requires=Requirements(attention_backends=frozenset({"triton"}), hf_config={"model_type": "qwen3"}),
    numerics=target.NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
    tokens_per_request=_tokens_per_request,
)
