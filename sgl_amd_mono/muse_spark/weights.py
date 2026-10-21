# SPDX-License-Identifier: Apache-2.0
"""What the kernels read of one Muse Spark decoder layer, checked at startup.

The kernels will read tensors through raw pointers: a differently-laid-out weight
would not fault, it would produce plausible garbage. So every module name, dtype,
shape and constant the kernels assume is checked once (``layer_reasons``, a refusal
at startup) on SGLang's ROCm Muse Spark modules. Stage 0 takes no views and makes
no copies; the checks pin the layout the kernel drop-in will meet.
"""

from __future__ import annotations

import math

import torch

from sgl_amd_mono.muse_spark import TP
from sgl_amd_mono.muse_spark.gfx950.config import (
    EXPERTS,
    GATE_TEMPERATURE,
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_EXPERTS,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MOE_HIDDEN,
    MXFP4_GROUP,
    POST_EPS,
    Q_K,
    QK_SCALE,
    QKVG_ROWS,
    RMS_EPS,
    ROUTE_EPS,
    ROUTE_SCALE,
    ROUTER_MULT,
    SOFTMAX_SCALE,
    TOP_K,
    WINDOW,
)

# The SGLang names these checks read, by file under the sglang package ("x": a
# string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/rocm_muse_spark/model.py": (
        "is_moe_layer",
        "self_attn",
        "mlp",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "use_qk_norm",
        "use_o_gate",
        "use_o_norm",
        "scaling",
        "scale_query_by",
        "norm_eps",
        "qkv_proj",
        "o_proj",
        "rotary_emb",
        "attn",
        "pre_attn_norm",
        "post_attn_norm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "post_attention_residual_gate",
        "post_feedforward_residual_gate",
        "tp_size",
        "ep_size",
        "num_local_experts",
        "top_k",
        "route_scale",
        "route_eps",
        "router_multiplier",
        "expert_hidden_size",
        "intermediate_size",
        "router",
        '"expert_bias"',
        "pre_expert_proj",
        "post_expert_proj",
        "pre_expert_norm",
        "post_expert_norm",
        "experts",
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "MuseSpark1xMxfp4Experts",
        "MuseSpark1xNativeExperts",
    ),
    "srt/rocm_muse_spark/layers.py": (
        "MuseSpark1xRMSNorm",
        "MuseSpark1xRMSNormWithInputScale",
        "eps",
        "has_weight",
        "weight",
        "gate",
        "temperature",
    ),
    "srt/layers/rotary_embedding/*.py": ("cos_sin_cache", "is_neox_style", "rotary_dim"),
    "srt/layers/radix_attention.py": ("sliding_window_size", "k_scale", "v_scale"),
}

NORM_TYPES = ("MuseSpark1xRMSNorm", "MuseSpark1xRMSNormWithInputScale")
EXPERT_TYPES = ("MuseSpark1xMxfp4Experts", "MuseSpark1xNativeExperts")


def _close(a, b: float) -> bool:
    return math.isclose(float(a), b, rel_tol=1e-9, abs_tol=0.0)


def _tensor_reasons(t, want_dtype, want_shape, what: str) -> list[str]:
    if t is None:
        return [f"{what}: missing"]
    if not (t.dtype == want_dtype and tuple(t.shape) == tuple(want_shape) and t.is_contiguous()):
        return [f"{what}: {t.dtype} {tuple(t.shape)}, want contiguous {want_dtype} {tuple(want_shape)}"]
    return []


def _norm_reasons(module, n: int, what: str, *, eps: float, weighted: bool) -> list[str]:
    """A MuseSpark1xRMSNorm of width ``n`` and ``eps``, with (``weighted``) a bf16
    gain the kernels read as ``weight + gain_center``."""
    if type(module).__name__ not in NORM_TYPES:
        return [f"{what}: {type(module).__name__}, the kernels apply MuseSpark1xRMSNorm"]
    why = []
    if not _close(module.eps, eps):
        why.append(f"{what}: eps {module.eps}, built for {eps}")
    if bool(module.has_weight) != weighted:
        why.append(f"{what}: {'weighted' if module.has_weight else 'weightless'}, built for {'weighted' if weighted else 'weightless'}")  # fmt: skip
    elif weighted:
        why += _tensor_reasons(module.weight, torch.bfloat16, (n,), f"{what} weight")
    return why


def _gate_reasons(module, what: str) -> list[str]:
    """A channel-wise norm-preserving residual gate at the built temperature."""
    g = module.gate
    if g is None or g.numel() != HIDDEN:
        return [f"{what}: gate {None if g is None else tuple(g.shape)}, want channel-wise [{HIDDEN}]"]
    if not _close(module.temperature, GATE_TEMPERATURE):
        return [f"{what}: temperature {module.temperature}, built for {GATE_TEMPERATURE}"]
    return []


def _attention_reasons(attn, i: int) -> list[str]:
    why = []
    shape = (attn.num_heads, attn.num_kv_heads, attn.head_dim)
    if shape != (LOCAL_Q_HEADS, LOCAL_KV_HEADS, HEAD_DIM):
        why.append(f"layer {i}: heads / dims {shape} a rank, built for {(LOCAL_Q_HEADS, LOCAL_KV_HEADS, HEAD_DIM)}")  # fmt: skip
    if not (attn.use_qk_norm and attn.use_o_gate and attn.use_o_norm):
        why.append(f"layer {i}: the kernels apply the q / k norms, the o norm and the o gate")
    if not _close(attn.scaling, SOFTMAX_SCALE):
        why.append(f"layer {i}: softmax scale {attn.scaling}, built for {SOFTMAX_SCALE}")
    if not _close(attn.scale_query_by, QK_SCALE):
        why.append(f"layer {i}: q scale {attn.scale_query_by}, built for {QK_SCALE}")
    if not _close(attn.norm_eps, RMS_EPS):
        why.append(f"layer {i}: q / k / o norm eps {attn.norm_eps}, built for {RMS_EPS}")
    why += _tensor_reasons(attn.qkv_proj.weight, torch.bfloat16, (QKVG_ROWS, HIDDEN), f"layer {i} qkv_proj")
    why += _tensor_reasons(attn.o_proj.weight, torch.bfloat16, (HIDDEN, Q_K), f"layer {i} o_proj")
    rope = attn.rotary_emb
    if rope is not None:
        table = rope.cos_sin_cache
        if not (table.dim() == 2 and table.shape[1] == HEAD_DIM and table.dtype == torch.float32):
            why.append(f"layer {i}: cos / sin cache {table.dtype} {tuple(table.shape)}, want fp32 [*, {HEAD_DIM}]")  # fmt: skip
        if rope.is_neox_style or rope.rotary_dim != HEAD_DIM:
            why.append(f"layer {i}: rotary neox {rope.is_neox_style} dim {rope.rotary_dim}, built for interleaved {HEAD_DIM}")  # fmt: skip
    window = attn.attn.sliding_window_size
    if window not in (WINDOW, -1):
        why.append(f"layer {i}: sliding window {window}, built for {WINDOW} or full attention")
    for scale, what in ((attn.attn.k_scale, "k_scale"), (attn.attn.v_scale, "v_scale")):
        if isinstance(scale, torch.Tensor) and scale.numel() != 1:
            why.append(f"layer {i} {what}: {tuple(scale.shape)}, want one scale")
    why += _norm_reasons(attn.pre_attn_norm, HIDDEN, f"layer {i} pre_attn_norm", eps=RMS_EPS, weighted=True)
    return why


def _experts_reasons(experts, i: int) -> list[str]:
    kind = type(experts).__name__
    if kind not in EXPERT_TYPES:
        return [f"layer {i}: experts {kind}, the kernels read Quark MXFP4 or bf16 experts"]
    E, what = LOCAL_EXPERTS, f"layer {i} experts"
    if kind == "MuseSpark1xNativeExperts":
        return _tensor_reasons(experts.w13_weight, torch.bfloat16, (E, 2 * INTER, MOE_HIDDEN), f"{what} w13") + _tensor_reasons(
            experts.w2_weight, torch.bfloat16, (E, MOE_HIDDEN, INTER), f"{what} w2"
        )
    why = _tensor_reasons(experts.w13_weight, torch.uint8, (E, 2 * INTER, MOE_HIDDEN // 2), f"{what} w13")
    why += _tensor_reasons(experts.w2_weight, torch.uint8, (E, MOE_HIDDEN, INTER // 2), f"{what} w2")
    why += _tensor_reasons(experts.w13_weight_scale, torch.uint8, (E, 2 * INTER, MOE_HIDDEN // MXFP4_GROUP), f"{what} w13 scale")  # fmt: skip
    why += _tensor_reasons(experts.w2_weight_scale, torch.uint8, (E, MOE_HIDDEN, INTER // MXFP4_GROUP), f"{what} w2 scale")  # fmt: skip
    return why


def _moe_reasons(moe, i: int) -> list[str]:
    why = []
    if (moe.tp_size, moe.ep_size) != (TP, TP):
        why.append(f"layer {i}: MoE TP {moe.tp_size} / EP {moe.ep_size}, the kernels shard experts over EP {TP}")
    shape = (moe.num_local_experts, moe.top_k, moe.expert_hidden_size, moe.intermediate_size)
    if shape != (LOCAL_EXPERTS, TOP_K, MOE_HIDDEN, INTER):
        why.append(f"layer {i}: experts / top-k / pooled / inter {shape}, built for {(LOCAL_EXPERTS, TOP_K, MOE_HIDDEN, INTER)}")  # fmt: skip
    for got, want, what in ((moe.route_scale, ROUTE_SCALE, "route scale"), (moe.route_eps, ROUTE_EPS, "route eps"),
                            (moe.router_multiplier, ROUTER_MULT, "router multiplier")):  # fmt: skip
        if not _close(got, want):
            why.append(f"layer {i}: {what} {got}, built for {want}")
    why += _tensor_reasons(moe.router.weight, torch.float32, (HIDDEN, EXPERTS), f"layer {i} router")
    why += _tensor_reasons(moe.expert_bias, torch.float32, (EXPERTS,), f"layer {i} expert_bias")
    for proj, shape, what in ((moe.pre_expert_proj, (MOE_HIDDEN, HIDDEN), "pooler"), (moe.post_expert_proj, (HIDDEN, MOE_HIDDEN), "unpooler")):  # fmt: skip
        if proj is None:
            why.append(f"layer {i}: no {what}; the kernels pool {HIDDEN} to {MOE_HIDDEN}")
        else:
            why += _tensor_reasons(proj.weight, torch.bfloat16, shape, f"layer {i} {what}")
    why += _norm_reasons(moe.pre_expert_norm, MOE_HIDDEN, f"layer {i} pre_expert_norm", eps=RMS_EPS, weighted=True)
    why += _norm_reasons(moe.post_expert_norm, MOE_HIDDEN, f"layer {i} post_expert_norm", eps=POST_EPS, weighted=True)
    return why + _experts_reasons(moe.experts, i)


def layer_reasons(layer, i: int) -> list[str]:
    """Every reason the kernels cannot read decoder layer ``i`` (empty: they can)."""
    if not layer.is_moe_layer:
        return [f"layer {i}: a dense MLP layer; the kernels run routed experts"]
    why = _attention_reasons(layer.self_attn, i)
    why += _norm_reasons(layer.post_attn_norm, HIDDEN, f"layer {i} post_attn_norm", eps=POST_EPS, weighted=False)
    why += _norm_reasons(layer.pre_feedforward_layernorm, HIDDEN, f"layer {i} pre_feedforward_layernorm", eps=RMS_EPS, weighted=True)  # fmt: skip
    why += _norm_reasons(layer.post_feedforward_layernorm, HIDDEN, f"layer {i} post_feedforward_layernorm", eps=POST_EPS, weighted=True)  # fmt: skip
    why += _gate_reasons(layer.post_attention_residual_gate, f"layer {i} post_attention_residual_gate")
    why += _gate_reasons(layer.post_feedforward_residual_gate, f"layer {i} post_feedforward_residual_gate")
    return why + _moe_reasons(layer.mlp, i)


def expert_kind(layer) -> str:
    """``"mxfp4"`` or ``"bf16"``: which experts a layer ``layer_reasons`` accepted holds."""
    return "bf16" if type(layer.mlp.experts).__name__ == "MuseSpark1xNativeExperts" else "mxfp4"
