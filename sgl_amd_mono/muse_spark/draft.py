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
graph capture) is exercised with an output identical to the normal path. Stage 1
(``=1``): each layer's attention half runs SGLang's modules and its MLP half (the
post norm's fused add, the convolutions, gate / up / down, the TP sum) runs the
D-MLP launch (``gfx950/draft_mlp.py``). Check mode (``SGLANG_AMD_MONO_CHECK=1``,
graphs off) runs SGLang's MLP half on copies and logs each layer's cosine.
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
        "norm",
        "input_layernorm",
        "post_attention_layernorm",
        "attention_conv",
        "mlp_conv",
        "self_attn",
        "mlp",
        "kernel_projection",
        "base_kernel",
        "gate_up_proj",
        "down_proj",
    ),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput", "next_token_logits", "hidden_states"),
    "srt/runtime_context.py": ("get_parallel", "get_spec"),
}

USE_KERNELS = os.environ.get("SGLANG_AMD_MONO_DRAFT_KERNELS", "0") == "1"


def _mlp_views(layer) -> dict:
    """A draft layer's MLP-half weights as the D-MLP launch reads them, in place."""
    conv = layer.mlp_conv
    return dict(
        w_post=layer.post_attention_layernorm.weight,
        w_kp=conv.kernel_projection.weight,
        base=conv.base_kernel,
        w_gu=layer.mlp.gate_up_proj.weight,
        w_down=layer.mlp.down_proj.weight,
    )


def _mlp_view_reasons(i: int, views: dict) -> list[str]:
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D

    shapes = {
        "w_post": (D.H,), "w_kp": (D.COEF_ROWS, D.H), "base": (2, D.TAPS, D.H), "w_gu": (D.GU_ROWS, D.H), "w_down": (D.H, D.I),
    }  # fmt: skip
    reasons = []
    for name, t in views.items():
        if not isinstance(t, torch.Tensor) or t.dtype != torch.bfloat16 or not t.is_contiguous() or tuple(t.shape) != shapes[name]:
            reasons.append(f"draft layer {i} {name}: {getattr(t, 'dtype', None)} {tuple(getattr(t, 'shape', ()))}, want contiguous bf16 {shapes[name]}")  # fmt: skip
    return reasons


class DraftRunner:
    """One rank's D-MLP launches: the layers' weights, the scratch, the peer
    buffer, the step epoch, one launch a (layer, rows)."""

    def __init__(self, layers, rank: int, npes: int, cpu_group, device):
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D
        from sgl_amd_mono.muse_spark.gfx950.config import MAX_TOKENS
        from sgl_amd_mono.muse_spark.gfx950.layout import scratch_bytes
        from sgl_amd_mono.muse_spark.gfx950.runner import _ensure_writable_flydsl_cache

        _ensure_writable_flydsl_cache()
        self.views = [_mlp_views(layer) for layer in layers]
        self.rank, self.npes, self.device = rank, npes, torch.device(device)
        dev = self.device
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        self.scratch = torch.zeros(scratch_bytes(D.dm_scratch_layout()), dtype=torch.uint8, device=dev)
        # the peer handshake is collective: every rank gets here (attach agreed first)
        self.peers = PeerBuffer(D.dm_sym_bytes(npes) + guard.FLAG_BYTES, cpu_group, rank, npes, dev)
        self.peers.bytes.zero_()
        self.res_out = torch.empty(MAX_TOKENS, D.H, dtype=torch.bfloat16, device=dev)
        self.out = torch.empty(MAX_TOKENS, D.H, dtype=torch.bfloat16, device=dev)
        self._launch: dict = {}

    def set_deadline(self, seconds: float) -> None:
        from sgl_amd_mono.common import guard

        ticks = guard.deadline_ticks(float(seconds))
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def kernel(self, rows: int):
        if rows not in self._launch:
            from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D

            self._launch[rows] = D.build_dmlp(D.DMLPBuild(tokens=rows, npes=self.npes))
        return self._launch[rows]

    def precompile(self, rows: int) -> None:
        from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

        compile_only(self.kernel(rows))

    def launch(self, i: int, rows: int, attn_out: torch.Tensor, residual: torch.Tensor, stream):
        """Layer ``i``'s MLP half on ``rows`` rows: (out, the updated residual), bf16."""
        from sgl_amd_mono.muse_spark.gfx950.runner import _ptr

        v = self.views[i]
        res_out, out = self.res_out[:rows], self.out[:rows]
        self.kernel(rows)(
            _ptr(residual), _ptr(attn_out), _ptr(v["w_post"]), _ptr(v["w_kp"]), _ptr(v["base"]), _ptr(v["w_gu"]), _ptr(v["w_down"]),
            _ptr(res_out), _ptr(out), _ptr(self.scratch), self.peers.local, _ptr(self.peers.addresses), self.rank,
            _ptr(self.epoch), i, stream=stream,
        )  # fmt: skip
        return out, res_out

    def end_step(self) -> None:
        self.epoch[0].add_(1)


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
    if USE_KERNELS:
        for i, layer in enumerate(model.layers):
            if layer.mlp_conv is not None and layer.attention_conv is not None:
                reasons += _mlp_view_reasons(i, _mlp_views(layer))
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
    """One rank's draft: stage 0 runs the model's own forward; stage 1 runs each
    layer's attention half on SGLang's modules and its MLP half as the D-MLP launch."""

    def __init__(self, model, original, runner=None):
        self.model, self.original, self.runner = model, original, runner
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1"

    def forward(self, plan, input_ids, positions, forward_batch, input_embeds=None, *args, **kwargs):
        """``DFlash2DraftModel.forward`` for a routed draft step."""
        if self.runner is None:
            return self.original(input_ids, positions, forward_batch, input_embeds=input_embeds, *args, **kwargs)
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        m = self.model
        if input_embeds is None:
            # the worker embeds the block with the target's table (a DFlash2 draft has none)
            raise ValueError("the Muse Spark DFlash2 draft forward needs input_embeds")
        rows = int(plan.key.rows)
        hidden, residual = input_embeds, None
        stream = torch.cuda.current_stream()
        check = self.check and not torch.cuda.is_current_stream_capturing()
        for i, layer in enumerate(m.layers):
            # the attention half, as DFlashDecoderLayer.forward runs it
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            hidden, kernel = layer.attention_conv.prepare(hidden)
            attn_out = layer.self_attn(positions=positions, hidden_states=hidden, forward_batch=forward_batch)
            attn_out = layer.attention_conv.finish(attn_out, kernel)
            ref = self._reference(layer, attn_out, residual) if check else None
            hidden, residual = self.runner.launch(i, rows, attn_out.contiguous(), residual.contiguous(), stream)
            if ref is not None:
                _log_check(i, (hidden, residual), ref)
        self.runner.end_step()
        hidden, _ = m.norm(hidden, residual)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)

    @staticmethod
    def _reference(layer, attn_out, residual):
        """Check mode: SGLang's MLP half on copies of the launch's inputs."""
        h, res = layer.post_attention_layernorm(attn_out.clone(), residual.clone())
        h, kernel = layer.mlp_conv.prepare(h)
        h = layer.mlp(h)
        return layer.mlp_conv.finish(h, kernel), res


def _log_check(i: int, got, ref) -> None:
    for name, g, r in (("out", got[0], ref[0]), ("residual", got[1], ref[1])):
        g, r = g.float(), r.float()
        cos = torch.nn.functional.cosine_similarity(g, r, dim=-1)
        rel = ((g - r).norm() / r.norm().clamp_min(1e-6)).item()
        logger.info("monokernel check: draft layer %d %s: rel %.4f, cosine min %.5f mean %.5f", i, name, rel, cos.min().item(), cos.mean().item())  # fmt: skip


def _build(model, hf_config, ctx) -> _Draft:
    original = type(model).forward.__get__(model, type(model))
    if not USE_KERNELS:
        logger.info("monokernel: muse_spark DFlash2 draft attached at stage 0 (the model's own forward)")
        return _Draft(model, original)
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    runner = DraftRunner(list(model.layers), p.tp_rank, p.tp_size, p.tp_group.cpu_group, model.norm.weight.device)
    runner.set_deadline(ctx.deadline_s)
    widths = list(ctx.widths)
    start = len(widths) * p.tp_rank // max(p.tp_size, 1)
    for rows in widths[start:] + widths[:start]:
        runner.precompile(rows)
    logger.info("monokernel: muse_spark DFlash2 draft attached at stage 1 (the MLP half as D-MLP launches)")
    return _Draft(model, original, runner)


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
