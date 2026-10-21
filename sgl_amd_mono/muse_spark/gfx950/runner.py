# SPDX-License-Identifier: Apache-2.0
"""One rank's Muse Spark launches on gfx950.

Stage 0: every launch is SGLang's own ROCm decoder layer (``MuseSpark1xDecoderLayer``
with the model's fused cross-layer FFN boundary), so a routed step's output is
identical to the normal path. The runner already binds and checks the attention
backend and the KV pool the kernels will read, at the first routed forward.
"""

from __future__ import annotations

import torch

from sgl_amd_mono.muse_spark.gfx950.config import HEAD_DIM, LOCAL_KV_HEADS, PAGE, WINDOW

# The SGLang names the runner reads, by file under the sglang package;
# test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/layers/attention/aiter_backend.py": (
        "AiterAttnBackend",
        "page_size",
        "kv_cache_dtype",
        "use_triton_unified_attention",
        "use_sliding_window_kv_pool",
        "token_to_kv_pool",
    ),
    "srt/mem_cache/swa_memory_pool.py": ("SWAKVPool", "layers_mapping", "get_kv_buffer"),
    "srt/rocm_muse_spark/model.py": (
        "forward(attn_input)",
        "forward(ffn_boundary)",
        "forward(pre_norm_dtype)",
        "self_attn",
        "attn",
    ),
    "srt/layers/radix_attention.py": ("sliding_window_size", "layer_id"),
}

FP8_CACHE = torch.float8_e4m3fn


class MuseMonoRunner:
    def __init__(self, *, layers, boundaries, rank: int, npes: int, device):
        self.layers = list(layers)
        self.boundaries = list(boundaries)
        self.rank, self.npes, self.device = rank, npes, device
        self.deadline_s = None
        self.backend = None
        self.pool = None

    def set_deadline(self, seconds: float) -> None:
        self.deadline_s = float(seconds)

    def precompile(self, rows: int) -> None:
        """Stage 0 builds no kernel."""

    def cache_reasons(self, backend, pool) -> list[str]:
        """Why the kernels cannot read this backend's caches (empty: they can)."""
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        if not isinstance(backend, AiterAttnBackend):
            return [f"attention backend {type(backend).__name__}, the kernels read the aiter backend's caches"]
        why = []
        if backend.page_size != PAGE:
            why.append(f"page size {backend.page_size}, built for {PAGE}")
        if backend.kv_cache_dtype != FP8_CACHE:
            why.append(f"KV cache {backend.kv_cache_dtype}, built for {FP8_CACHE}")
        if not backend.use_triton_unified_attention:
            why.append("the kernels mirror aiter's unified attention")
        if not backend.use_sliding_window_kv_pool or type(pool).__name__ != "SWAKVPool":
            return why + [f"KV pool {type(pool).__name__}, the kernels read the sliding-window pool"]
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn.attn
            _, is_swa = pool.layers_mapping[attn.layer_id]
            if is_swa != (attn.sliding_window_size == WINDOW):
                why.append(f"layer {i}: pool {'sliding' if is_swa else 'full'} vs window {attn.sliding_window_size}")
            k, v = pool.get_kv_buffer(attn.layer_id)
            for t, what in ((k, "K"), (v, "V")):
                if not (t.dtype == FP8_CACHE and tuple(t.shape[-2:]) == (LOCAL_KV_HEADS, HEAD_DIM)):
                    why.append(f"layer {i} {what} cache: {t.dtype} {tuple(t.shape)}, want {FP8_CACHE} [*, {LOCAL_KV_HEADS}, {HEAD_DIM}]")  # fmt: skip
        return why

    def bind(self, backend, pool) -> None:
        self.backend, self.pool = backend, pool

    def launch(self, *, i: int, rows: int, positions, forward_batch, stream, attn_input, pre_norm_dtype):
        """Layer ``i`` on ``rows`` rows: (stream, next layer's attention input)."""
        return self.layers[i](
            positions,
            stream,
            forward_batch,
            attn_input=attn_input,
            ffn_boundary=self.boundaries[i],
            pre_norm_dtype=pre_norm_dtype,
        )
