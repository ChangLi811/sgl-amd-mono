# SPDX-License-Identifier: Apache-2.0
"""K1: one Muse Spark layer's attention front, the normed input through the KV
cache write, for S = ``tokens`` rows (a token each; 8 consecutive rows a request)::

    x (bf16: the layer's attention input, the previous boundary's pre-norm)
      -> q | k | v | gate GEMV on the bf16 weight (fp32 MFMA accumulate, one
         bf16 rounding, as the split-K GEMM of ``qkv_proj_qk_norm_rope_cache``)
      -> q, k: per-head RMSNorm without weight (eps 1e-5, bf16); q then scaled by
         qk_scale_factor and rounded; then interleaved RoPE in the layers that
         have one, with bf16 cos / sin and each product rounded, as the fused
         kernel of the ROCm model does
      -> q, k, v, gate out (bf16); K / V -> the layer's page-16 NHD fp8 cache
         (unit KV scale, checked at bind; the sliding layers' slots through the
         SWA pool's full-to-SWA table)

Stage 1 is the GEMV tasks: every row group (16 rows) as two K halves, 288 tasks
over the 256 CTAs (CTAs 0 .. 31 take a second), each half's fp32 partials to the
``k1_part`` mailbox. Stage 2 is a head task a q, k or v head on CTAs ``HT0`` ..
and a gate task a q head after them, each summing the halves in order and
rounding once, 32 tokens a pass on 16 lanes of 4 dims. Dependencies only point
from stage 2 to stage 1 and all CTAs are co-resident, so every wait makes
progress.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.mx import clamp_fp8
from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_rsq,
    mfma_bf16,
    row_sum,
    rsrc,
)
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES
from sgl_amd_mono.muse_spark.gfx950.config import (
    G_OFF,
    HEAD_DIM,
    HIDDEN,
    K_OFF,
    KV_K,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    Q_K,
    QK_SCALE,
    QKVG_ROWS,
    RMS_EPS,
    V_OFF,
)
from sgl_amd_mono.muse_spark.gfx950.device import acquire_loads
from sgl_amd_mono.muse_spark.gfx950.layout import (
    ELEMS,
    GATE_TASKS,
    GT0,
    HALVES,
    HEAD_TASKS,
    HT0,
    K_HALF,
    K_PER_WAVE,
    K_STEP,
    K_STEPS,
    RED_FLOATS,
    ROWS,
    SPLIT_TASKS,
    STEP_BATCH,
    TILE,
    TOKEN_LANES,
    TOKENS_PER_PASS,
    XS_LDS_MAX,
    token_passes,
    token_tiles,
)


def gemv_k(lane, wave, half, kc):
    """The k of step ``kc`` this lane streams, in K half ``half``."""
    return half * K_HALF + wave * K_PER_WAVE + kc * K_STEP + 8 * (lane // 16)


def gemv_w_batch(c, rg, half, b, w):
    """Batch ``b`` of GEMV task (``rg``, ``half``)'s weight loads, this lane's: in
    flight. Row-major bf16, 16 B a lane a k-step, read once a layer (nontemporal)."""
    lane, wave = c["lane"], c["wave"]
    row = rg * ROWS + lane % 16
    r_w = rsrc(w)
    return [
        fx.Vector(
            bo.buffer_load(
                r_w,
                (row * HIDDEN + gemv_k(lane, wave, half, b * STEP_BATCH + j)) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )
        for j in range(STEP_BATCH)
    ]


@flyc.jit
def stage_x(c):
    """Narrow steps: every token's x into LDS once, 16 B a thread a round."""
    S, tid, xs = c["S"], c["tid"], c["xs"]
    if const_expr(xs is None):
        return
    r_x = rsrc(c["args"]["x"])
    for i in range_constexpr(S * HIDDEN // 8 // THREADS):
        v = tid + THREADS * i  # 16 B unit
        fx.ptr_store(
            fx.Vector(
                bo.buffer_load(r_x, v * 4, vec_width=4, dtype=T.i32, cache_modifier=c["x_cm"])
            ).bitcast(fx.Float32),
            xs + v * 4,
        )
    gpu.barrier()


@flyc.jit
def stage_gemv(c, task, first=None):
    """GEMV task ``task``: K half ``task % 2`` of row group ``task / 2``, for every
    token tile. Lane l of wave w holds row l % 16 and, per k-step, k = half 4096 +
    512 w + 32 kc + 8 (l / 16) .. + 8 of the weight (A) and of token 16 t + l % 16
    (B, from LDS on narrow steps); the waves' accumulators are summed in wave order
    and the half's fp32 partials go to the mailbox. ``first``: the leading weight
    batches, already in flight (``gemv_w_batch``)."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red, xs, put, part = c["red"], c["xs"], c["put"], c["part"]
    a = c["args"]
    rg = task // HALVES
    half = task % HALVES
    tiles = token_tiles(S)
    r_x = rsrc(a["x"])
    x_cm = c["x_cm"]
    V4F = fx.Vector.make_type(4, fx.Float32)

    def w_batch(b):
        if const_expr(first is not None and b < len(first)):
            return first[b]
        return gemv_w_batch(c, rg, half, b, a["w"])

    def x_batch(b, t):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        if const_expr(xs is None):
            return [
                fx.Vector(
                    bo.buffer_load(
                        r_x,
                        (tok * HIDDEN + gemv_k(lane, wave, half, b * STEP_BATCH + j)) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=x_cm,
                    )
                )
                for j in range(STEP_BATCH)
            ]
        return [
            fx.Vector(
                fx.ptr_load(
                    xs + (tok * (HIDDEN // 2) + gemv_k(lane, wave, half, b * STEP_BATCH + j) // 2),
                    result_type=V4F,
                )
            )
            for j in range(STEP_BATCH)
        ]

    # each batch's weights go out before the previous batch's MFMAs; a token
    # tile's x right before its MFMAs (L2-resident past LDS)
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(tiles)]
    nb = K_STEPS // STEP_BATCH
    cur = w_batch(0)
    for b in range_constexpr(nb):
        if const_expr(b + 1 < nb):
            nxt = w_batch(b + 1)
        for t in range_constexpr(tiles):
            xb = x_batch(b, t)
            for j in range_constexpr(STEP_BATCH):
                accs[t] = mfma_bf16(
                    cur[j].bitcast(fx.BFloat16), xb[j].bitcast(fx.BFloat16), accs[t]
                )
        if const_expr(b + 1 < nb):
            cur = nxt
    for t in range_constexpr(tiles):
        fx.ptr_store(accs[t], red + (t * RED_FLOATS + (wave * 64 + lane) * 4))
    gpu.barrier()
    # this half's fp32 partials: the head and gate tasks sum the halves and round
    n_out = ROWS * S
    for p in range_constexpr(-(-n_out // THREADS)):
        idx = tid + p * THREADS
        if idx < n_out:
            r = idx % ROWS
            tk = idx // ROWS
            v = row_sum(red + (tk // TILE) * RED_FLOATS, r, tk % TILE)
            put(part, (half * MAX_TOKENS + tk) * QKVG_ROWS + rg * ROWS + r, v)
    gpu.barrier()  # ``red`` is reused by the CTA's next task


def half_sum(poll, part, tk, pb):
    """Token ``tk``'s elements ``pb`` .. ``pb + ELEMS`` of the projection: the K
    halves' fp32 partials from the mailbox, summed in half order, bf16."""
    gq = poll(
        [
            (part, (hf * MAX_TOKENS + tk) * QKVG_ROWS + pb + 2 * p, 2)
            for hf in range(HALVES)
            for p in range(2)
        ]
    )
    out = []
    for j in range_constexpr(ELEMS):
        v = gq[j // 2][j % 2].bitcast(fx.Float32)
        for hf in range_constexpr(1, HALVES):
            v = v + gq[hf * 2 + j // 2][j % 2].bitcast(fx.Float32)
        out.append(bf16_round(v))
    return out[0], out[1], out[2], out[3]


@flyc.jit
def stage_head(c, h):
    """Head task ``h`` (q heads, then k heads, then v heads): every token's head
    from the halves' partials, token ``tid / 16`` (+ 32 a pass) on 16 lanes of
    ``ELEMS`` dims each. q and k: the per-head norm (q then the query scale) and
    RoPE; k and v: the fp8 cache write."""
    S, tid, lane = c["S"], c["tid"], c["lane"]
    poll, part = c["poll"], c["part"]
    a = c["args"]
    is_q = h < LOCAL_Q_HEADS
    is_k = (h >= LOCAL_Q_HEADS) & (h < LOCAL_Q_HEADS + LOCAL_KV_HEADS)
    kv_head = is_k.select(h - LOCAL_Q_HEADS, h - LOCAL_Q_HEADS - LOCAL_KV_HEADS)
    col0 = is_q.select(h * HEAD_DIM, is_k.select(K_OFF, V_OFF) + kv_head * HEAD_DIM)
    normed = is_q | is_k
    d0 = (lane % 16) * ELEMS
    use_rope, use_swa = a["use_rope"], a["use_swa"]
    slot64, swa64 = a["slot64"], a["swa64"]
    r_cs = rsrc(a["cos_sin"])
    for p in range_constexpr(token_passes(S)):
        tk = tid // TOKEN_LANES + p * TOKENS_PER_PASS
        if tk < S:
            # positions are int64 and slots int64 or int32: this token's low word
            pos = fx.Int32(
                bo.buffer_load(rsrc(a["positions"]), 2 * tk, vec_width=1, dtype=T.i32)
            )
            slot = fx.Int32(
                bo.buffer_load(
                    rsrc(a["slots"]), (slot64 != 0).select(2 * tk, tk), vec_width=1, dtype=T.i32
                )
            )
            if (use_swa != 0) & (slot >= 0):
                slot = fx.Int32(
                    bo.buffer_load(
                        rsrc(a["swa_map"]),
                        (swa64 != 0).select(2 * slot, slot),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
            # this lane's two RoPE pairs: cos / sin of dims d0 / 2 and d0 / 2 + 1
            cs = fx.Vector(bo.buffer_load(r_cs, pos * HEAD_DIM + d0 // 2, vec_width=2, dtype=T.f32))
            sn = fx.Vector(
                bo.buffer_load(r_cs, pos * HEAD_DIM + HEAD_DIM // 2 + d0 // 2, vec_width=2, dtype=T.f32)
            )
            e0, e1, e2, e3 = half_sum(poll, part, tk, col0 + d0)
            e = [e0, e1, e2, e3]
            # the per-head norm: sum of squares over the head's 16 lanes
            ss = fx.Float32(0.0)
            for j in range_constexpr(ELEMS):
                ss = ss + e[j] * e[j]
            ss = butterfly(ss, (8, 4, 2, 1))
            rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
            n = [bf16_round(e[j] * rstd) for j in range(ELEMS)]
            # the query scale after the norm, rounded
            n = [is_q.select(bf16_round(n[j] * QK_SCALE), n[j]) for j in range(ELEMS)]
            # interleaved RoPE on (x, y) = dims (2i, 2i + 1), bf16 cos / sin, each
            # product rounded, then the sum rounded (the fused kernel's order)
            c0, c1 = bf16_round(cs[0]), bf16_round(cs[1])
            s0, s1 = bf16_round(sn[0]), bf16_round(sn[1])
            rot = [
                bf16_round(n[0] * c0) - bf16_round(n[1] * s0),
                bf16_round(n[1] * c0) + bf16_round(n[0] * s0),
                bf16_round(n[2] * c1) - bf16_round(n[3] * s1),
                bf16_round(n[3] * c1) + bf16_round(n[2] * s1),
            ]
            roped = normed & (use_rope != 0)
            out = [
                normed.select(roped.select(bf16_round(rot[j]), n[j]), e[j])
                for j in range(ELEMS)
            ]
            vec = fx.Vector.from_elements(out, fx.Float32).to(fx.BFloat16)
            if is_q:
                bo.buffer_store(vec, rsrc(a["q_out"]), tk * Q_K + h * HEAD_DIM + d0)
            else:
                if is_k:
                    bo.buffer_store(vec, rsrc(a["k_out"]), tk * KV_K + kv_head * HEAD_DIM + d0)
                else:
                    bo.buffer_store(vec, rsrc(a["v_out"]), tk * KV_K + kv_head * HEAD_DIM + d0)
                if slot >= 0:
                    packed = fp8_pack4(
                        clamp_fp8(out[0]), clamp_fp8(out[1]), clamp_fp8(out[2]), clamp_fp8(out[3])
                    )
                    # NHD page-16 fp8: slot s, head hh, dim d at byte (s KV_HEADS + hh) HEAD_DIM + d
                    idx = (slot * LOCAL_KV_HEADS + kv_head) * (HEAD_DIM // 4) + d0 // 4
                    if is_k:
                        bo.buffer_store(packed, rsrc(a["k_cache"]), idx)
                    else:
                        bo.buffer_store(packed, rsrc(a["v_cache"]), idx)


@flyc.jit
def stage_gate(c, g):
    """Gate task ``g``: rows ``G_OFF + 64 g`` .. of every token from the halves'
    partials, rounded once, out as bf16 (the o_gate of q head ``g``)."""
    S, tid, lane = c["S"], c["tid"], c["lane"]
    poll, part = c["poll"], c["part"]
    g_out = rsrc(c["args"]["g_out"])
    d0 = (lane % 16) * ELEMS
    for p in range_constexpr(token_passes(S)):
        tk = tid // TOKEN_LANES + p * TOKENS_PER_PASS
        if tk < S:
            e = half_sum(poll, part, tk, G_OFF + g * HEAD_DIM + d0)
            vec = fx.Vector.from_elements(list(e), fx.Float32).to(fx.BFloat16)
            bo.buffer_store(vec, g_out, tk * Q_K + g * HEAD_DIM + d0)


@flyc.jit
def run_k1(c, bid, x_flag=None):
    """K1's stages on this CTA: the x staging, its GEMV task(s), then its head or
    gate task. ``x_flag`` (a mailbox region): x was written earlier in this launch
    (the previous layer's boundary), flagged a token each; the first task's first
    weight batches go out before the wait."""
    first = None
    if const_expr(x_flag is not None):
        first = [gemv_w_batch(c, bid // HALVES, bid % HALVES, b, c["args"]["w"]) for b in range(2)]
        rocdl.sched_barrier(0)
        c["poll"]([(x_flag, t, 1) for t in range(c["S"])])
        acquire_loads()
    stage_x(c)
    stage_gemv(c, bid, first)
    if bid < SPLIT_TASKS - BLOCKS:
        stage_gemv(c, BLOCKS + bid)
    if (bid >= HT0) & (bid < HT0 + HEAD_TASKS):
        stage_head(c, bid - HT0)
    if (bid >= GT0) & (bid < GT0 + GATE_TASKS):
        stage_gate(c, bid - GT0)


def k1_lds(tokens: int):
    """K1's LDS for a step of ``tokens`` rows: a reduction row set a token tile
    and, on narrow steps, the tokens' x."""
    stage = tokens <= XS_LDS_MAX

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, token_tiles(tokens) * RED_FLOATS, 16]
        xs: fx.Array[fx.Float32, (tokens * HIDDEN // 2) if stage else 4, 16]

    return Lds
