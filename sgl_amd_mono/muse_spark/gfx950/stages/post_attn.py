# SPDX-License-Identifier: Apache-2.0
"""K2: one Muse Spark layer from the attention output to the experts' input,
for S = ``tokens`` rows; per token::

    o (bf16, the attention output), gate (K1's o_gate)
      -> per-head RMSNorm without weight (eps RMS_EPS) x sigmoid(gate), one bf16
         rounding (the fused norm-gate)
      -> o_proj GEMV (bf16 MFMA, fp32 accumulate, one rounding) -> the bf16
         partial rows pushed to every rank's ``attn`` region -> rank-ordered fp32
         sum, one rounding (the aiter one-stage all-reduce's numerics)
      -> attention boundary (the fused post-norm / residual / pre-norm with a
         weightless post norm): nb = bf16(attn_out rstd) (eps POST_EPS);
         stream = alpha stream + beta nb in fp32, in place; x = bf16(stream rstd2
         w_pre) (eps RMS_EPS), the FFN input
      -> router: the fp32 weight as bf16 hi + lo (the joint router / pooler's
         split): x @ hi + x @ lo in fp32, x ROUTER_MULT, sigmoid, + expert bias,
         top-8 by one integer max a pick (a tie to the lower expert), the picks'
         sigmoids / (sum + ROUTE_EPS) x ROUTE_SCALE -> ids (int32), scores (f32)
      -> pooler GEMV (bf16, one rounding) -> pre-expert norm (eps RMS_EPS,
         weight) -> expert input (bf16 [S, MOE_HIDDEN])

The work is task lists over the CTAs. Loads complete in order, so a CTA's
mailbox poll waits behind its own weight loads in flight: the last
``MAX_TOKENS`` CTAs (the chain CTAs, from ``CHAIN0``) issue their pooler rows'
weights at once, landed before their first poll, and run the per-token tasks
unqueued; the other CTAs (the stream CTAs) issue the pooler's weights after
their o tasks and wait on the tokens' flags behind them, as they must anyway::

    og      S tasks, stream CTA OG0 + t: token t's normed, gated o -> ``og``
    o       512 tasks (16 rows each), two a stream CTA, three on CTAs 0 ..
            O_THIRD: o_proj partial rows -> every rank's ``attn`` region
    a_part  a_parts(S) tasks a token, stream CTAs AP0 ..: the all-reduce sum of
            the part's columns, the boundary with the row's sums of squares
            exchanged through ``ss``, x -> ``xbuf`` (plain bf16 words), the
            stream in place, part 0 the token's ``xflag``
    router  128 tasks (16 expert tiles x an eighth of K), stream CTAs R0 ..:
            on the tokens' x staged in LDS by DMA, partial logits -> ``logits``
    pool    256 tasks (16 rows each), one a CTA: pooled rows -> ``pool``
    route   S tasks, chain CTA CHAIN0 + t: top-8 -> ids / scores
    xn      S tasks, chain CTA CHAIN0 + t: the pre-expert norm -> expert input

Width scaling (8 .. 64 rows): the o GEMV holds every token's og row in LDS
(S x 2 KB) and runs a token tile of 16 an MFMA accumulator, each tile's rows
summed through one reduction row set in turn; the router / pooler GEMVs stage x
in LDS by DMA in passes of ``XT`` = 8 tokens (128 KB, the same region) with one
accumulator a pass; the boundary runs ``a_parts(S)`` = 8 / 4 / 2 / 1 CTAs a token
so its tasks stay on the stream CTAs with two o tasks (``A_TASKS_MAX`` = 64);
the og tasks sit on the same CTAs. LDS is 141 KB at every width (``k2_lds_bytes``).

The geometry constants, ``k2_scratch_layout`` and ``sym_layout`` belong in
``layout.py`` and live here until the K2 / K3 ports merge: ``sym_layout`` is the
rank's peer buffer for the whole layer (K3 pushes the FFN partials to ``ffn``
and the unpooled rows to ``unp``), so the K3 port imports it from here.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    hw_exp2,
    hw_rsq,
    mfma_bf16,
    row_sum,
    rsrc,
    sum_partials,
)
from sgl_amd_mono.common.plan import BLOCKS, PAIR_BYTES, THREADS, WAVES, pair_layout
from sgl_amd_mono.muse_spark.gfx950.layout import TRACE_SLOTS
from sgl_amd_mono.common.sync import preg
from sgl_amd_mono.muse_spark.gfx950.config import (
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    LOCAL_KV_HEADS,
    LOG2E,
    MAX_TOKENS,
    MOE_HIDDEN,
    POST_EPS,
    Q_K,
    RMS_EPS,
    ROUTE_EPS,
    ROUTE_SCALE,
    ROUTER_MULT,
    TOP_K,
)
from sgl_amd_mono.muse_spark.gfx950.device import acquire_loads, release_stores, stamp
from sgl_amd_mono.muse_spark.gfx950.layout import (
    GQA,
    K_STEP,
    RED_FLOATS,
    ROWS,
    STEP_BATCH,
    TILE,
    scratch_bytes,
    token_tiles,
)

# ---------------------------------------------------------------- K2 geometry (for layout.py)
O_TASKS = HIDDEN // ROWS  # 512 o_proj row groups
O_K_PER_WAVE = Q_K // WAVES  # 128: the o GEMV's K over the waves
O_STEPS = O_K_PER_WAVE // K_STEP  # 4 MFMA steps a wave a token tile
OG_THREADS = Q_K // 4  # 256: the og task's threads, 4 o elements each
OG_POLL = 8  # tokens an og-row poll on a stream CTA (16 B a thread a token)
K_PER_WAVE = HIDDEN // WAVES  # 1024: the router / pooler GEMVs' K over the waves
K_STEPS = K_PER_WAVE // K_STEP  # 32
R_TILES = EXPERTS // ROWS  # 16 expert tiles
R_PARTS = WAVES  # the router's K split: a task takes one wave's share of K
R_TASKS = R_TILES * R_PARTS  # 128: partial logits, summed by the route tasks
P_TASKS = MOE_HIDDEN // ROWS  # 256 pooler tasks, one a CTA
XT = 8  # tokens a router / pooler pass: their x in LDS (16 KB a token)
A_PARTS_MAX = 8  # CTAs a token's attention boundary at most
A_PAIRS_MAX = HIDDEN // (2 * THREADS)  # 8: a thread's column pairs with the whole row on one CTA
CHAIN0 = BLOCKS - MAX_TOKENS  # 192: the chain CTAs, token bid - CHAIN0's route and xn
O_THIRD = O_TASKS - 2 * CHAIN0  # 128: CTAs 0 .. O_THIRD take a third o task
AP0 = O_THIRD  # the boundary tasks: the stream CTAs with two o tasks
OG0 = AP0  # the og tasks: the same CTAs, before their weights go out
A_TASKS_MAX = CHAIN0 - AP0  # 64 boundary tasks at most
R0 = 0  # the router tasks
assert 2 * CHAIN0 < O_TASKS <= 3 * CHAIN0 and R0 + R_TASKS <= CHAIN0
assert MAX_TOKENS <= CHAIN0 and MAX_TOKENS <= A_TASKS_MAX
assert K_STEPS % STEP_BATCH == 0 and K_STEPS % WAVES == 0 and O_K_PER_WAVE % K_STEP == 0
assert EXPERTS == 4 * 64 and TOP_K <= 64  # a lane holds 4 routing keys; the picks on one wave
assert HIDDEN == 16 * THREADS and MOE_HIDDEN == 8 * THREADS and Q_K == 2 * THREADS
assert P_TASKS == BLOCKS and MAX_TOKENS % TILE == 0 and OG_THREADS <= THREADS


def a_parts(s: int) -> int:
    """CTAs a token's attention boundary at width ``s``: the largest power of two
    (at most A_PARTS_MAX) whose tasks fit the A_TASKS_MAX stream CTAs."""
    p = A_PARTS_MAX
    while p > 1 and p * s > A_TASKS_MAX:
        p //= 2
    assert p * s <= A_TASKS_MAX and A_PAIRS_MAX % p == 0, (s, p)
    return p


def k2_scratch_layout() -> dict:
    """K2's scratch regions, name -> (byte offset, bytes), sized for MAX_TOKENS:
    the tokens' normed gated o (bf16 pairs), a flag a token for its x, the
    boundary parts' sums of squares (post, then pre) and their x flags, the
    router parts' partial logits (fp32), the pooled rows (bf16 pairs), x as plain
    bf16 words (``k2_xbuf``, bytes given as pairs), and the merged launch's
    hand-off flags: a token's routing / expert input out, a (request, KV head,
    row tile)'s o rows out."""
    return pair_layout(
        [
            ("k2_og", MAX_TOKENS * Q_K // 2),
            ("k2_xflag", MAX_TOKENS),
            ("k2_ss", 2 * MAX_TOKENS * A_PARTS_MAX),
            ("k2_xpart", MAX_TOKENS * A_PARTS_MAX),
            ("k2_logits", R_PARTS * MAX_TOKENS * EXPERTS),
            ("k2_pool", MAX_TOKENS * MOE_HIDDEN // 2),
            ("k2_xbuf", MAX_TOKENS * HIDDEN * 2 // PAIR_BYTES),
            ("k2_rflag", MAX_TOKENS),
            ("k2_xinflag", MAX_TOKENS),
            ("k2_oflag", MAX_TOKENS * LOCAL_KV_HEADS * 4),
            ("k2_trace", BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def sym_layout(npes: int) -> dict:
    """A rank's peer-symmetric buffer, name -> (byte offset, bytes): the attention
    and FFN partial rows, [source rank][token][width] as bf16 pairs, and the
    unpooled rows [token][HIDDEN / 2] (each rank's rows from that rank). Shared
    by the layer: K2 pushes ``attn``, K3 ``ffn`` and ``unp``. The guard's peer
    flags follow at ``sym_bytes(npes)``."""
    return pair_layout(
        [
            ("attn", npes * MAX_TOKENS * HIDDEN // 2),
            ("ffn", npes * MAX_TOKENS * MOE_HIDDEN // 2),
            ("unp", MAX_TOKENS * HIDDEN // 2),
        ],
        align=256,
    )


def sym_bytes(npes: int) -> int:
    """The peer buffer's data bytes: the guard flags' offset."""
    return scratch_bytes(sym_layout(npes))


def k2_xs_floats(tokens: int) -> int:
    """The ``xs`` LDS region: every token's og row (the o GEMV's B rows), then
    the router / pooler passes' x (XT tokens of HIDDEN bf16)."""
    return max(tokens * Q_K // 2, min(tokens, XT) * HIDDEN // 2)


def k2_lds_bytes(tokens: int) -> int:
    return 4 * (k2_xs_floats(tokens) + RED_FLOATS + ROWS * MAX_TOKENS + EXPERTS)


def k2_lds(tokens: int):
    """K2's LDS for a step of ``tokens`` rows: the o GEMV's B rows, then the
    router / pooler GEMVs' staged x (the same region), one reduction row set, a
    task's output rows for every token, a token's sigmoids."""

    @fx.struct
    class Lds:
        xs: fx.Array[fx.Float32, k2_xs_floats(tokens), 16]
        red: fx.Array[fx.Float32, RED_FLOATS, 16]
        misc: fx.Array[fx.Float32, ROWS * MAX_TOKENS, 16]
        sig: fx.Array[fx.Float32, EXPERTS, 16]

    return Lds


# ---------------------------------------------------------------- device helpers
def sigmoid(z):
    """Triton's ``tl.sigmoid``: 1 / (1 + exp2(-z log2(e)))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def route_key(v, e):
    """Signed-orderable i32 of score ``v`` with its low 8 bits replaced by 255 -
    expert ``e``: a pick is one integer max, a tie goes to the lower expert."""
    b = v.bitcast(fx.Int32)
    b = (b < 0).select(b ^ fx.Int32(0x7FFFFFFF), b)
    return (b & fx.Int32(-256)) | (255 - e)


@flyc.jit
def block_sum(c, v, slot):
    """The CTA's sum of ``v`` over its threads, in wave order; ``slot`` picks the
    WAVES words of ``red`` (consecutive calls alternate slots)."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    ws = butterfly(v, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(ws, red + (slot * WAVES + wave))
    gpu.barrier()
    tot = fx.ptr_load(red + slot * WAVES)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(red + (slot * WAVES + w))
    return tot


def ld_bf(ptr, e, n, cm=0):
    """``n`` (2, 4, 8 or 16) bf16 at element ``e`` (even) of a bf16 buffer, as
    f32; at most 4 dwords a load."""
    out = []
    for e0 in range(0, n, 8):
        m = min(8, n - e0)
        w = bo.buffer_load(
            rsrc(ptr), (e + e0) // 2, vec_width=m // 2, dtype=T.i32, cache_modifier=cm
        )
        if m == 2:
            lo, hi = bf2_f32(fx.Int32(w))
            out += [lo, hi]
        else:
            v = fx.Vector(w).bitcast(fx.BFloat16).to(fx.Float32)
            out += [fx.Float32(v[j]) for j in range(m)]
    return out


def ld_f32(ptr, e, n, cm=0):
    """``n`` (2, 4, 8 or 16) f32 at element ``e`` of an f32 buffer; at most 4 a load."""
    out = []
    for e0 in range(0, n, 4):
        m = min(4, n - e0)
        w = bo.buffer_load(rsrc(ptr), e + e0, vec_width=m, dtype=T.f32, cache_modifier=cm)
        v = fx.Vector(w)
        out += [fx.Float32(v[j]) for j in range(m)]
    return out


def st_f32(ptr, e, vals, cm=0):
    """``vals`` (2, 4, 8 or 16 f32) at element ``e`` of an f32 buffer; at most 4 a store."""
    for e0 in range(0, len(vals), 4):
        chunk = vals[e0 : e0 + 4]
        bo.buffer_store(
            fx.Vector.from_elements(chunk, fx.Float32), rsrc(ptr), e + e0, cache_modifier=cm
        )


def st_bf(ptr, e, vals, cm=0):
    """``vals`` (4 or 8 f32) as bf16 at element ``e`` of a bf16 buffer."""
    for e0 in range(0, len(vals), 4):
        bo.buffer_store(
            fx.Vector.from_elements(vals[e0 : e0 + 4], fx.Float32).to(fx.BFloat16),
            rsrc(ptr),
            e + e0,
            cache_modifier=cm,
        )


def st_words(ptr, d, words, cm=0):
    """``words`` (1 .. 8 i32) at dword ``d`` of a buffer; at most 4 a store."""
    for d0 in range(0, len(words), 4):
        chunk = words[d0 : d0 + 4]
        if len(chunk) == 1:
            bo.buffer_store(chunk[0], rsrc(ptr), d + d0, cache_modifier=cm)
        else:
            bo.buffer_store(
                fx.Vector.from_elements(chunk, fx.Int32), rsrc(ptr), d + d0, cache_modifier=cm
            )


# ---------------------------------------------------------------- og tasks
@flyc.jit
def stage_og(c, t, o_flag=None, QL=8):
    """og task: token ``t``'s o (16 heads of 64), per-head RMSNorm without weight
    x sigmoid(gate), one bf16 rounding, to the ``og`` mailbox; thread tid holds
    head tid / 16, dims 4 (tid % 16) ... ``o_flag`` (a merged launch): the
    request's (KV head, row tile) flags first, then device-scope loads."""
    tid, lane = c["tid"], c["lane"]
    put_bf, poll, og = c["put_bf"], c["poll"], c["og"]
    a = c["args"]
    o_cm = 0  # o from memory the host wrote: the default cache policy
    if const_expr(o_flag is not None):
        o_cm = CM_DEV  # the attention's device-scope stores, earlier in this launch
    if tid < OG_THREADS:
        e = (tid // 16) * HEAD_DIM + (lane % 16) * 4
        if const_expr(o_flag is not None):
            rocdl.sched_barrier(0)
            RT = (QL * GQA + 15) // 16  # the attention's row tiles a (request, KV head)
            poll(
                [
                    (o_flag, (t // QL) * LOCAL_KV_HEADS * RT + 2 * j, 2)
                    for j in range(LOCAL_KV_HEADS * RT // 2)
                ]
            )
            acquire_loads()
        ov = ld_bf(a["o"], t * Q_K + e, 4, o_cm)
        gv = ld_bf(a["g"], t * Q_K + e, 4)
        ss = fx.Float32(0.0)
        for j in range_constexpr(4):
            ss = ss + ov[j] * ov[j]
        ss = butterfly(ss, (8, 4, 2, 1))
        rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
        put_bf(og, t * Q_K + e, [ov[j] * rstd * sigmoid(gv[j]) for j in range(4)])


# ---------------------------------------------------------------- o tasks
def o_w_loads(c, task):
    """The o_proj weight words of o task ``task``'s rows, this lane's: in flight
    (a task past the last reads past the buffer: zeros, no traffic). Row-major
    bf16, 16 B a lane a k-step, read once a layer (nontemporal)."""
    lane, wave = c["lane"], c["wave"]
    row = task * ROWS + lane % 16
    r_wo = rsrc(c["args"]["w_o"], HIDDEN * Q_K * 2)
    return [
        fx.Vector(
            bo.buffer_load(
                r_wo,
                (row * Q_K + wave * O_K_PER_WAVE + s * K_STEP + 8 * (lane // 16)) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )
        for s in range(O_STEPS)
    ]


@flyc.jit
def stage_og_rows(c):
    """Every token's normed gated o, polled from ``og`` into LDS: the o tasks' B
    rows (16 B a thread a token, OG_POLL tokens a poll)."""
    S, tid, xs = c["S"], c["tid"], c["xs"]
    poll, og = c["poll"], c["og"]
    rocdl.sched_barrier(0)
    if tid < OG_THREADS:
        for t0 in range_constexpr(0, S, OG_POLL):
            n = min(OG_POLL, S - t0)
            got = poll([(og, (t0 + j) * (Q_K // 2) + tid * 2, 2) for j in range(n)])
            for j in range_constexpr(n):
                fx.ptr_store(
                    got[j][0].bitcast(fx.Float32), xs + ((t0 + j) * (Q_K // 2) + tid * 2)
                )
                fx.ptr_store(
                    got[j][1].bitcast(fx.Float32), xs + ((t0 + j) * (Q_K // 2) + tid * 2 + 1)
                )
    gpu.barrier()


@flyc.jit
def stage_o(c, task, wts):
    """o task ``task``: rows ``16 task ..`` of o_proj for every token tile (lane l
    of wave w holds row l % 16 and k = 128 w + 32 s + 8 (l / 16) .. of token
    16 t + l % 16, from LDS), the waves' accumulators summed in wave order, one
    bf16 rounding, the partial rows pushed to every rank's ``attn`` region.
    ``wts``: the task's weights, in flight (``o_w_loads``)."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    xs, red, misc, put_bf = c["xs"], c["red"], c["misc"], c["put_bf"]
    W, bases, rank, sy_attn = c["W"], c["bases"], c["rank"], c["sy_attn"]
    l16, g4 = lane % 16, lane // 16
    V4F = fx.Vector.make_type(4, fx.Float32)
    tiles = token_tiles(S)
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(tiles)]
    for s in range_constexpr(O_STEPS):
        k0 = wave * O_K_PER_WAVE + s * K_STEP + 8 * g4
        wa = wts[s].bitcast(fx.BFloat16)
        for t in range_constexpr(tiles):
            tok = fx.min(t * TILE + l16, S - 1)
            xb = fx.Vector(fx.ptr_load(xs + (tok * (Q_K // 2) + k0 // 2), result_type=V4F))
            accs[t] = mfma_bf16(wa, xb.bitcast(fx.BFloat16), accs[t])
    # each tile's accumulators through the one reduction row set in turn (LDS)
    for t in range_constexpr(tiles):
        fx.ptr_store(accs[t], red + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < ROWS * TILE:
            r = tid % ROWS
            tk = t * TILE + tid // ROWS
            if tk < S:
                fx.ptr_store(bf16_round(row_sum(red, r, tid // ROWS)), misc + (tk * ROWS + r))
        gpu.barrier()
    # pair i: token i / 8, rows 2 (i % 8), + 1; to every rank (itself too)
    n_pairs = (ROWS // 2) * S
    for rnd in range_constexpr(-(-n_pairs // THREADS)):
        i = tid + rnd * THREADS
        if i < n_pairs:
            tk = i // (ROWS // 2)
            rr = (i % (ROWS // 2)) * 2
            v0 = fx.ptr_load(misc + (tk * ROWS + rr))
            v1 = fx.ptr_load(misc + (tk * ROWS + rr + 1))
            for p in range_constexpr(W):
                put_bf(
                    preg(bases[p], sy_attn, "attn"),
                    (rank * MAX_TOKENS + tk) * HIDDEN + task * ROWS + rr,
                    [v0, v1],
                )
    gpu.barrier()  # ``misc`` is reused by the CTA's next task


# ---------------------------------------------------------------- boundary tasks
@flyc.jit
def stage_a_part(c, t, part):
    """Part ``part`` of token ``t``'s attention boundary (columns HIDDEN / A_PARTS
    part .., 2 PAIRS a thread): the ranks' partial o rows summed in rank order
    (``sum_partials``), one bf16 rounding; the post norm without weight and the
    pre-FFN norm with the row's sums of squares exchanged through ``ss``; the
    stream updated in place (fp32); x out as plain bf16 words; part 0 raises the
    token's flag once every part's x is out."""
    S, tid, W = c["S"], c["tid"], c["W"]
    put, poll = c["put"], c["poll"]
    own, ss_mb, xpart, xflag, xbuf = c["own_attn"], c["ss"], c["xpart"], c["xflag"], c["xbuf"]
    a = c["args"]
    A_PARTS = a_parts(S)
    PAIRS = A_PAIRS_MAX // A_PARTS
    NC = 2 * PAIRS  # columns a thread
    e0 = part * (HIDDEN // A_PARTS) + tid * NC
    # the operands first: after the poll each would be a dependent round trip
    prw = ld_bf(a["pre_w"], e0, NC)
    al = ld_f32(a["alpha"], e0, NC)
    be = ld_f32(a["beta"], e0, NC)
    st = ld_f32(a["res"], t * HIDDEN + e0, NC)
    rocdl.sched_barrier(0)
    # the all-reduce: pair j of this thread from every rank, summed in rank order
    pb = t * (HIDDEN // 2) + e0 // 2
    v = []
    for j in range_constexpr(PAIRS):
        pj = pb + j
        lo, hi = sum_partials(poll, own, lambda src: src * (MAX_TOKENS * HIDDEN // 2) + pj, W)
        v += [bf16_round(lo), bf16_round(hi)]
    ssp = v[0] * v[0]
    for k in range_constexpr(1, NC):
        ssp = ssp + v[k] * v[k]
    ssp = block_sum(c, ssp, 0)
    if const_expr(A_PARTS > 1):
        if tid == 0:
            put(ss_mb, (t * A_PARTS_MAX + part) * 2, ssp)
        got = poll([(ss_mb, (t * A_PARTS_MAX + q) * 2, 1) for q in range(A_PARTS)])
        tot = got[0][0].bitcast(fx.Float32)
        for q in range_constexpr(1, A_PARTS):
            tot = tot + got[q][0].bitcast(fx.Float32)
    else:
        tot = ssp
    rstd = hw_rsq(tot / float(HIDDEN) + POST_EPS)
    n = [bf16_round(v[k] * rstd) for k in range(NC)]
    new = [al[k] * st[k] + be[k] * n[k] for k in range(NC)]
    st_f32(a["res"], t * HIDDEN + e0, new, CM_DEV)
    ss2 = new[0] * new[0]
    for k in range_constexpr(1, NC):
        ss2 = ss2 + new[k] * new[k]
    ss2 = block_sum(c, ss2, 1)
    if const_expr(A_PARTS > 1):
        if tid == 0:
            put(ss_mb, (t * A_PARTS_MAX + part) * 2 + 1, ss2)
        got2 = poll([(ss_mb, (t * A_PARTS_MAX + q) * 2 + 1, 1) for q in range(A_PARTS)])
        tot2 = got2[0][0].bitcast(fx.Float32)
        for q in range_constexpr(1, A_PARTS):
            tot2 = tot2 + got2[q][0].bitcast(fx.Float32)
    else:
        tot2 = ss2
    rstd2 = hw_rsq(tot2 / float(HIDDEN) + RMS_EPS)
    words = [
        bf16_pair(new[2 * k] * rstd2 * prw[2 * k], new[2 * k + 1] * rstd2 * prw[2 * k + 1]).bitcast(fx.Int32)  # fmt: skip
        for k in range(PAIRS)
    ]
    st_words(xbuf, (t * HIDDEN + e0) // 2, words, CM_DEV)
    release_stores()
    gpu.barrier()
    if const_expr(A_PARTS > 1):
        if tid == 0:
            put(xpart, t * A_PARTS_MAX + part, fx.Int32(1))
        if part == 0:
            poll([(xpart, t * A_PARTS_MAX + q, 1) for q in range(A_PARTS)])
            if tid == 0:
                put(xflag, t, fx.Int32(1))
    else:
        if tid == 0:
            put(xflag, t, fx.Int32(1))


# ---------------------------------------------------------------- router and pooler GEMVs
def x_w_loads(c, w, rg, w_lo=None, part=None):
    """Every weight batch of ``x_gemv(c, rg, ..)``: rows ``16 rg ..`` of the bf16
    [rows, HIDDEN] weight ``w`` (and of ``w_lo``, the split router's low part),
    this lane's, in flight; with ``part`` (traced) over that eighth of K only,
    this wave's share. A list of (hi loads, lo loads) a batch."""
    lane, wave = c["lane"], c["wave"]
    l16, g4 = lane % 16, lane // 16
    row = rg * ROWS + l16
    r_w = rsrc(w)
    r_lo = rsrc(w_lo) if w_lo is not None else None
    steps = K_STEPS if part is None else K_STEPS // WAVES
    batch = min(STEP_BATCH, steps)

    def k_of(kc):
        if part is None:
            return wave * K_PER_WAVE + kc * K_STEP + 8 * g4
        return part * K_PER_WAVE + (wave * steps + kc) * K_STEP + 8 * g4

    def load(kc, r):
        return fx.Vector(
            bo.buffer_load(
                r, (row * HIDDEN + k_of(kc)) // 2, vec_width=4, dtype=T.i32, cache_modifier=CM_NT
            )
        )

    return [
        (
            [load(b * batch + j, r_w) for j in range(batch)],
            [load(b * batch + j, r_lo) for j in range(batch)] if w_lo is not None else [],
        )
        for b in range(steps // batch)
    ]


@flyc.jit
def x_gemv(c, rg, bufs, publish, t0, nt, part=None):
    """16 rows ``16 rg ..`` of a bf16 [rows, HIDDEN] weight on the staged tokens
    t0 .. t0 + nt (lane l16 holds token t0 + l16, clamped; the lo batches of
    ``bufs``, when present, add the same rows of the split router's low part;
    with ``part``, over that eighth of K only, this wave's share), the weights
    ``bufs`` (``x_w_loads``, in flight); ``publish(r, tk, total)`` takes row r
    of token tk, the waves' accumulators summed in wave order."""
    tid, lane, wave, xs, red = c["tid"], c["lane"], c["wave"], c["xs"], c["red"]
    l16, g4 = lane % 16, lane // 16
    V4F = fx.Vector.make_type(4, fx.Float32)
    if const_expr(part is None):
        steps = K_STEPS
    else:
        steps = K_STEPS // WAVES
    batch = min(STEP_BATCH, steps)
    tl = fx.min(l16, nt - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for b in range_constexpr(steps // batch):
        xls = []
        for j in range_constexpr(batch):
            kc = b * batch + j
            if const_expr(part is None):
                k = wave * K_PER_WAVE + kc * K_STEP + 8 * g4
            else:
                k = part * K_PER_WAVE + (wave * steps + kc) * K_STEP + 8 * g4
            xls.append(
                fx.Vector(fx.ptr_load(xs + (tl * (HIDDEN // 2) + k // 2), result_type=V4F))
            )
        for j in range_constexpr(batch):
            acc = mfma_bf16(bufs[b][0][j].bitcast(fx.BFloat16), xls[j].bitcast(fx.BFloat16), acc)
            if const_expr(len(bufs[b][1]) > 0):
                acc = mfma_bf16(bufs[b][1][j].bitcast(fx.BFloat16), xls[j].bitcast(fx.BFloat16), acc)  # fmt: skip
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * nt:
        publish(tid % ROWS, t0 + tid // ROWS, row_sum(red, tid % ROWS, tid // ROWS))
    gpu.barrier()  # the pass's x and red are done with


@flyc.jit
def wait_x(c):
    """The tokens' x flags, then device-scope loads."""
    S, poll, xflag = c["S"], c["poll"], c["xflag"]
    rocdl.sched_barrier(0)
    poll([(xflag, t, 1) for t in range(S)])
    acquire_loads()


@flyc.jit
def stage_pass(c, t0, nt):
    """Tokens t0 .. t0 + nt's x (``xbuf``, plain bf16 words) into LDS by DMA (no
    staging registers beside the weights in flight): 16 B a lane a round at
    M0 + 16 lane."""
    tid, wave, xs = c["tid"], c["wave"], c["xs"]
    r_x = rsrc(c["xbuf"])
    for i in range_constexpr(nt * (HIDDEN // 8) // THREADS):
        v = tid + THREADS * i
        rocdl.raw_ptr_buffer_load_lds(
            r_x,
            (xs + (i * THREADS + wave * 64) * 4).llvm_ptr,
            16,
            t0 * (HIDDEN * 2) + v * 16,
            fx.Int32(0),
            0,
            CM_DEV,
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()


@flyc.jit
def stage_pool_out(c):
    """This CTA's pooled rows (``misc``: token k's 16 rows from 16 k) to the
    ``pool`` mailbox, a pair a thread a round."""
    S, tid, bid, misc = c["S"], c["tid"], c["bid"], c["misc"]
    put_bf, pool = c["put_bf"], c["pool"]
    n_pairs = (ROWS // 2) * S
    for rnd in range_constexpr(-(-n_pairs // THREADS)):
        i = tid + rnd * THREADS
        if i < n_pairs:
            tk = i // (ROWS // 2)
            rr = (i % (ROWS // 2)) * 2
            put_bf(
                pool,
                tk * MOE_HIDDEN + bid * ROWS + rr,
                [fx.ptr_load(misc + (tk * ROWS + rr)), fx.ptr_load(misc + (tk * ROWS + rr + 1))],
            )


# ---------------------------------------------------------------- route and xn tasks
@flyc.jit
def stage_route(c, t, route_flag=None):
    """route task: token ``t``'s top-8 on wave 0 (lane l holds experts l, l + 64,
    ..): the parts' partial logits summed in part order, sigmoid(logit x
    ROUTER_MULT), + bias, a pick an integer max over ``route_key`` (a tie to the
    lower expert), the picks' sigmoids / (sum + ROUTE_EPS) x ROUTE_SCALE."""
    tid, lane, wave, sig = c["tid"], c["lane"], c["wave"], c["sig"]
    put, poll, logits = c["put"], c["poll"], c["logits"]
    a = c["args"]
    if wave == 0:
        got = poll(
            [
                (logits, (p * MAX_TOKENS + t) * EXPERTS + lane + 64 * h, 1)
                for p in range(R_PARTS)
                for h in range(4)
            ]
        )
        keys = []
        for h in range_constexpr(4):
            e = lane + 64 * h
            logit = got[h][0].bitcast(fx.Float32)
            for p in range_constexpr(1, R_PARTS):
                logit = logit + got[p * 4 + h][0].bitcast(fx.Float32)
            sc = sigmoid(logit * fx.Float32(ROUTER_MULT))
            fx.ptr_store(sc, sig + e)
            b = fx.Float32(bo.buffer_load(rsrc(a["bias"]), e, vec_width=1, dtype=T.f32))
            keys.append(route_key(sc + b, e))
        k0, k1, k2_, k3 = keys
        pid = fx.Int32(0)
        for p in range_constexpr(TOP_K):
            m = butterfly(fx.max(fx.max(k0, k1), fx.max(k2_, k3)), (32, 16, 8, 4, 2, 1), fx.max)  # fmt: skip
            k0 = (k0 == m).select(fx.Int32(-(2**31)), k0)
            k1 = (k1 == m).select(fx.Int32(-(2**31)), k1)
            k2_ = (k2_ == m).select(fx.Int32(-(2**31)), k2_)
            k3 = (k3 == m).select(fx.Int32(-(2**31)), k3)
            pid = (lane == p).select(255 - (m & 255), pid)
        if lane < TOP_K:
            w = fx.ptr_load(sig + pid)
            tot = butterfly(w, (1, 2, 4))
            bo.buffer_store(pid, rsrc(a["ids"]), t * TOP_K + lane, cache_modifier=CM_DEV)
            bo.buffer_store(
                (w / (tot + fx.Float32(ROUTE_EPS)) * fx.Float32(ROUTE_SCALE)).bitcast(fx.Int32),
                rsrc(a["scores"]),
                t * TOP_K + lane,
                cache_modifier=CM_DEV,
            )
    if const_expr(route_flag is not None):
        # the token's routing is out: its flag, for the experts' plan
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(route_flag, t, fx.Int32(1))


@flyc.jit
def stage_xn(c, t, xin_flag=None):
    """xn task: token ``t``'s pre-expert norm of its pooled row (8 elements a
    thread), weight ``xn_w``, one bf16 rounding -> the expert input."""
    tid, poll, put, pool = c["tid"], c["poll"], c["put"], c["pool"]
    a = c["args"]
    e0 = tid * 8
    got = poll([(pool, (t * MOE_HIDDEN + e0) // 2 + 2 * j, 2) for j in range(2)])
    v = []
    for j in range_constexpr(2):
        for q in range_constexpr(2):
            lo, hi = bf2_f32(got[j][q])
            v += [lo, hi]
    ss = fx.Float32(0.0)
    for j in range_constexpr(8):
        ss = ss + v[j] * v[j]
    rstd = hw_rsq(block_sum(c, ss, 0) / float(MOE_HIDDEN) + RMS_EPS)
    xw = ld_bf(a["xn_w"], e0, 8)
    out = [v[j] * rstd * xw[j] for j in range(8)]
    st_bf(a["xin"], t * MOE_HIDDEN + e0, out, CM_DEV)
    if const_expr(xin_flag is not None):
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(xin_flag, t, fx.Int32(1))


# ---------------------------------------------------------------- the CTAs' roles
@flyc.jit
def run_chain(c, route_flag=None, xin_flag=None):
    """A chain CTA (``bid >= CHAIN0``): its pooler rows' weights at once, the
    tokens' flags, the pooler passes, then token ``bid - CHAIN0``'s route and xn."""
    S, bid, misc = c["S"], c["bid"], c["misc"]
    a = c["args"]
    tok = bid - CHAIN0

    def put_pooled(r, tk, total):
        fx.ptr_store(bf16_round(total), misc + (tk * ROWS + r))

    pool_w = x_w_loads(c, a["w_pool"], bid)
    rocdl.sched_barrier(0)
    wait_x(c)
    stamp(c, 4)  # every token's x arrived
    for ps in range_constexpr(-(-S // XT)):
        t0 = ps * XT
        nt = min(XT, S - t0)
        stage_pass(c, t0, nt)
        x_gemv(c, bid, pool_w, put_pooled, t0, nt)
    stamp(c, 5)  # the pooler passes done
    stage_pool_out(c)
    stamp(c, 6)  # the pooled rows out
    if tok < S:
        stage_route(c, tok, route_flag)
        stage_xn(c, tok, xin_flag)


@flyc.jit
def run_stream(c, o_flag=None, QL=8):
    """A stream CTA (``bid < CHAIN0``): its og task, its o tasks (``bid``, ``bid +
    CHAIN0`` and, below O_THIRD, ``bid + 2 CHAIN0``; every CTA issues the third's
    loads, past the buffer on the rest), its boundary part, then its pooler
    weights, the tokens' flags, and the router / pooler passes (a router task's
    rows on CTAs R0 ..: every stream CTA loads a tile's rows, the index clamped;
    they stream while the first pass is staged)."""
    S, bid, misc, put, logits = c["S"], c["bid"], c["misc"], c["put"], c["logits"]
    a = c["args"]
    A_PARTS = a_parts(S)
    if (bid >= OG0) & (bid < OG0 + S):
        stage_og(c, bid - OG0, o_flag, QL)
    stamp(c, 1)  # the og task done (og CTAs)
    w_first = o_w_loads(c, bid)
    w_second = o_w_loads(c, bid + CHAIN0)
    w_third = o_w_loads(c, bid + 2 * CHAIN0)
    stage_og_rows(c)
    stage_o(c, bid, w_first)
    stage_o(c, bid + CHAIN0, w_second)
    if bid < O_THIRD:
        stage_o(c, bid + 2 * CHAIN0, w_third)
    stamp(c, 2)  # the o tasks done
    # the boundary parts, before this CTA's pooler weights go out (its poll
    # would wait behind them)
    if (bid >= AP0) & (bid < AP0 + A_PARTS * S):
        stage_a_part(c, (bid - AP0) // A_PARTS, (bid - AP0) % A_PARTS)
    stamp(c, 3)  # the boundary part done (a_part CTAs)
    pool_w = x_w_loads(c, a["w_pool"], bid)
    rocdl.sched_barrier(0)
    wait_x(c)
    stamp(c, 4)  # every token's x arrived
    r_tile = fx.min((bid - R0) // R_PARTS, fx.Int32(R_TILES - 1))
    r_part = (bid - R0) % R_PARTS
    router_w = x_w_loads(c, a["w_hi"], r_tile, a["w_lo"], r_part)

    def put_logit(r, tk, total):
        put(logits, (r_part * MAX_TOKENS + tk) * EXPERTS + r_tile * ROWS + r, total)

    def put_pooled(r, tk, total):
        fx.ptr_store(bf16_round(total), misc + (tk * ROWS + r))

    for ps in range_constexpr(-(-S // XT)):
        t0 = ps * XT
        nt = min(XT, S - t0)
        stage_pass(c, t0, nt)
        if (bid >= R0) & (bid < R0 + R_TASKS):
            x_gemv(c, r_tile, router_w, put_logit, t0, nt, r_part)
        x_gemv(c, bid, pool_w, put_pooled, t0, nt)
    stamp(c, 5)  # the router / pooler passes done
    stage_pool_out(c)
    stamp(c, 6)  # the pooled rows out


@flyc.jit
def run_k2(c, route_flag=None, xin_flag=None, o_flag=None, QL=8):
    """K2's stages on this CTA. ``route_flag`` / ``xin_flag`` (mailbox regions):
    in a launch that goes on to the experts, the route and xn tasks raise a
    token's flag once its routing / expert input is out. ``o_flag``: the
    attention ran earlier in this launch, a flag a (request of ``QL`` rows, KV
    head, row tile) once its o rows are out."""
    bid = c["bid"]
    stamp(c, 0)
    if bid >= CHAIN0:
        run_chain(c, route_flag, xin_flag)
    else:
        run_stream(c, o_flag, QL)
    stamp(c, 7)  # end
