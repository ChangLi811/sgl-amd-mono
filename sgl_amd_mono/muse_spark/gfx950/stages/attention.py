# SPDX-License-Identifier: Apache-2.0
"""K-attn: one Muse Spark layer's attention over the fp8 KV cache, for S =
``tokens`` rows of R = S / QL requests, ``QL`` consecutive rows a request (a
DFlash verify's 8 draft tokens, or one for decode). Per request and KV head, the
rank's 8 query heads of each of its QL rows attend the request's keys, as the
ROCm model's verify kernels (``swa_verify_splitkv``, ``gqa_verify_splitkv_2kv``)
and aiter's unified attention compute them::

    S = scale log2(e) (q . k)   q bf16 (K1 already applied the query scale),
                                K E4M3 upcast to bf16, fp32
    masked: key < context + qpos + 1 (causal), context + qpos - key < window
    online softmax in the exp2 domain, P -> bf16, O += P V (V upcast)
    O / l -> bf16

Tasks: the attention tasks, one per (request, KV head, row tile of 16 rows,
partition of ``part`` keys; the partitions a sliding window masks whole are
skipped), each a loop over 256-key chunks with the waves splitting the keys: QK
on wave-local key tiles, the row statistics, P through LDS, V transposed through
LDS, PV; the key splits merged in LDS; the partial (m, l, O) published. Then one
merge task per (request, KV head, row tile) combines the partitions and writes O.

The row of a request's KV head group is ``8 qpos + head``: QL = 8 fills four
MFMA row tiles (four tasks a partition), QL = 1 half of one.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    butterfly,
    fp8_to_bf16x8,
    hw_exp2,
    mfma_bf16,
    rsrc,
)
from sgl_amd_mono.common.plan import BLOCKS, WAVES
from sgl_amd_mono.muse_spark.gfx950.config import HEAD_DIM, LOCAL_KV_HEADS, LOG2E, Q_K
from sgl_amd_mono.muse_spark.gfx950.device import release_stores, stamp
from sgl_amd_mono.muse_spark.gfx950.layout import (
    ATTN_CHUNK as CHUNK,
    ATTN_KEY_SPLITS as KEY_SPLITS,
    ATTN_KEY_TILES as KEY_TILES,
    ATTN_KEYS_PER_WAVE as KEYS_PER_WAVE,
    ATTN_MAX_PARTS as MAX_PARTS,
    ATTN_P_PITCH as P_PITCH,
    ATTN_PART_ROW as PART_ROW,
    ATTN_Q_PITCH as Q_PITCH,
    ATTN_V_PITCH as V_PITCH,
    ATTN_V_STEPS as V_STEPS,
    GQA,
)

TILE = 16  # keys an MFMA tile
NEG = -1.0e30  # a masked score, and "no key yet" of a row's running max
NO_MAX = -1.0e29  # a running max below this is "no key yet": exp2 then takes 0


def kattn_lds(nt: int):
    """K-attn's LDS: a task's Q tile(s), each wave's P tile(s) and V^T, the key
    splits' merge rows, a reduction row."""

    @fx.struct
    class Lds:
        qs: fx.Array[fx.BFloat16, nt * 16 * Q_PITCH, 16]  # the task's Q rows (its row tiles)
        pt: fx.Array[fx.BFloat16, WAVES * nt * 16 * P_PITCH, 16]  # each wave's P tiles
        vt: fx.Array[fx.BFloat16, WAVES * HEAD_DIM * V_PITCH, 16]  # each wave's V^T
        comb: fx.Array[fx.Float32, nt * 16 * (KEY_SPLITS - 1) * (2 + HEAD_DIM), 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]

    return Lds


@flyc.jit
def run_kattn(c, o_flag=None):
    """K-attn's tasks on this CTA for ``c["S"]`` rows, ``c["QL"]`` a request, ``c["NT"]``
    row tiles a task. ``o_flag`` (a mailbox region): each merge task raises its
    (request, KV head, row tile)'s flag once its o rows are out (a launch that goes
    on to K2)."""
    S, QL, NT = c["S"], c["QL"], c["NT"]
    tid, bid, lane, wave = c["tid"], c["bid"], c["lane"], c["wave"]
    put, poll, part_mb = c["put"], c["poll"], c["part"]
    a = c["args"]
    R = S // QL
    ROWS = QL * GQA  # rows a (request, KV head): 64 or 8
    ROW_TILES = (ROWS + 15) // 16  # 4 or 1: each its own task (more CTAs busy)
    l16 = lane % 16
    g4 = lane // 16
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    qs = c["qs"]
    pt = c["pt"] + wave * (NT * 16 * P_PITCH)
    vt = c["vt"] + wave * (HEAD_DIM * V_PITCH)
    comb = c["comb"]
    table_stride, extra, window, part_in = a["table_stride"], a["extra"], a["window"], a["part_in"]
    lens64 = a["lens64"]
    qk_scale = a["scale_bits"].bitcast(fx.Float32) * fx.Float32(LOG2E)
    r_k = rsrc(a["k_cache"])
    r_v = rsrc(a["v_cache"])
    r_pt = rsrc(a["page_table"], a["table_bytes"])
    r_q = rsrc(a["q"])

    def seq_len_of(r):
        """Request r's keys (int32, or the low word of its int64), the step's rows included."""
        return (
            fx.Int32(
                bo.buffer_load(
                    rsrc(a["seq_lens"]), (lens64 != 0).select(2 * r, r), vec_width=1, dtype=T.i32
                )
            )
            + extra
        )

    def chunk_range(keys, qpos_lo):
        """The chunks [lo, hi) of a request with ``keys`` keys a window from its
        first row's position leaves unmasked (every chunk of a full one)."""
        hi = (keys + CHUNK - 1) // CHUNK
        ctx = keys - QL
        first = fx.max(ctx + qpos_lo - window + 1, fx.Int32(0))
        lo = (window > 0).select(first // CHUNK, fx.Int32(0))
        return lo, hi

    # the partitions: a request's keys from its first unmasked chunk (a sliding
    # window's start; 0 for a full layer), the longest such span over at most
    # MAX_PARTS of them
    def span_of(r):
        lo_r, hi_r = chunk_range(seq_len_of(r), 0)
        return hi_r - lo_r

    max_span = span_of(fx.Int32(0))
    for rr in range_constexpr(1, R):
        max_span = fx.max(max_span, span_of(fx.Int32(rr)))
    auto = ((max_span * CHUNK + MAX_PARTS * CHUNK - 1) // (MAX_PARTS * CHUNK)) * CHUNK
    part = (part_in > 0).select(part_in, auto)

    def part_index(r, h, p, row):
        """Pair index of (request r, head h, partition p)'s row's record."""
        return (((r * LOCAL_KV_HEADS + h) * MAX_PARTS + p) * ROWS + row) * PART_ROW

    # ============================================ attention tasks
    TASK_TILES = ROW_TILES // NT

    def attn_task(r, h, rt0, p):
        """Row tiles rt0 .. rt0 + NT of (request r, KV head h) over partition p:
        one K / V load and one V^T staging a chunk serve every tile."""
        keys = seq_len_of(r)
        ctx = keys - QL
        ks = wave  # this wave's key split of the task's row tiles
        # Q: rows 16 rt + i = 8 qpos + hg of (request r, head h) into LDS, 16 B a thread
        if tid < NT * 16 * HEAD_DIM // 8:
            i_t = tid // (16 * HEAD_DIM // 8)
            q_row = (tid % (16 * HEAD_DIM // 8)) // (HEAD_DIM // 8)
            row = (rt0 + i_t) * 16 + q_row
            d0 = (tid % (HEAD_DIM // 8)) * 8
            t = r * QL + fx.min(row, ROWS - 1) // GQA
            qv = fx.Vector(
                bo.buffer_load(
                    r_q, (t * Q_K + (h * GQA + fx.min(row, ROWS - 1) % GQA) * HEAD_DIM + d0) // 2,
                    vec_width=4, dtype=T.i32,
                )  # fmt: skip
            )
            fx.ptr_store(qv.bitcast(fx.BFloat16), qs + ((i_t * 16 + q_row) * Q_PITCH + d0))
        gpu.barrier()
        # each tile's A operands of this lane: row l16, dims 8 g4 + 32 step
        qa = [
            [
                fx.Vector(fx.ptr_load(qs + ((it * 16 + l16) * Q_PITCH + 32 * step + 8 * g4), result_type=V8B))  # fmt: skip
                for step in range(2)
            ]
            for it in range(NT)
        ]
        # this lane's rows 4 g4 + e of each tile; their query positions
        qpos = [[fx.min(((rt0 + it) * 16 + 4 * g4 + e) // GQA, QL - 1) for e in range(4)] for it in range(NT)]  # fmt: skip
        live = [[((rt0 + it) * 16 + 4 * g4 + e) < ROWS for e in range(4)] for it in range(NT)]
        # each tile's running row statistics and O, carried through the chunk loop
        m_run = [fx.Vector.filled(4, NEG, fx.Float32) for _ in range(NT)]
        l_run = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(NT)]
        accs = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)] for _ in range(NT)]
        lo, hi = chunk_range(keys, 0)
        c_lo = fx.max(lo, lo + p * (part // CHUNK))
        c_hi = fx.min(hi, lo + (p + 1) * (part // CHUNK))

        # The chunk loop is software-pipelined: chunk cc + 1's K / V loads and
        # chunk cc + 2's page ids are in flight while chunk cc computes (each
        # would otherwise be a dependent round trip: the page id, then the
        # slots). A wave's 32 keys of a chunk are two pages, its two K tiles,
        # and its V passes read the same two pages.
        assert KEY_TILES == 2 and V_STEPS == 1

        def pages_of(cc):
            """The chunk's two page ids, left in VGPRs (every lane loads the same
            one): a readfirstlane here would wait on them, and so on every load
            issued before them, the next chunk's K / V among them."""
            key0 = cc * CHUNK + ks * KEYS_PER_WAVE
            return [
                fx.Int32(bo.buffer_load(r_pt, r * table_stride + (key0 + kt * TILE) // 16, vec_width=1, dtype=T.i32))  # fmt: skip
                for kt in range(KEY_TILES)
            ]

        def kv_of(pg0, pg1):
            """The chunk's K words (tile kt, step) and V words (pass ps) of this lane."""
            kws = []
            for pg in (pg0, pg1):
                slot = pg * 16 + l16  # the tile is one page
                for step in range_constexpr(2):
                    kws.append(
                        fx.Vector(
                            bo.buffer_load(
                                r_k, ((slot * LOCAL_KV_HEADS + h) * HEAD_DIM + 32 * step + 8 * g4) // 4,
                                vec_width=2, dtype=T.i32,
                            )  # fmt: skip
                        )
                    )
            vws = []
            for pg in (pg0, pg1):
                slot = pg * 16 + lane // 4
                vws.append(
                    fx.Vector(
                        bo.buffer_load(
                            r_v, ((slot * LOCAL_KV_HEADS + h) * HEAD_DIM + (lane % 4) * 16) // 4, vec_width=4, dtype=T.i32
                        )  # fmt: skip
                    )
                )
            return kws + vws

        c_last = fx.max(c_hi - 1, c_lo)
        pa0, pa1 = pages_of(c_lo)
        k00, k01, k10, k11, v0, v1 = kv_of(pa0, pa1)
        pb0, pb1 = pages_of(fx.min(c_lo + 1, c_last))
        for cc in range(c_lo, c_hi):
            key0 = cc * CHUNK + ks * KEYS_PER_WAVE
            kw_cur = [k00, k01, k10, k11]
            vw_cur = [v0, v1]
            # the next chunk's words and the one after's pages go out first
            k00, k01, k10, k11, v0, v1 = kv_of(pb0, pb1)
            pb0, pb1 = pages_of(fx.min(cc + 2, c_last))
            # the chunk's K operands, decoded once for every tile
            kb = [fp8_to_bf16x8(kw[0], kw[1]) for kw in kw_cur]
            m_next, l_next, acc_scaled = [], [], []
            for it in range_constexpr(NT):
                # ---- QK: S tile kt, key l16 of it; dims 8 g4 + 32 step
                s_t = []
                for kt in range_constexpr(KEY_TILES):
                    st = fx.Vector.filled(4, 0.0, fx.Float32)
                    for step in range_constexpr(2):
                        st = mfma_bf16(qa[it][step], kb[kt * 2 + step], st)
                    s_t.append(st)
                # ---- scale and mask: lane's rows 4 g4 + e, key l16 of tile kt
                sc = []
                for kt in range_constexpr(KEY_TILES):
                    kidx = key0 + kt * TILE + l16
                    vals = []
                    for e in range_constexpr(4):
                        ok = live[it][e] & (kidx < ctx + qpos[it][e] + 1)
                        ok = ok & ((window == 0) | ((ctx + qpos[it][e] - kidx) < window))
                        vals.append(ok.select(fx.Float32(s_t[kt][e]) * qk_scale, fx.Float32(NEG)))
                    sc.append(vals)
                # ---- row max over the wave's keys (16 lanes x KEY_TILES); a row
                # without a key yet takes 0 (the verify kernels' guard)
                m_new, m_g = [], []
                for e in range_constexpr(4):
                    mx = sc[0][e]
                    for kt in range_constexpr(1, KEY_TILES):
                        mx = fx.max(mx, sc[kt][e])
                    mx = butterfly(mx, (1, 2, 4, 8), fx.max)
                    mn = fx.max(fx.Float32(m_run[it][e]), mx)
                    m_new.append(mn)
                    m_g.append((mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0)))
                # ---- P = exp2(S - m) -> LDS (bf16), its row sums, the rescale
                l_c = [fx.Float32(0.0) for _ in range(4)]
                for kt in range_constexpr(KEY_TILES):
                    for e in range_constexpr(4):
                        pe = hw_exp2(sc[kt][e] - m_g[e])
                        l_c[e] = l_c[e] + pe
                        fx.ptr_store(
                            pe.to(fx.BFloat16),
                            pt + ((it * 16 + 4 * g4 + e) * P_PITCH + kt * TILE + l16),
                        )
                alphas = []
                for e in range_constexpr(4):
                    l_c[e] = butterfly(l_c[e], (1, 2, 4, 8))
                    alphas.append(hw_exp2(fx.Float32(m_run[it][e]) - m_g[e]))
                l_next.append(
                    fx.Vector.from_elements([fx.Float32(l_run[it][e]) * alphas[e] + l_c[e] for e in range(4)], fx.Float32)  # fmt: skip
                )
                m_next.append(fx.Vector.from_elements(m_new, fx.Float32))
                al = fx.Vector.from_elements(alphas, fx.Float32)
                acc_scaled.append([accs[it][dt] * al for dt in range(4)])
            m_run = m_next
            l_run = l_next
            accs = acc_scaled
            # ---- PV over 32-key steps: V^T staged in LDS once, 16 keys a pass
            for vs in range_constexpr(V_STEPS):
                for ps in range_constexpr(2):
                    d0 = (lane % 4) * 16
                    vw = vw_cur[ps]  # keys key0 + 16 ps + lane / 4, dims d0 ..
                    vb_lo = fp8_to_bf16x8(vw[0], vw[1])  # dims d0 .. d0 + 8 as bf16
                    vb_hi = fp8_to_bf16x8(vw[2], vw[3])
                    for j in range_constexpr(8):
                        fx.ptr_store(vb_lo[j], vt + ((d0 + j) * V_PITCH + ps * 16 + lane // 4))
                        fx.ptr_store(vb_hi[j], vt + ((d0 + 8 + j) * V_PITCH + ps * 16 + lane // 4))
                rocdl.s_waitcnt(lgkmcnt=0)
                b_v = [
                    fx.Vector(fx.ptr_load(vt + ((dt * 16 + l16) * V_PITCH + 8 * g4), result_type=V8B))
                    for dt in range(4)
                ]
                acc_pv = []
                for it in range_constexpr(NT):
                    a_p = fx.Vector(
                        fx.ptr_load(pt + ((it * 16 + l16) * P_PITCH + vs * 32 + 8 * g4), result_type=V8B)
                    )
                    acc_pv.append([mfma_bf16(a_p, b_v[dt], accs[it][dt]) for dt in range(4)])
                accs = acc_pv
                rocdl.s_waitcnt(lgkmcnt=0)  # the next pass rewrites vt
        # ---- the key splits of each row tile merge in LDS: split 0 collects
        for it in range_constexpr(NT):
            base = (it * (KEY_SPLITS - 1) + ks - 1) * 16 * (2 + HEAD_DIM)
            if ks > 0:
                for e in range_constexpr(4):
                    row = 4 * g4 + e
                    if l16 == 0:
                        fx.ptr_store(fx.Float32(m_run[it][e]), comb + (base + row * (2 + HEAD_DIM)))
                        fx.ptr_store(fx.Float32(l_run[it][e]), comb + (base + row * (2 + HEAD_DIM) + 1))
                    for dt in range_constexpr(4):
                        fx.ptr_store(fx.Float32(accs[it][dt][e]), comb + (base + row * (2 + HEAD_DIM) + 2 + dt * 16 + l16))  # fmt: skip
        gpu.barrier()
        if ks == 0:
            for it in range_constexpr(NT):
                m_f = [fx.Float32(m_run[it][e]) for e in range(4)]
                l_f = [fx.Float32(l_run[it][e]) for e in range(4)]
                o_f = [[fx.Float32(accs[it][dt][e]) for e in range(4)] for dt in range(4)]
                for src in range_constexpr(1, KEY_SPLITS):
                    sb = (it * (KEY_SPLITS - 1) + src - 1) * 16 * (2 + HEAD_DIM)
                    for e in range_constexpr(4):
                        row = 4 * g4 + e
                        m2 = fx.ptr_load(comb + (sb + row * (2 + HEAD_DIM)))
                        l2 = fx.ptr_load(comb + (sb + row * (2 + HEAD_DIM) + 1))
                        m = fx.max(m_f[e], m2)
                        mg = (m > fx.Float32(NO_MAX)).select(m, fx.Float32(0.0))
                        a1 = hw_exp2(m_f[e] - mg)
                        a2 = hw_exp2(m2 - mg)
                        l_f[e] = l_f[e] * a1 + l2 * a2
                        m_f[e] = m
                        for dt in range_constexpr(4):
                            v2 = fx.ptr_load(comb + (sb + row * (2 + HEAD_DIM) + 2 + dt * 16 + l16))
                            o_f[dt][e] = o_f[dt][e] * a1 + v2 * a2
                # publish the tile's partial: a row's (m, l, O[64]) as fp32 pairs
                for e in range_constexpr(4):
                    row = (rt0 + it) * 16 + 4 * g4 + e
                    if row < ROWS:
                        pi = part_index(r, h, p, row)
                        if l16 == 0:
                            put(part_mb, pi, m_f[e])
                            put(part_mb, pi + 1, l_f[e])
                        for dt in range_constexpr(4):
                            put(part_mb, pi + 2 + dt * 16 + l16, o_f[dt][e])
        gpu.barrier()

    stamp(c, 0)
    n_parts = (max_span * CHUNK + part - 1) // part
    for job in range(bid, R * LOCAL_KV_HEADS * TASK_TILES * n_parts, BLOCKS):
        p = job % n_parts
        rt0 = ((job // n_parts) % TASK_TILES) * NT
        h = (job // (n_parts * TASK_TILES)) % LOCAL_KV_HEADS
        r = job // (n_parts * TASK_TILES * LOCAL_KV_HEADS)
        keys = seq_len_of(r)
        lo, hi = chunk_range(keys, 0)
        c_lo = fx.max(lo, lo + p * (part // CHUNK))
        c_hi = fx.min(hi, lo + (p + 1) * (part // CHUNK))
        if c_lo < c_hi:
            attn_task(r, h, rt0, p)

    stamp(c, 1)  # the attention tasks done
    # ============================================ merge tasks: (request, head, row tile)
    # a thread: row 16 rt + tid / 32 (clamped past ROWS: its output is not
    # written), dims 8 ((tid / 4) % 8) .., every fourth partition from tid % 4,
    # two a spin; the four partition groups' partials combine in LDS
    mbid = BLOCKS - 1 - bid  # the last CTAs: the attention task loop's extra tasks sit on the first
    if mbid < R * LOCAL_KV_HEADS * ROW_TILES:
        rt = mbid % ROW_TILES
        h = (mbid // ROW_TILES) % LOCAL_KV_HEADS
        r = mbid // (ROW_TILES * LOCAL_KV_HEADS)
        keys = seq_len_of(r)
        lo, hi = chunk_range(keys, 0)
        p_lo = fx.Int32(0)
        p_hi = ((hi - lo) * CHUNK + part - 1) // part
        row16 = tid // 32
        dch = (tid // 4) % 8
        pg = tid % 4
        d0 = dch * 8
        row = fx.min(rt * 16 + row16, ROWS - 1)
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        ov = fx.Vector.filled(8, 0.0, fx.Float32)

        def specs(p):
            pi = part_index(r, h, p, row)
            # a poll spec is at most two pairs (a 16 B load)
            return [(part_mb, pi, 2)] + [(part_mb, pi + 2 + d0 + 2 * j, 2) for j in range(4)]

        def fold(m, l, ov, m2, l2, v2, ok):
            """(m, l, O) with a partial folded in (``ok``: it exists; a clamped
            duplicate changes nothing)."""
            mn = fx.max(m, m2)
            mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
            a1 = ok.select(hw_exp2(m - mg), fx.Float32(1.0))
            a2 = ok.select(hw_exp2(m2 - mg), fx.Float32(0.0))
            return ok.select(mn, m), l * a1 + l2 * a2, ov * a1 + v2 * a2

        def words(g):
            return (
                g[0][0].bitcast(fx.Float32),
                g[0][1].bitcast(fx.Float32),
                fx.Vector.from_elements([g[1 + j // 2][j % 2].bitcast(fx.Float32) for j in range(8)], fx.Float32),  # fmt: skip
            )

        # four partitions a thread a spin (16 a spin over the four groups), one round trip
        for gi in range(0, (p_hi - p_lo + 15) // 16):
            ps = [p_lo + gi * 16 + pg + 4 * k for k in range(4)]
            got = poll(sum([specs(fx.min(pk, p_hi - 1)) for pk in ps], []), batch=20)
            for k in range_constexpr(4):
                m, l, ov = fold(m, l, ov, *words(got[5 * k : 5 * k + 5]), ps[k] < p_hi)
        slot = comb + (((row16 * 8 + dch) * 4 + pg) * 10)
        fx.ptr_store(m, slot)
        fx.ptr_store(l, slot + 1)
        for j in range_constexpr(8):
            fx.ptr_store(fx.Float32(ov[j]), slot + (2 + j))
        gpu.barrier()
        if (pg == 0) & (rt * 16 + row16 < ROWS):
            for src in range_constexpr(1, 4):
                m2 = fx.ptr_load(slot + src * 10)
                l2 = fx.ptr_load(slot + (src * 10 + 1))
                v2 = fx.Vector.from_elements([fx.ptr_load(slot + (src * 10 + 2 + j)) for j in range(8)], fx.Float32)  # fmt: skip
                m, l, ov = fold(m, l, ov, m2, l2, v2, fx.Int32(1) == fx.Int32(1))
            inv = fx.Float32(1.0) / l
            t = r * QL + (rt * 16 + row16) // GQA
            bo.buffer_store(
                (ov * inv).to(fx.BFloat16),
                rsrc(a["o"]),
                t * Q_K + (h * GQA + (rt * 16 + row16) % GQA) * HEAD_DIM + d0,
                cache_modifier=CM_DEV,
            )
        if const_expr(o_flag is not None):
            # the (request, head, row tile)'s o rows are out: its flag, for the og tasks
            release_stores()
            gpu.barrier()
            if tid == 0:
                put(o_flag, (r * LOCAL_KV_HEADS + h) * ROW_TILES + rt, fx.Int32(1))
    stamp(c, 2)  # the merge done (end)
