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

Tasks. An attention task is one (request, KV head, ``nt`` row tiles, partition
of ``part`` keys; the partitions a sliding window masks whole are skipped): a
chunk's K / V loads and its V^T staging serve the task's tiles (QL = 8 is 64
rows = four tiles; two a task fit the registers, four spill). Each is a loop over 256-key chunks
with the waves splitting the keys: QK on wave-local key tiles, the row
statistics, P through LDS, V transposed through LDS, PV; the waves' key splits
merge in LDS as a tree (8 -> 4 -> 2 -> 1); the partial (m, l, O) a row goes out
as plain fp32 words with one flag a task. The partitions are sized to at most
``MAX_PARTS`` a request from the longest request's span (``part_in`` overrides),
so one 95k-key request's full layer is 128 tasks of three chunks a KV head and a
sliding layer eight tasks of one; the partition grows with the step's total
chunk work so a step stays near two equal tasks a CTA whatever the request mix.

Merge. A (request, KV head, row tile) with at most ``DIRECT_MAX`` partitions
folds them in one job; a longer one in two levels: ``MERGE_GROUPS`` level-1 jobs
each fold a contiguous eighth of the partitions (a thread a (row, 8 dims,
partition lane), the lanes combined in LDS) into a level-2 partial, and the
level-2 job folds those. The merge jobs sit on the last CTAs (the task loop's
extra tasks fall on the first); a level-2 job's CTA runs its level-1 job first,
so every wait points to work that is already running.
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
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES
from sgl_amd_mono.muse_spark.gfx950.config import HEAD_DIM, LOCAL_KV_HEADS, LOG2E, Q_K
from sgl_amd_mono.muse_spark.gfx950.device import acquire_loads, poll_flags, release_stores, stamp
from sgl_amd_mono.muse_spark.gfx950.layout import (
    ATTN_CHUNK as CHUNK,
    ATTN_KEY_SPLITS as KEY_SPLITS,
    ATTN_KEY_TILES as KEY_TILES,
    ATTN_KEYS_PER_WAVE as KEYS_PER_WAVE,
    ATTN_MAX_PARTS as MAX_PARTS,
    ATTN_MERGE_GROUPS as GROUPS,
    ATTN_P_PITCH as P_PITCH,
    ATTN_PART_ROW as PART_ROW,
    ATTN_Q_PITCH as Q_PITCH,
    ATTN_V_PITCH as V_PITCH,
    ATTN_V_STEPS as V_STEPS,
    GQA,
    attn_rows,
)

TILE = 16  # keys an MFMA tile
NEG = -1.0e30  # a masked score, and "no key yet" of a row's running max
NO_MAX = -1.0e29  # a running max below this is "no key yet": exp2 then takes 0
O_OFF = 4  # a partial row: m, l, pad, pad, O[64] (O 16 B aligned)
COMB_ROW = PART_ROW  # a tile row's (m, l, pad, pad, O[64]) in the LDS tree
MERGE_LANES = 4  # partitions folded side by side a (row, 8 dims): 16 x 8 x 4 = THREADS
DIRECT_MAX = 16  # partitions a level-2 job folds itself (four a lane); more go through level 1


def comb_floats(nt: int) -> int:
    """The key-split merge's LDS floats: every wave's state of ``nt`` tiles."""
    return WAVES * nt * 16 * COMB_ROW


def kattn_lds(nt: int):
    """K-attn's LDS as two aliasing views (a union of them, built by the launch): a
    task's Q tiles and the merge jobs' reduction, then either each wave's P tiles
    and V^T (the chunk loop) or the key splits' merge tree (after it)."""

    @fx.struct
    class Loop:
        qs: fx.Array[fx.BFloat16, nt * 16 * Q_PITCH, 16]  # the task's Q rows (its row tiles)
        red: fx.Array[fx.Float32, THREADS * 10, 16]  # the merge jobs' lane combine
        pt: fx.Array[fx.BFloat16, WAVES * nt * 16 * P_PITCH, 16]  # each wave's P tiles
        vt: fx.Array[fx.BFloat16, WAVES * HEAD_DIM * V_PITCH, 16]  # each wave's V^T

    @fx.struct
    class Tree:
        qs: fx.Array[fx.BFloat16, nt * 16 * Q_PITCH, 16]
        red: fx.Array[fx.Float32, THREADS * 10, 16]
        comb: fx.Array[fx.Float32, comb_floats(nt), 16]

    return Loop, Tree


def lds_views(loop, tree):
    """``c`` entries of the two ``kattn_lds`` views (``.peek()`` of union members)."""
    return {"qs": loop.qs.ptr, "red": loop.red.ptr, "pt": loop.pt.ptr, "vt": loop.vt.ptr, "comb": tree.comb.ptr}


@flyc.jit
def run_kattn(c, o_flag=None):
    """K-attn's work on this CTA for ``c["S"]`` rows, ``c["QL"]`` a request.
    ``o_flag`` (a mailbox region): each level-2 merge raises its (request, KV
    head, row tile)'s flag once its o rows are out (a launch that goes on to K2)."""
    S, QL = c["S"], c["QL"]
    tid, bid, lane, wave = c["tid"], c["bid"], c["lane"], c["wave"]
    put, poll = c["put"], c["poll"]
    part_w, pflag, l2_w, l2flag = c["part"], c["pflag"], c["l2"], c["l2flag"]
    a = c["args"]
    R = S // QL
    ROWS = QL * GQA  # rows a (request, KV head): 64 or 8
    ROW_TILES = (ROWS + 15) // 16  # 4 or 1
    NT = c["NT"]  # row tiles a task (the build's; 4 spills registers, 2 fits)
    TASK_TILES = ROW_TILES // NT
    l16 = lane % 16
    g4 = lane // 16
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    V4F = fx.Vector.make_type(4, fx.Float32)
    qs, pt0, vt0, comb, red = c["qs"], c["pt"], c["vt"], c["comb"], c["red"]
    pt = pt0 + wave * (NT * 16 * P_PITCH)
    vt = vt0 + wave * (HEAD_DIM * V_PITCH)
    table_stride, extra, window, part_in = a["table_stride"], a["extra"], a["window"], a["part_in"]
    lens64 = a["lens64"]
    qk_scale = a["scale_bits"].bitcast(fx.Float32) * fx.Float32(LOG2E)
    r_k = rsrc(a["k_cache"])
    r_v = rsrc(a["v_cache"])
    r_pt = rsrc(a["page_table"], a["table_bytes"])
    r_q = rsrc(a["q"])
    r_part = rsrc(part_w.value)
    r_l2 = rsrc(l2_w.value)
    stamp(c, 0)

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
    total_span = max_span
    for rr in range_constexpr(1, R):
        sp = span_of(fx.Int32(rr))
        max_span = fx.max(max_span, sp)
        total_span = total_span + sp
    # the partition (chunks a task) from the step's total chunk work, so that the
    # step has about two equal tasks a CTA whatever the request mix (a task costs
    # ~6 us beside its chunks: eight 95k requests at 128 partitions each were 15
    # tasks a CTA, 212 us; one partition a request for all would starve the CTAs
    # on a mixed step), and at least what keeps the longest request within MAX_PARTS
    per_cta = (total_span * (LOCAL_KV_HEADS * TASK_TILES) + 2 * BLOCKS - 1) // (2 * BLOCKS)
    floor_ = (max_span + MAX_PARTS - 1) // MAX_PARTS
    pc = fx.max(fx.max(per_cta, floor_), fx.Int32(1))
    part = (part_in > 0).select(part_in, pc * CHUNK)
    n_parts = (max_span * CHUNK + part - 1) // part  # partitions a (request, head) at most

    def n_parts_of(r):
        """Request r's partitions (its span's, the rest of the task slots idle)."""
        lo, hi = chunk_range(seq_len_of(r), 0)
        return ((hi - lo) * CHUNK + part - 1) // part

    def part_row(r, h, p, row):
        """Float index of (request r, head h, partition p)'s row record."""
        return (((r * LOCAL_KV_HEADS + h) * MAX_PARTS + p) * ROWS + row) * PART_ROW

    def l2_row(r, h, rt, g, row16):
        """Float index of (request r, head h, row tile rt, group g)'s level-2 row record."""
        return ((((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS + g) * 16 + row16) * PART_ROW

    def fold(m, l, ov, m2, l2, v2, ok):
        """(m, l, O) with a partial folded in (``ok``: it exists)."""
        mn = fx.max(m, m2)
        mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
        a1 = ok.select(hw_exp2(m - mg), fx.Float32(1.0))
        a2 = ok.select(hw_exp2(m2 - mg), fx.Float32(0.0))
        return ok.select(mn, m), l * a1 + l2 * a2, ov * a1 + v2 * a2

    # ============================================ attention tasks
    def attn_task(r, h, rt0, p):
        """Row tiles rt0 .. rt0 + NT of (request r, KV head h) over partition p: one
        K / V load and one V^T staging a chunk serve every tile of the task."""
        keys = seq_len_of(r)
        ctx = keys - QL
        ks = wave  # this wave's key split
        # Q: rows 16 (rt0 + it) + i = 8 qpos + hg of (request r, head h) into LDS, 16 B a thread
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
            """The chunk's two page ids, left in VGPRs (every lane loads the same one)."""
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
        gpu.barrier()  # every wave is done with pt / vt: the merge reuses them

        # ---- the key splits merge in LDS in one round: every wave writes its
        # (m, l, O) rows (a (split, tile, row) record is COMB_ROW floats, O from
        # O_OFF); then each wave owns 2 NT of the task's rows, folds the eight splits
        # of them (a lane: one row, DPL dims) and publishes them with vector stores
        def comb_at(slot, it, row):
            return comb + ((slot * NT + it) * 16 + row) * COMB_ROW

        for it in range_constexpr(NT):
            for e in range_constexpr(4):
                row = 4 * g4 + e
                base = comb_at(ks, it, row)
                if l16 == 0:
                    fx.ptr_store(fx.Float32(m_run[it][e]), base)
                    fx.ptr_store(fx.Float32(l_run[it][e]), base + 1)
                for dt in range_constexpr(4):
                    fx.ptr_store(fx.Float32(accs[it][dt][e]), base + (O_OFF + dt * 16 + l16))
        gpu.barrier()
        RPW = 2 * NT  # rows a wave: NT 16 / 8
        LPR = 64 // RPW  # lanes a row: 16 or 32
        DPL = HEAD_DIM // LPR  # dims a lane: 4 or 2
        VD = fx.Vector.make_type(DPL, fx.Float32)
        trow = wave * RPW + lane // LPR  # the task row (0 .. NT 16)
        it_f, row_f = trow // 16, trow % 16
        dd = (lane % LPR) * DPL
        m_f = fx.Float32(NEG)
        l_f = fx.Float32(0.0)
        o_f = fx.Vector.filled(DPL, 0.0, fx.Float32)
        for src in range_constexpr(KEY_SPLITS):
            base = comb_at(src, it_f, row_f)
            m2 = fx.ptr_load(base)
            l2 = fx.ptr_load(base + 1)
            v2 = fx.Vector(fx.ptr_load(base + (O_OFF + dd), result_type=VD))
            mn = fx.max(m_f, m2)
            mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
            a1 = hw_exp2(m_f - mg)
            a2 = hw_exp2(m2 - mg)
            m_f, l_f, o_f = mn, l_f * a1 + l2 * a2, o_f * a1 + v2 * a2
        # ---- the task's partial rows: plain fp32 words, then the flag
        row = rt0 * 16 + trow
        if row < ROWS:
            base = part_row(r, h, p, row)
            if lane % LPR == 0:
                bo.buffer_store(fx.Vector.from_elements([m_f, l_f], fx.Float32), r_part, base, cache_modifier=CM_DEV)
            bo.buffer_store(o_f, r_part, base + O_OFF + dd, cache_modifier=CM_DEV)
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(pflag, (r * LOCAL_KV_HEADS + h) * MAX_PARTS + p, fx.Int32(1))

    # the tasks, request-major over the requests' own partition counts (a short
    # request has few): job -> (request, head, tile pair, partition) through the
    # requests' cumulative task counts, so a CTA iterates real tasks only
    counts = [n_parts_of(fx.Int32(rr)) * (LOCAL_KV_HEADS * TASK_TILES) for rr in range(R)]
    starts = [fx.Int32(0)]
    for rr in range_constexpr(1, R):
        starts.append(starts[-1] + counts[rr - 1])
    total = starts[-1] + counts[-1]
    for job in range(bid, total, BLOCKS):
        r = fx.Int32(0)
        for rr in range_constexpr(1, R):
            r = (job >= starts[rr]).select(fx.Int32(rr), r)
        base = starts[0]
        for rr in range_constexpr(1, R):
            base = (r == rr).select(starts[rr], base)
        np_r = n_parts_of(r)
        local = job - base
        p = local % np_r
        rt0 = ((local // np_r) % TASK_TILES) * NT
        h = (local // (np_r * TASK_TILES)) % LOCAL_KV_HEADS
        attn_task(r, h, rt0, p)
    stamp(c, 1)  # the attention tasks done

    # ============================================ merge jobs
    # a thread: row 16 rt + tid / 32, dims 8 ((tid / 4) % 8) .., partition lane tid % 4
    row16 = tid // 32
    dch = (tid // 4) % 8
    pl = tid % 4
    d0 = dch * 8
    mbid = BLOCKS - 1 - bid  # the last CTAs: the task loop's extra tasks sit on the first
    n_l1 = R * LOCAL_KV_HEADS * ROW_TILES * GROUPS
    n_l2 = R * LOCAL_KV_HEADS * ROW_TILES

    def load_rec(r_src, base):
        """A record's (m, l) and this thread's 8 dims of O (plain words)."""
        ml = fx.Vector(bo.buffer_load(r_src, base, vec_width=2, dtype=T.f32, cache_modifier=CM_DEV))
        o0 = fx.Vector(bo.buffer_load(r_src, base + O_OFF + d0, vec_width=4, dtype=T.f32, cache_modifier=CM_DEV))
        o1 = fx.Vector(bo.buffer_load(r_src, base + O_OFF + d0 + 4, vec_width=4, dtype=T.f32, cache_modifier=CM_DEV))
        return ml, o0, o1

    def fold_rec(m, l, o0, o1, rec, ok):
        ml, p0, p1 = rec
        m2, l2 = fx.Float32(ml[0]), fx.Float32(ml[1])
        mn = fx.max(m, m2)
        mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
        a1 = ok.select(hw_exp2(m - mg), fx.Float32(1.0))
        a2 = ok.select(hw_exp2(m2 - mg), fx.Float32(0.0))
        return ok.select(mn, m), l * a1 + l2 * a2, o0 * a1 + p0 * a2, o1 * a1 + p1 * a2

    def combine_lanes(m, l, o0, o1):
        """The four partition lanes of a (row, 8 dims) fold in LDS: every lane folds
        the four in lane order (the same result on each; lane 0 stores)."""
        slot = red + tid * 10
        fx.ptr_store(m, slot)
        fx.ptr_store(l, slot + 1)
        for j in range_constexpr(4):
            fx.ptr_store(fx.Float32(o0[j]), slot + (2 + j))
            fx.ptr_store(fx.Float32(o1[j]), slot + (6 + j))
        gpu.barrier()
        base = red + (tid - pl) * 10
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        for src in range_constexpr(MERGE_LANES):
            s2 = base + src * 10
            m2 = fx.ptr_load(s2)
            l2 = fx.ptr_load(s2 + 1)
            p0 = fx.Vector.from_elements([fx.ptr_load(s2 + (2 + j)) for j in range(4)], fx.Float32)
            p1 = fx.Vector.from_elements([fx.ptr_load(s2 + (6 + j)) for j in range(4)], fx.Float32)
            mn = fx.max(m, m2)
            mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
            a1 = hw_exp2(m - mg)
            a2 = hw_exp2(m2 - mg)
            m, l, o0, o1 = mn, l * a1 + l2 * a2, o0 * a1 + p0 * a2, o1 * a1 + p1 * a2
        gpu.barrier()  # ``red`` is reused by the CTA's next job
        return m, l, o0, o1

    def merge_l1(r, h, rt, g):
        """Level-1 job: partitions [g PPG, (g + 1) PPG) of (request r, head h) for
        row tile rt, folded four lanes a (row, 8 dims) -> a level-2 record."""
        np_r = n_parts_of(r)
        ppg = (np_r + GROUPS - 1) // GROUPS
        p_lo = g * ppg
        cnt = fx.max(fx.min(np_r - p_lo, ppg), fx.Int32(0))
        poll_flags(c, [(pflag, (r * LOCAL_KV_HEADS + h) * MAX_PARTS + p_lo, cnt)])
        row = fx.min(rt * 16 + row16, ROWS - 1)
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        # four partitions a lane in flight a round
        for k0 in range(0, (ppg + MERGE_LANES * 4 - 1) // (MERGE_LANES * 4)):
            ps = [p_lo + (k0 * 4 + k) * MERGE_LANES + pl for k in range(4)]
            recs = [load_rec(r_part, part_row(r, h, fx.min(pk, p_lo + cnt - 1), row)) for pk in ps]
            for k in range_constexpr(4):
                m, l, o0, o1 = fold_rec(m, l, o0, o1, recs[k], ps[k] < p_lo + cnt)
        m, l, o0, o1 = combine_lanes(m, l, o0, o1)
        if (pl == 0) & (rt * 16 + row16 < ROWS):
            base = l2_row(r, h, rt, g, row16)
            if dch == 0:
                bo.buffer_store(fx.Vector.from_elements([m, l], fx.Float32), r_l2, base, cache_modifier=CM_DEV)
            bo.buffer_store(o0, r_l2, base + O_OFF + d0, cache_modifier=CM_DEV)
            bo.buffer_store(o1, r_l2, base + O_OFF + d0 + 4, cache_modifier=CM_DEV)
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(l2flag, ((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS + g, fx.Int32(1))

    def merge_l2(r, h, rt):
        """Level-2 job: the groups' records (or, with at most DIRECT_MAX partitions,
        the partitions themselves) of (request r, head h, row tile rt) -> o rows."""
        np_r = n_parts_of(r)
        direct = np_r <= DIRECT_MAX
        row = fx.min(rt * 16 + row16, ROWS - 1)
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        # the source records: the partitions themselves (at most GROUPS of them), or
        # the level-1 records; one resource over the selected base address
        if direct:
            poll_flags(c, [(pflag, (r * LOCAL_KV_HEADS + h) * MAX_PARTS, np_r)])
        else:
            poll_flags(c, [(l2flag, ((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS, GROUPS)])
        r_src = rsrc(direct.select(part_w.value, l2_w.value))
        for k in range_constexpr(DIRECT_MAX // MERGE_LANES):
            g = k * MERGE_LANES + pl  # a partition (direct) or a group (g < GROUPS)
            idx = direct.select(part_row(r, h, fx.min(g, np_r - 1), row), l2_row(r, h, rt, fx.min(g, GROUPS - 1), row16))
            ok = direct.select(g < np_r, g < GROUPS)
            m, l, o0, o1 = fold_rec(m, l, o0, o1, load_rec(r_src, idx), ok)
        m, l, o0, o1 = combine_lanes(m, l, o0, o1)
        if (pl == 0) & (rt * 16 + row16 < ROWS):
            inv = fx.Float32(1.0) / l
            t = r * QL + (rt * 16 + row16) // GQA
            ov = fx.Vector.from_elements([fx.Float32(o0[j]) for j in range(4)] + [fx.Float32(o1[j]) for j in range(4)], fx.Float32)  # fmt: skip
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

    for j in range(mbid, n_l1, BLOCKS):
        g = j % GROUPS
        rt = (j // GROUPS) % ROW_TILES
        h = (j // (GROUPS * ROW_TILES)) % LOCAL_KV_HEADS
        r = j // (GROUPS * ROW_TILES * LOCAL_KV_HEADS)
        if n_parts_of(r) > DIRECT_MAX:  # else the level-2 job folds the partitions itself
            merge_l1(r, h, rt, g)
    if mbid < n_l2:
        rt = mbid % ROW_TILES
        h = (mbid // ROW_TILES) % LOCAL_KV_HEADS
        r = mbid // (ROW_TILES * LOCAL_KV_HEADS)
        merge_l2(r, h, rt)
    stamp(c, 2)  # the merge done (end)
