# SPDX-License-Identifier: Apache-2.0
"""Build the Muse Spark launches: K1 (``build_k1``) and K-attn (``build_kattn``) standalone for now."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.plan import BLOCKS, THREADS, kernel_sources, key_tuple
from sgl_amd_mono.common.sync import Mailbox, sreg
from sgl_amd_mono.muse_spark.gfx950.config import MAX_TOKENS
from sgl_amd_mono.muse_spark.gfx950.device import step_tag
from sgl_amd_mono.muse_spark.gfx950.layout import (
    XS_LDS_MAX,
    attn_rows,
    k1_scratch_layout,
    kattn_scratch_layout,
)
from sgl_amd_mono.muse_spark.gfx950.stages import attention, pre_attn

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)


class K1Build(msgspec.Struct, frozen=True):
    """A K1 build: the step's row count and the fault injection site
    (SGLANG_AMD_MONO_FAULT_INJECT: the wait this rank stalls at, once
    ``fault_after`` launches have run)."""

    tokens: int
    fault_site: int = 0
    fault_after: int = 0


def k1_context(key: K1Build, lds, args: dict, scratch, tag, layout: dict) -> dict:
    """K1's context: LDS views, the mailbox, the scratch regions and arguments."""
    s = key.tokens
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    c = {
        "S": s,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": lds.red.ptr,
        "xs": lds.xs.ptr if s <= XS_LDS_MAX else None,
        "put": mb.put,
        "poll": mb.poll,
        "args": args,
        "x_cm": 0,  # x from memory the host wrote: the default cache policy
    }
    for region, (off, _) in layout.items():
        c[region[3:]] = sreg(scratch, off, region)  # "k1_part" -> c["part"]
    return c


def build_k1(key: K1Build):
    """``@flyc.jit`` launcher of K1 for a step of ``key.tokens`` rows."""
    tokens = key.tokens
    assert 1 <= tokens <= MAX_TOKENS
    name = kernel_symbol("muse_spark_k1", s=tokens, f=key.fault_site)
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout = k1_scratch_layout()
    Lds = pre_attn.k1_lds(tokens)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        x: Int64,
        w: Int64,
        cos_sin: Int64,
        positions: Int64,
        slots: Int64,
        swa_map: Int64,
        k_cache: Int64,
        v_cache: Int64,
        q_out: Int64,
        k_out: Int64,
        v_out: Int64,
        g_out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        use_rope: Int32,
        use_swa: Int32,
        slot64: Int32,
        swa64: Int32,
    ):
        # every wait runs under the guard (deadline, error words); K1 exchanges
        # nothing with its peers, so a failure stays on this GPU
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "x": x, "w": w, "cos_sin": cos_sin, "positions": positions, "slots": slots,
                "swa_map": swa_map, "k_cache": k_cache, "v_cache": v_cache, "q_out": q_out,
                "k_out": k_out, "v_out": v_out, "g_out": g_out, "use_rope": use_rope,
                "use_swa": use_swa, "slot64": slot64, "swa64": swa64,
            }  # fmt: skip
            c = k1_context(key, lds, args, scratch, step_tag(epoch, layer), layout)
            pre_attn.stage_x(c)
            pre_attn.run_k1(c, c["bid"])

    @flyc.jit
    def launch(
        x: Int64,
        w: Int64,
        cos_sin: Int64,
        positions: Int64,
        slots: Int64,
        swa_map: Int64,
        k_cache: Int64,
        v_cache: Int64,
        q_out: Int64,
        k_out: Int64,
        v_out: Int64,
        g_out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        use_rope: Int32,
        use_swa: Int32,
        slot64: Int32,
        swa64: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            x, w, cos_sin, positions, slots, swa_map, k_cache, v_cache, q_out, k_out,
            v_out, g_out, scratch, epoch, layer, use_rope, use_swa, slot64, swa64,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch


class KAttnBuild(msgspec.Struct, frozen=True):
    """A K-attn build: the step's row count and rows a request, the row tiles a
    task shares its K / V loads over (1; 2 won +8 % on full layers and lost 48 % on
    sliding ones in the old branch), and the fault injection site."""

    tokens: int
    q_len: int = 8
    nt: int = 1
    fault_site: int = 0
    fault_after: int = 0


def kattn_context(key: KAttnBuild, lds, args: dict, scratch, tag, layout: dict) -> dict:
    """K-attn's context: LDS views, the mailbox, the scratch region and arguments."""
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    c = {
        "S": key.tokens,
        "QL": key.q_len,
        "NT": key.nt,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "qs": lds.qs.ptr,
        "pt": lds.pt.ptr,
        "vt": lds.vt.ptr,
        "comb": lds.comb.ptr,
        "red": lds.red.ptr,
        "put": mb.put,
        "poll": mb.poll,
        "args": args,
    }
    c["part"] = sreg(scratch, layout["attn_part"][0], "attn_part")
    return c


def build_kattn(key: KAttnBuild):
    """``@flyc.jit`` launcher of K-attn for ``key.tokens`` rows, ``key.q_len`` a request."""
    S, QL, NT = key.tokens, key.q_len, key.nt
    assert QL in (1, 8) and S % QL == 0 and 1 <= S <= MAX_TOKENS
    row_tiles = (attn_rows(QL) + 15) // 16
    assert row_tiles % NT == 0, (row_tiles, NT)
    name = kernel_symbol("muse_spark_kattn", s=S, q=QL, n=NT, f=key.fault_site)
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout = kattn_scratch_layout(S, QL)
    Lds = attention.kattn_lds(NT)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kattn(
        q: Int64,
        o: Int64,
        k_cache: Int64,
        v_cache: Int64,
        page_table: Int64,
        table_stride: Int32,  # int32 entries a request's row of the table
        table_bytes: Int32,  # the table's size: a key past the row reads page 0
        seq_lens: Int64,  # int32 / int64: the request's keys before this step's rows
        lens64: Int32,  # seq_lens is int64
        extra: Int32,  # keys past seq_lens (the step's rows a verify adds: QL; 0 for decode)
        window: Int32,  # the sliding window (keys), 0 for full attention
        part_in: Int32,  # keys a partition (a multiple of CHUNK); 0: sized here
        scale_bits: Int32,  # the softmax scale, f32 bits
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
    ):
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "q": q, "o": o, "k_cache": k_cache, "v_cache": v_cache, "page_table": page_table,
                "table_stride": table_stride, "table_bytes": table_bytes, "seq_lens": seq_lens,
                "lens64": lens64, "extra": extra, "window": window, "part_in": part_in,
                "scale_bits": scale_bits,
            }  # fmt: skip
            c = kattn_context(key, lds, args, scratch, step_tag(epoch, layer), layout)
            attention.run_kattn(c)

    @flyc.jit
    def launch(
        q: Int64,
        o: Int64,
        k_cache: Int64,
        v_cache: Int64,
        page_table: Int64,
        table_stride: Int32,
        table_bytes: Int32,
        seq_lens: Int64,
        lens64: Int32,
        extra: Int32,
        window: Int32,
        part_in: Int32,
        scale_bits: Int32,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kattn(
            q, o, k_cache, v_cache, page_table, table_stride, table_bytes, seq_lens, lens64,
            extra, window, part_in, scale_bits, scratch, epoch, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
