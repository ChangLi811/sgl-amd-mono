# SPDX-License-Identifier: Apache-2.0
"""Build the Muse Spark launches: K1 standalone (``build_k1``) for now."""

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
from sgl_amd_mono.muse_spark.gfx950.layout import XS_LDS_MAX, k1_scratch_layout
from sgl_amd_mono.muse_spark.gfx950.stages import pre_attn

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
