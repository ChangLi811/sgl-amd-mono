# SPDX-License-Identifier: Apache-2.0
"""Muse Spark's device helpers that ``sgl_amd_mono.common`` does not hold: the
step's mailbox tag and the fences of a plain-words hand-off (on the llvm dialect
directly, which every FlyDSL this family runs on provides)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import T, as_ir_value

from sgl_amd_mono.common.ops import clock, rsrc, uniform
from sgl_amd_mono.muse_spark.gfx950.config import LAYER_SLOTS


def step_tag(epoch_ptr, layer):
    """The layer's mailbox tag this step, ``step * LAYER_SLOTS + layer + 1``: the
    step is word 0 of the epoch buffer, which the host moves on after a step's last
    launch. Graph-safe, and no mailbox is cleared between steps."""
    step = uniform(bo.buffer_load(rsrc(epoch_ptr), 0, vec_width=1, dtype=T.i32))
    return step * LAYER_SLOTS + layer + 1


def release_stores():
    """Order this wave's earlier device-scope stores before a flag that other CTAs
    poll: wait for them (a workgroup release fence alone does not wait for global
    stores), then the fence."""
    rocdl.s_waitcnt(vmcnt=0)
    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="workgroup")


def acquire_loads():
    """Order this thread's later loads after a flag it polled."""
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="workgroup")


# ---------------------------------------------------------------- stage trace
TRACE_SLOTS = 16  # stamps a CTA: int64 clock ticks (100 MHz) at trace[(bid TRACE_SLOTS + idx)]


@flyc.jit
def stamp(c, idx: int):
    """Trace stamp ``idx`` of this CTA, compiled in by a build with ``trace``: the
    100 MHz realtime clock, an int64 in the launch's ``*_trace`` scratch region
    (``c["trace"]``, a pair a stamp), written by thread 0."""
    if const_expr(not c.get("trace_on", False)):
        return
    if c["tid"] == 0:
        now = clock()
        addr = c["trace"].value + fx.Int64((c["bid"] * TRACE_SLOTS + idx) * 8)
        ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr)).result
        llvm.StoreOp(as_ir_value(now), ptr, alignment=8)
