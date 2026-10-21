# SPDX-License-Identifier: Apache-2.0
"""Muse Spark's device helpers that ``sgl_amd_mono.common`` does not hold: the
step's mailbox tag and the fences of a plain-words hand-off (the same ones the
MiniMax-M3 family carries)."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import rsrc, uniform
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
    fx.memory_fence(
        syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Release
    )


def acquire_loads():
    """Order this thread's later loads after a flag it polled."""
    fx.memory_fence(
        syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Acquire
    )
