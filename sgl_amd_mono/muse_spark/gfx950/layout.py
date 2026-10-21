# SPDX-License-Identifier: Apache-2.0
"""Where the Muse Spark launches put their tasks and data, per step width.

Plain Python, evaluated at build time: which CTA runs each stage's tasks, the
scratch regions and the LDS sizes. Kept free of FlyDSL so it is testable without
a GPU (``test/unit/test_muse_spark_layout.py``).
"""

from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, pair_layout

TRACE_SLOTS = 16  # stage stamps a CTA (``device.stamp``), an int64 pair each
from sgl_amd_mono.muse_spark.gfx950.config import (
    HEAD_DIM,
    HIDDEN,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    Q_K,
    QKVG_ROWS,
)

# ---------------------------------------------------------------- K1 (pre_attn)
ROWS = 16  # projection rows a GEMV task: the MFMA's M
TILE = 16  # tokens an MFMA B operand holds: a token tile
K_STEP = 32  # the bf16 MFMA's K
HALVES = 2  # every row group as two K halves: 288 tasks over 256 CTAs
K_HALF = HIDDEN // HALVES  # 4096
K_PER_WAVE = K_HALF // WAVES  # 512: each wave's contiguous K range of a half
K_STEPS = K_PER_WAVE // K_STEP  # 16 MFMAs a wave a token tile
STEP_BATCH = 8  # k-steps whose weight loads go out together (32 VGPRs)
GEMV_TASKS = QKVG_ROWS // ROWS  # 144 row groups
SPLIT_TASKS = GEMV_TASKS * HALVES  # 288: CTAs 0 .. 31 take a second task
HEAD_TASKS = LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS  # 20: a CTA a q, k or v head
GATE_TASKS = Q_K // HEAD_DIM  # 16: the gate rows' half sums, 64 rows a task
HT0 = SPLIT_TASKS - BLOCKS  # the first head task's CTA, past the CTAs with a second GEMV task
GT0 = HT0 + HEAD_TASKS  # the first gate task's CTA
ELEMS = 4  # a head task lane's dims: 16 lanes x 4 = HEAD_DIM, two RoPE pairs
TOKEN_LANES = 16
TOKENS_PER_PASS = THREADS // TOKEN_LANES  # 32 tokens a head / gate pass
XS_LDS_MAX = 8  # steps of up to this many rows stage x in LDS (16 KB a row)
RED_FLOATS = WAVES * 64 * 4  # one token tile's MFMA accumulators in LDS
assert QKVG_ROWS % ROWS == 0 and K_STEPS % STEP_BATCH == 0
assert BLOCKS <= SPLIT_TASKS <= 2 * BLOCKS
assert GT0 + GATE_TASKS <= BLOCKS
assert MAX_TOKENS % TILE == 0


def token_tiles(s: int) -> int:
    """The token tiles of a step of ``s`` rows (a GEMV accumulator each)."""
    return -(-s // TILE)


def token_passes(s: int) -> int:
    """The head / gate task passes of a step: TOKENS_PER_PASS tokens a pass."""
    return -(-s // TOKENS_PER_PASS)


def k1_scratch_layout() -> dict:
    """K1's scratch regions, name -> (byte offset, bytes), sized for MAX_TOKENS:
    the halves' fp32 partials (a pair a (half, token, row)) and, in a merged
    launch, a flag a token for its x."""
    return pair_layout(
        [("k1_part", HALVES * MAX_TOKENS * QKVG_ROWS), ("k1_xflag", MAX_TOKENS), ("k1_trace", BLOCKS * TRACE_SLOTS)],
        align=256,
    )


def scratch_bytes(layout: dict) -> int:
    return max(off + nbytes for off, nbytes in layout.values())


def k1_lds_floats(s: int) -> tuple[int, int]:
    """(reduction floats, staged-x floats) of K1's LDS at width ``s``."""
    xs = s * HIDDEN // 2 if s <= XS_LDS_MAX else 4
    return token_tiles(s) * RED_FLOATS, xs


def k1_lds_bytes(s: int) -> int:
    return 4 * sum(k1_lds_floats(s))

# ---------------------------------------------------------------- K-attn (attention)
GQA = 8  # query heads a KV head on a rank
ATTN_CHUNK = 256  # keys a chunk of the task loop
ATTN_MAX_PARTS = 64  # partitions a request at most (the host sizes ``part`` to fit)
ATTN_Q_PITCH = HEAD_DIM + 4  # LDS pitches (bf16 elements), 16 B aligned, bank-spread
ATTN_V_PITCH = 40  # keys: a 32-key V^T sub-block's row
ATTN_PART_ROW = 2 + HEAD_DIM  # fp32 pairs a partial row: (m, l, O[64])
ATTN_KEY_SPLITS = WAVES  # the waves share a task's keys
ATTN_KEYS_PER_WAVE = ATTN_CHUNK // ATTN_KEY_SPLITS  # 32
ATTN_KEY_TILES = ATTN_KEYS_PER_WAVE // 16  # 2
ATTN_V_STEPS = ATTN_KEYS_PER_WAVE // 32  # 32-key PV steps a wave: 1
ATTN_P_PITCH = ATTN_KEYS_PER_WAVE + 8  # bf16 elements, 16 B aligned
assert GQA * LOCAL_KV_HEADS == LOCAL_Q_HEADS


def attn_rows(q_len: int) -> int:
    """Rows a (request, KV head): the request's q_len rows x GQA heads."""
    return q_len * GQA


def kattn_scratch_layout(tokens: int, q_len: int) -> dict:
    """K-attn's scratch: (request, head, partition)'s (m, l, O[64]) a row, fp32 pairs."""
    rows = attn_rows(q_len)
    pairs = (tokens // q_len) * LOCAL_KV_HEADS * ATTN_MAX_PARTS * rows * ATTN_PART_ROW
    return pair_layout([("attn_part", pairs), ("attn_trace", BLOCKS * TRACE_SLOTS)], align=256)


def kattn_lds_bytes(nt: int) -> int:
    qs = nt * 16 * ATTN_Q_PITCH * 2
    pt = WAVES * nt * 16 * ATTN_P_PITCH * 2
    vt = WAVES * HEAD_DIM * ATTN_V_PITCH * 2
    comb = nt * 16 * (ATTN_KEY_SPLITS - 1) * (2 + HEAD_DIM) * 4
    return qs + pt + vt + comb + RED_FLOATS * 4

