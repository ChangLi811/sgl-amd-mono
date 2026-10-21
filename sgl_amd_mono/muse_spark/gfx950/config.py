# SPDX-License-Identifier: Apache-2.0
"""The shapes the Muse Spark gfx950 kernels are built for: one rank at TP8 / EP8."""

HIDDEN = 8192
HEAD_DIM = 64
LOCAL_Q_HEADS = 16  # 128 / TP8
LOCAL_KV_HEADS = 2  # 16 / TP8
Q_K = LOCAL_Q_HEADS * HEAD_DIM  # 1024: the q, o_gate and attention-output widths
KV_K = LOCAL_KV_HEADS * HEAD_DIM  # 128
QKVG_ROWS = 2 * Q_K + 2 * KV_K  # 2304: q | k | v | o_gate rows of the fused projection
EXPERTS = 256
LOCAL_EXPERTS = 32  # 256 / EP8
TOP_K = 8
MOE_HIDDEN = 4096  # the pooled width between the pooler and the unpooler
INTER = 4096  # an expert's intermediate width (whole: EP8 keeps expert TP at 1)
MXFP4_GROUP = 32
WINDOW = 2047  # a sliding layer attends to sliding_window - 1 keys
PAGE = 16
RMS_EPS = 1e-5
POST_EPS = 1e-8
ROUTE_EPS = 1e-15
ROUTER_MULT = 0.17677669529663687
ROUTE_SCALE = 1.0
QK_SCALE = 5.4730064863875  # applied to q after its per-head norm
SOFTMAX_SCALE = HEAD_DIM**-0.5
GATE_TEMPERATURE = 0.3
MAX_TOKENS = 64  # C8 x 8 draft tokens
WIDTHS = (8, 16, 32, 64)  # C1 / C2 / C4 / C8 verify steps at 8 draft tokens
