"""Stage timeline of the Muse Spark launches on one gfx950 GPU (a TP / EP group of
one): the kernels built with ``trace`` stamp the 100 MHz clock at their stage
boundaries, a CTA each; this runs one layer step (K1, attention, K2, K3) at a
width and prints, per launch and stamp, when the CTAs reached it (median / p90 /
max over the CTAs, in us from the launch's first stamp), by CTA class where the
stages differ. ``python trace_muse_spark_layer.py [S] [lens...] [--window W]``."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_muse_spark_layer import LAYER, LENS, Buffers, Layer, _attn_args, _epoch, _k1, _k2_args, _k3_args  # noqa: E402

K1_STAMPS = ["entry", "x ready", "gemv done", "head/gate done"]
KA_STAMPS = ["entry", "attention tasks done", "merge done"]
K2_STAMPS = ["entry", "og done", "o tasks done", "a_part done", "x arrived", "passes done", "pooled out", "end"]
K3_STAMPS = ["entry", "flags", "plan", "ug done", "dn done", "fin done", "unpool done", "end", "unp flags", "unp dma", "unp mfma", "unp pushed"]


def _stamps(scratch, layout, region, blocks, slots):
    off, nbytes = layout[region]
    t = scratch[off : off + nbytes].view(torch.int64).reshape(-1)[: blocks * slots].reshape(blocks, slots).cpu()
    return t


def _report(name, t, names, classes=None):
    """t [BLOCKS, SLOTS] ticks; classes: list of (label, cta mask)."""
    used = [i for i in range(len(names)) if (t[:, i] != 0).any()]
    t0 = t[:, 0][t[:, 0] != 0].min().item()
    us = (t.double() - t0) / 100.0  # 100 MHz -> us
    print(f"== {name}: {len(used)} stamps, launch span {us[:, used].max().item():7.1f} us (first entry -> last stamp)")
    groups = classes or [("all CTAs", torch.ones(t.shape[0], dtype=torch.bool))]
    for label, mask in groups:
        if not mask.any():
            continue
        row = []
        for i in used:
            v = us[mask, i]
            v = v[t[mask, i] != 0]
            if v.numel() == 0:
                continue
            row.append(f"{names[i]} p50 {v.median().item():6.1f} p90 {v.quantile(0.9).item():6.1f} max {v.max().item():6.1f}")
        print(f"  [{label:>14s} n={int(mask.sum())}] " + " | ".join(row))


def main():
    from sgl_amd_mono.common.plan import BLOCKS
    from sgl_amd_mono.muse_spark.gfx950.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx950.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx950.layer import K1Build, KAttnBuild, build_k1, build_kattn
    from sgl_amd_mono.muse_spark.gfx950.layout import TRACE_SLOTS, k1_scratch_layout, kattn_scratch_layout
    from sgl_amd_mono.muse_spark.gfx950.stages import ffn, post_attn
    from sgl_amd_mono.muse_spark.gfx950.stages.ffn import k3_bases

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    window = 0
    if "--window" in sys.argv:
        window = int(sys.argv[sys.argv.index("--window") + 1])
        args = [a for a in args if a != str(window)] if False else args
    S = int(args[0]) if args else 8
    lens = [int(x) for x in args[1:]] if len(args) > 1 else LENS[: S // 8]
    ql = 8
    dev = torch.device("cuda")
    L = Layer(S, ql, dev, seed=70 + S, lens=lens, window=window)
    B = Buffers(S, ql, dev)
    epoch = _epoch(dev, deadline_s=60.0)
    stream = torch.cuda.current_stream()
    k1 = build_k1(K1Build(tokens=S, trace=True))
    ka = build_kattn(KAttnBuild(tokens=S, q_len=ql, trace=True))
    k2 = build_k2(K2Build(tokens=S, npes=1, trace=True))
    k3 = build_k3(K3Build(tokens=S, npes=1, trace=True))
    try:
        for rep in range(3):  # the third step's stamps are reported (warm caches, steady tags)
            B.res.copy_(L.res0)
            _k1(k1, L, B, L.x0, L.w_qkvg, L.k_cache, L.v_cache, epoch, LAYER, 0, stream)
            ka(*_attn_args(L, B), epoch.data_ptr(), LAYER, stream=stream)
            k2(B.o.data_ptr(), *_k2_args(L, B), B.peers.local, B.peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, stream=stream)  # fmt: skip
            a3 = _k3_args(L, B)
            k3(B.xin.data_ptr(), B.ids.data_ptr(), B.scores.data_ptr(), a3[0], *a3[1:5], a3[5], B.res.data_ptr(), *a3[6:],
               B.peers.local, B.peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, 0, stream=stream)  # fmt: skip
            epoch[0] += 1
            torch.cuda.synchronize()
        print(f"S={S} lens={lens} window={window}; ids of token 0: {B.ids[0].tolist()}")
        cta = torch.arange(BLOCKS)
        _report("K1", _stamps(B.scratch_k1, k1_scratch_layout(), "k1_trace", BLOCKS, TRACE_SLOTS), K1_STAMPS, [
            ("head tasks", (cta >= 32) & (cta < 52)), ("gate tasks", (cta >= 52) & (cta < 68)), ("half+quarter", (cta >= 68) & (cta < 132)), ("1 half task", (cta < 32) | (cta >= 132)),
        ])  # fmt: skip
        _report("K-attn", _stamps(B.scratch_a, kattn_scratch_layout(S, ql), "attn_trace", BLOCKS, TRACE_SLOTS), KA_STAMPS, [
            ("merge CTAs", cta >= BLOCKS - (S // ql) * 2 * 4), ("other CTAs", cta < BLOCKS - (S // ql) * 2 * 4),
        ])  # fmt: skip
        chain0 = post_attn.CHAIN0
        _report("K2", _stamps(B.scratch_k2, post_attn.k2_scratch_layout(), "k2_trace", BLOCKS, TRACE_SLOTS), K2_STAMPS, [
            ("stream CTAs", cta < chain0), ("chain CTAs", cta >= chain0),
        ])  # fmt: skip
        b = k3_bases(S, 1)
        _report("K3", _stamps(B.scratch_k3, ffn.k3_scratch_layout(), "k3_trace", BLOCKS, TRACE_SLOTS), K3_STAMPS, [
            ("bnd CTAs", cta < S), ("fin CTAs", (cta >= b["fin0"]) & (cta < b["unp0"])), ("unpool CTAs", cta >= b["unp0"]),
        ])  # fmt: skip
    finally:
        B.close()


if __name__ == "__main__":
    main()
