"""Page 5 -- bottleneck attribution, scoped to what this sweep can actually support.

No profiler ever ran here: there is no .ncu-rep, no .nsys-rep and no
results/profile/. So there are no counters, no occupancy and no measured DRAM
traffic, and every panel that would need them says so instead of guessing. What
is left is mechanism-consistent evidence -- an analytic roofline built from
modelled traffic, the empirical scaling exponents, allocator footprint, kernel
counts from the dispatch traces, and which backend sat on each side of a
ranking inversion. That is an argument about mechanism, not an attribution.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from _data import (COLOURS, filter_bar, gpu_meta, not_collected, note, ok,
                   provenance, short_gpu)

# tflops bakes a 3.5x forward/backward multiplier into the prefill numerator and
# uses a linear FLOP count for decode, and the two timers do not share a clock.
# One selector over the legal combinations is the only safe way to touch it: a
# free-form facet would let a reader put fwd and fwd_bwd on one axis.
_ARMS = [("prefill", "fwd", "block_bench"), ("prefill", "fwd_bwd", "block_bench"),
         ("decode", "fwd", "block_bench"), ("decode", "fwd", "do_bench_cudagraph")]

_ARM_LABEL = {
    ("prefill", "fwd", "block_bench"): "prefill, forward (block_bench)",
    ("prefill", "fwd_bwd", "block_bench"): "prefill, fwd+bwd (block_bench)",
    ("decode", "fwd", "block_bench"): "decode, forward (block_bench)",
    ("decode", "fwd", "do_bench_cudagraph"): "decode, CUDA-graph (do_bench_cudagraph)",
}

_STRIP_NOTE = ("P4h-fa3 is sm90-only (UNSUPPORTED on every A100 prefill row, never "
               "built on A10); D1-inductor is ~96% UNSUPPORTED and the "
               "P1-inductor-nofuse/-where arms ~91%, all deliberate bf16-only "
               "strips. Their absence is a declined configuration, not a defeat.")


def render(D):
    st.title("Bottleneck attribution")
    st.write(
        "What limits each kernel is a question this sweep answers only indirectly. "
        "No hardware counters were collected, so nothing below is an attribution in "
        "the profiler sense. Each panel states which mechanism it is consistent "
        "with and what it cannot separate.")

    # rows.parquet already carries a 'regime' column and _data.cell_key() concats
    # a second one, so df.regime comes back as a two-column frame and every
    # comparison against it raises. Drop the duplicate label here rather than in
    # _data.py, which is shared and not mine to change.
    rows, cells = _dedupe(D.get("rows")), _dedupe(D.get("cells"))
    if rows is None or not len(rows):
        not_collected("Everything on this page",
                      "rows.parquet is empty -- run a sweep, then python -m akp.analysis.")
        return
    okr = ok(rows)

    _absent()
    st.divider()
    _roofline(okr, D)
    st.divider()
    _regimes(okr)
    st.divider()
    _scaling(D.get("scaling"))
    st.divider()
    _memory(okr)
    st.divider()
    _kernels(D.get("audit"))
    st.divider()
    _inversions(D.get("inversions"), D.get("audit"), cells)


# --------------------------------------------------------------------------- #

def _dedupe(df):
    if df is None or not len(df):
        return df
    return df.loc[:, ~df.columns.duplicated()]


def _absent():
    st.header("What was not measured")
    not_collected(
        "Hardware counter table",
        "Nsight Compute was never run: there is no .ncu-rep in the repository and "
        "results/profile/ does not exist. No DRAM throughput, L2 hit rate or "
        "instruction mix figure exists to report.")
    not_collected(
        "Occupancy and warp-stall fingerprint",
        "Achieved occupancy and stall reasons come from ncu's scheduler counters, "
        "which need GPU performance-counter permission the rented hosts did not "
        "grant. Nothing in results/ carries a per-warp figure.")
    not_collected(
        "Measured-traffic roofline",
        "A measured roofline needs counted DRAM bytes per kernel from Nsight. The "
        "roofline below uses bytes computed from the tensor shapes, so it can only "
        "be as right as that model.")
    note("Launch-overhead fractions and per-kernel time shares are likewise "
         "unavailable: block_bench keeps a median over blocks and discards the "
         "individual block times, and no Nsight Systems timeline was captured.")


def _roofline(okr, D):
    st.header("Analytic roofline -- modelled traffic")
    st.caption(
        "Arithmetic intensity is FLOPs over bytes as computed from the tensor "
        "shapes, not bytes the hardware moved. Points sit where the model says "
        "they should, so this shows which regime a kernel lives in, not how much "
        "of the machine it used.")

    have = {tuple(t) for t in
            okr[["regime", "mode", "timer"]].drop_duplicates().itertuples(index=False)}
    arms = [a for a in _ARMS if a in have]
    if not arms:
        not_collected("Roofline",
                      "No OK rows carry regime, mode and timer together.")
        return
    arm = st.radio("Arm", arms, format_func=lambda a: _ARM_LABEL[a],
                   horizontal=True, key="attr_arm")
    st.caption(
        "One arm at a time on purpose. Decode FLOPs are linear and land two to "
        "three orders of magnitude below prefill; fwd_bwd multiplies the prefill "
        "FLOP count by an assumed 3.5; and the two timers do not share a clock. "
        "Any of those pooled onto one axis produces a wrong number.")

    d = okr[(okr.regime == arm[0]) & (okr["mode"] == arm[1]) & (okr.timer == arm[2])]
    d = filter_bar(d, key="attr_roof", fields=("gpu", "dt", "D", "B", "N"))
    if not len(d):
        st.info("No rows for this selection.")
        return

    gpus = sorted(d.gpu.unique())
    fig = px.scatter(
        d, x="arith_intensity", y="tflops", color="implementation",
        facet_col="gpu", category_orders={"gpu": gpus},
        color_discrete_map=COLOURS, log_x=True, log_y=True, opacity=0.65,
        hover_data=["N", "B", "D", "dt", "median_us"],
        labels={"arith_intensity": "arithmetic intensity (modelled FLOP/byte)",
                "tflops": "achieved TFLOP/s"})
    # One roof only. There is no peak-FLOPs figure anywhere in the manifest, and
    # pasting a vendor spec sheet number would be inventing a measurement.
    bw = d.groupby("gpu").peak_bw_gbs.max()
    x0, x1 = float(d.arith_intensity.min()), float(d.arith_intensity.max())
    for j, g in enumerate(gpus):
        b = bw.get(g)
        if not b or pd.isna(b):
            continue
        fig.add_trace(
            go.Scatter(x=[x0, x1], y=[x0 * b / 1e3, x1 * b / 1e3], mode="lines",
                       line=dict(dash="dash", width=1.5, color="#888888"),
                       name="memory roof", legendgroup="roof",
                       showlegend=(j == 0),
                       hovertemplate="memory roof, %d GB/s<extra></extra>" % b),
            row=1, col=j + 1)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(col=1, title_text="achieved TFLOP/s (no compute roof: the "
                                       "repository holds no peak-FLOPs figure)")
    fig.update_layout(legend_title_text="", height=460)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The dashed roof is arithmetic intensity times measured_peak_bw_gbs, an "
        "achieved 512 MB device-to-device copy. That copy sits below the true "
        "read-stream peak, so points can and do cross it -- about 546 rows exceed "
        "100% of it sweep-wide. Crossing means the copy benchmark under-measures "
        "the peak, not that a kernel beat physics. theoretical_bw_gbs is null on "
        "all three devices, so nothing here should be called a theoretical peak.")
    gm = gpu_meta(D.get("summary") or {})
    if len(gm):
        cols = [c for c in ["gpu", "cc", "sm_count", "measured_bw_gbs",
                            "theoretical_bw_gbs", "l2_mb", "event_overhead_us"]
                if c in gm.columns]
        st.dataframe(gm[cols], use_container_width=True, hide_index=True)
    prov = provenance(okr[okr.regime == arm[0]])
    if len(prov):
        st.caption("Rows in this regime by commit. A10 and A100 decode came off a "
                   "different commit and driver branch than H100 decode, so any "
                   "cross-GPU decode gap carries a software difference too; "
                   "prefill is commit-homogeneous.")
        st.dataframe(prov, use_container_width=True, hide_index=True)
    note(_STRIP_NOTE)


def _regimes(okr):
    st.header("Prefill and decode do not share a regime")
    st.caption(
        "The same axis, split by regime and pass, so the separation is shown "
        "rather than assumed. Prefill forward runs at a modelled 16-2048 "
        "FLOP/byte; decode never exceeds 4. They are not two points on one curve, "
        "and no single roofline covers both.")
    d = okr.assign(arm=okr.regime + " / " + okr["mode"])
    fig = px.box(d.sort_values(["regime", "mode"]), x="implementation",
                 y="arith_intensity", color="implementation", facet_col="arm",
                 log_y=True, color_discrete_map=COLOURS, points=False,
                 labels={"arith_intensity": "modelled FLOP/byte"})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(showticklabels=False, title_text="")
    fig.update_layout(legend_title_text="", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        d.groupby("arm").agg(rows=("arith_intensity", "size"),
                             min_flop_per_byte=("arith_intensity", "min"),
                             median_flop_per_byte=("arith_intensity", "median"),
                             max_flop_per_byte=("arith_intensity", "max"),
                             median_us=("median_us", "median")).round(2).reset_index(),
        use_container_width=True, hide_index=True)
    note("Spread inside an arm is configuration spread -- head dim, batch, KV "
         "length -- across cells. It is not a within-cell distribution: "
         "block_bench keeps only a median with p5/p95 and discards its 30 block "
         "times, so no per-call distribution exists anywhere in this dataset.")


def _scaling(sc):
    st.header("Empirical scaling exponents")
    if sc is None or not len(sc):
        not_collected("Scaling exponents",
                      "scaling.parquet is empty: the stage-A fit needs several "
                      "sequence lengths per series and none survived.")
        return
    st.caption(
        "alpha is the fitted exponent of median latency against KV or sequence "
        "length within one otherwise-fixed series. This is the closest thing to a "
        "bound argument the data supports: an exponent is consistent with a "
        "mechanism, it does not identify one.")
    r2min = st.slider("Minimum fit r-squared", 0.0, 0.99, 0.90, 0.05, key="attr_r2")
    s = sc.assign(gpu=sc.gpu_name.map(short_gpu))
    s = s[s.r2 >= r2min]
    note("%d of %d fitted series kept at r2 >= %.2f." % (len(s), len(sc), r2min))
    if not len(s):
        st.info("No series survive that threshold.")
        return
    fig = px.box(s.sort_values(["regime", "implementation"]), x="implementation",
                 y="alpha", color="gpu", facet_row="regime", points="outliers",
                 labels={"alpha": "fitted exponent of latency vs N"})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(matches=None)
    fig.update_xaxes(title_text="")
    fig.update_layout(height=560, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Reference points, deliberately not drawn on the chart: prefill attention "
        "work is quadratic in N (alpha 2) while its memory traffic is linear "
        "(alpha 1), and a decode step that reads the whole KV cache is linear "
        "(alpha 1). Prefill sits near 1.5 and decode below 1. That is what a "
        "kernel which is neither purely compute-bound nor purely stream-bound "
        "looks like -- and equally what a kernel with a fixed launch floor at "
        "small N looks like. Without counters those two cannot be told apart.")
    st.dataframe(
        s.groupby(["regime", "implementation"]).agg(
            series=("alpha", "size"), median_alpha=("alpha", "median"),
            p5_alpha=("alpha", lambda x: x.quantile(0.05)),
            p95_alpha=("alpha", lambda x: x.quantile(0.95)),
            median_r2=("r2", "median")).round(3).reset_index(),
        use_container_width=True, hide_index=True)


def _memory(okr):
    st.header("Allocator footprint and fragmentation")
    st.caption(
        "mem_overhead is peak allocated over the working set the shapes imply, so "
        "it counts workspace and materialised intermediates. Reserved over "
        "allocated is what the caching allocator held but never handed out -- "
        "fragmentation and block rounding, not kernel traffic.")
    d = filter_bar(okr, key="attr_mem", fields=("gpu", "dt", "D", "B", "N", "md"))
    if not len(d):
        st.info("No rows for this selection.")
        return
    d = d.assign(frag=d.peak_reserved_mb / d.peak_allocated_mb)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(d.sort_values("regime"), x="implementation", y="mem_overhead",
                     color="implementation", facet_col="regime", log_y=True,
                     color_discrete_map=COLOURS, points=False,
                     labels={"mem_overhead": "peak allocated / modelled working set"})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_xaxes(showticklabels=False, title_text="")
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        note("A kernel that materialises the full N-by-N score matrix shows up "
             "here as an overhead in the hundreds; a fused one stays near 1.")
    with c2:
        fig = px.scatter(d, x="peak_allocated_mb", y="frag", color="implementation",
                         facet_col="regime", log_x=True, opacity=0.5,
                         color_discrete_map=COLOURS,
                         hover_data=["N", "B", "D", "dt"],
                         labels={"peak_allocated_mb": "peak allocated (MB)",
                                 "frag": "peak reserved / peak allocated"})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.add_hline(y=1.0, line_dash="dot", line_width=1)
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        note("Ratios far above 1 at small allocations are the allocator's block "
             "granularity, not a kernel property. The cases worth reading are "
             "large allocations that still reserve well beyond what they use.")
    st.dataframe(
        d.groupby(["regime", "implementation"]).agg(
            cells=("mem_overhead", "size"),
            median_overhead=("mem_overhead", "median"),
            p95_overhead=("mem_overhead", lambda x: x.quantile(0.95)),
            median_frag=("frag", "median"),
            max_reserved_mb=("peak_reserved_mb", "max")).round(2).reset_index(),
        use_container_width=True, hide_index=True)


def _kernels(audit):
    st.header("Kernels per call")
    if audit is None or not len(audit):
        not_collected("Kernel counts",
                      "dispatch_audit.parquet is empty: no call was traced.")
        return
    st.caption(
        "How many distinct CUDA kernels one traced call launched. No durations "
        "were kept, so this says nothing about where the time went -- but a call "
        "that launches ten kernels is not fused and one that launches a single "
        "kernel is, and that is real evidence about mechanism.")
    # Repeats duplicate the same (cell, gpu, impl); collapse before counting. The
    # trace miss is biased toward the fastest implementations, so unprobed stays
    # its own category rather than being normalised away.
    a = audit.drop_duplicates(["cell", "gpu_name", "implementation"]).copy()
    a["traced"] = a.observed.ne("unprobed") & a.probed.fillna(False).astype(bool)
    share = a.groupby(["implementation", "traced"]).size().rename("cells").reset_index()
    share["trace"] = share.traced.map({True: "traced", False: "unprobed"})
    fig = px.bar(share, x="implementation", y="cells", color="trace",
                 color_discrete_map={"traced": "#2e9e5b", "unprobed": "#9e9e9e"})
    fig.update_xaxes(title_text="")
    fig.update_layout(height=340, legend_title_text="", barmode="stack")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Grey is the '<no-cuda-kernels>' sentinel: the dispatch hook returned "
        "nothing for that call. It covers roughly a third of OK cells and the miss "
        "is worst on the fastest implementations, so every count below is "
        "conditional on there being a trace at all.")

    t = a[a.traced]
    if not len(t):
        st.info("No traced cells to count kernels from.")
        return
    fig = px.box(t, x="implementation", y="n_kernels", color="implementation",
                 color_discrete_map=COLOURS, points="outliers",
                 labels={"n_kernels": "distinct kernels in the trace"})
    fig.update_xaxes(title_text="")
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        a.groupby("implementation").agg(
            cells=("traced", "size"),
            unprobed_pct=("traced", lambda x: round(100 * (1 - x.mean()), 1)),
        ).join(t.groupby("implementation").agg(
            median_kernels=("n_kernels", "median"),
            max_kernels=("n_kernels", "max"),
            backends=("observed", lambda s: ", ".join(sorted(set(s)))))
        ).reset_index(),
        use_container_width=True, hide_index=True)
    note("Kernel counts are across cells, not within one call: the trace records "
         "which kernels a call launched, never how long each ran. " + _STRIP_NOTE)


def _inversions(inv, audit, cells):
    st.header("Which kernel replaced which")
    if inv is None or not len(inv):
        not_collected("Inversion attribution",
                      "inversions.parquet is empty: an inversion needs the same "
                      "pair of implementations measured on two GPUs.")
        return
    st.caption(
        "Every ranking inversion, joined to the backend each implementation "
        "actually dispatched to on each GPU and to the absolute latencies behind "
        "the ratio. r1 and r2 are the a-over-b latency ratio on GPU 1 and GPU 2; "
        "a crossing of 1.0 between them is the inversion.")
    i = inv.copy()
    i["regime"] = i.cell.str.split("|").str[0]
    i["gpu_1"] = i.gpu_a.map(short_gpu)
    i["gpu_2"] = i.gpu_b.map(short_gpu)

    if audit is not None and len(audit):
        g = (audit.drop_duplicates(["cell", "gpu_name", "implementation"])
             .groupby(["cell", "gpu_name", "implementation"])["observed"]
             .agg(lambda s: "/".join(sorted(set(s)))))
        for col, (imp, gpu) in {"a_on_1": ("a", "gpu_a"), "b_on_1": ("b", "gpu_a"),
                                "a_on_2": ("a", "gpu_b"), "b_on_2": ("b", "gpu_b")}.items():
            i[col] = g.reindex(pd.MultiIndex.from_arrays([i.cell, i[gpu], i[imp]])).values
    # A ratio without the latencies it came from hides whether the flip is worth
    # anything: a 1.2x inversion at 12us and at 12ms are different findings.
    if cells is not None and len(cells):
        m = cells.set_index(["cell", "gpu_name", "implementation"]).median_us
        for col, (imp, gpu) in {"a_us_1": ("a", "gpu_a"), "b_us_1": ("b", "gpu_a"),
                                "a_us_2": ("a", "gpu_b"), "b_us_2": ("b", "gpu_b")}.items():
            i[col] = m.reindex(pd.MultiIndex.from_arrays([i.cell, i[gpu], i[imp]])).values

    c1, c2, c3 = st.columns(3)
    regs = sorted(i.regime.dropna().unique())
    reg = c1.multiselect("regime", regs, default=regs, key="attr_inv_reg")
    pairs = sorted((i.gpu_1 + " / " + i.gpu_2).unique())
    pair = c2.multiselect("GPU pair", pairs, default=pairs, key="attr_inv_pair")
    only = c3.checkbox("Practically separated only", value=True, key="attr_inv_prac")
    v = i[i.regime.isin(reg)] if reg else i
    if pair:
        v = v[(v.gpu_1 + " / " + v.gpu_2).isin(pair)]
    if only:
        v = v[v.practical.fillna(False)]
    st.caption("%d of %d inversions shown. Sweep-wide: %d practically separated, "
               "%d statistically separated."
               % (len(v), len(i), int(i.practical.fillna(False).sum()),
                  int(i.sig.fillna(False).sum())))
    if not len(v):
        st.info("No inversions match this selection.")
        return

    keep = [c for c in ["regime", "gpu_1", "gpu_2", "a", "b", "r1", "r2",
                        "a_on_1", "b_on_1", "a_on_2", "b_on_2",
                        "a_us_1", "b_us_1", "a_us_2", "b_us_2",
                        "sig", "practical", "pairs_examined", "cell"] if c in v]
    st.dataframe(v[keep].sort_values(["regime", "r1"]).round(3),
                 use_container_width=True, hide_index=True,
                 column_config={"a_us_1": "a us (1)", "b_us_1": "b us (1)",
                                "a_us_2": "a us (2)", "b_us_2": "b us (2)"})
    if "a_on_1" in v.columns:
        st.caption(
            "In %.0f%% of the shown inversions implementation `a` dispatched to a "
            "different backend on the two GPUs: a kernel substitution underneath a "
            "stable label, which is the mechanism the ratio flip is consistent "
            "with. Where both sides kept their backend on both GPUs the flip is an "
            "architecture effect instead, and nothing here says which part of the "
            "architecture." % (100 * (v.a_on_1 != v.a_on_2).mean()))
    st.caption(
        "A backend reading 'unprobed', or two names joined by a slash, means at "
        "least one repeat of that cell returned no kernel trace. Decode inversions "
        "involving H100 also span a commit and driver change (see the provenance "
        "table above), so those carry a software difference alongside the "
        "architecture one.")
    st.info("None of this is counter-based attribution. It shows which kernel ran "
            "on each side of a flip and how long each side took, not why one was "
            "faster.", icon=":material/info:")
