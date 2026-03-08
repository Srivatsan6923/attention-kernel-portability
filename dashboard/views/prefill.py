"""Prefill performance.

Prefill is the regime where the implementations actually separate: the median
slowest/fastest ratio is ~10x per configuration here against ~3x in decode, so
this is the page where a backend choice is worth arguing about.

Nothing here is recomputed from raw timings -- analysis.py owns the numbers.
The one derived quantity is the fwd_bwd/fwd latency ratio, built from median_us
pairs on purpose: tflops bakes an ASSUMED 3.5x FLOP multiplier into its fwd_bwd
numerator, so a ratio taken from tflops would only return 3.5.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from _data import (COLOURS, STATUS_MEANING, short_gpu, provenance, ok, note,
                   not_collected, filter_bar)

GPU_ORDER = ["A10", "A100", "H100"]
BASELINE = "P2c-sdpa-flash"

# sm90-only or deliberate bf16-only strips. Their UNSUPPORTED rows are a design
# decision, not a defeat, and every panel they appear on has to say so.
COVERAGE_NOTE = {
    "P4h-fa3": "sm90 only -- UNSUPPORTED on every A100 prefill row, never ran on A10",
    "P1-inductor-nofuse": "bf16-only ablation arm, ~94% UNSUPPORTED by design",
    "P1-inductor-where": "bf16-only ablation arm, ~94% UNSUPPORTED by design",
}


def _prefill(D):
    r = D["rows"]
    if "regime" not in r.columns:      # no parquet found, or an empty frame
        return pd.DataFrame()
    # _data concatenates cell_key() onto rows, which already carries 'regime',
    # so the frame arrives with a duplicated label and any mask on it explodes.
    r = r.loc[:, ~r.columns.duplicated()]
    return r[r.regime == "prefill"]


def _orders(df):
    return {"gpu": [g for g in GPU_ORDER if g in set(df.gpu)],
            "implementation": sorted(df.implementation.unique())}


def _tidy(fig):
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(legend_title_text="", margin=dict(t=46, b=8, l=8, r=8))
    return fig


def _empty():
    st.info("No rows for this selection.")


def render(D):
    st.title("Prefill performance")
    p = _prefill(D)
    if not len(p):
        not_collected("Prefill", "rows.parquet contains no prefill rows.")
        return

    spread = (D["summary"] or {}).get("spread_by_regime", {})
    if spread:
        st.warning(
            "Prefill is where the implementations differ most: median "
            "slowest/fastest ratio %.1fx per configuration, against %.1fx in "
            "decode. A backend choice worth ~3x in decode is worth ~10x here."
            % (spread.get("prefill", float("nan")),
               spread.get("decode", float("nan"))))

    st.caption("Coverage limits that every panel below carries: "
               + "; ".join("**%s** -- %s" % (k, v)
                           for k, v in COVERAGE_NOTE.items()))
    cov = (p[p.implementation.isin(COVERAGE_NOTE)]
           .groupby(["implementation", "gpu"]).status.value_counts()
           .unstack(fill_value=0).reset_index())
    with st.expander("Rows behind those limits"):
        st.dataframe(cov, use_container_width=True, hide_index=True)

    prov = provenance(p.loc[:, ~p.columns.duplicated()])
    if len(prov):
        note("Provenance: prefill is commit-homogeneous -- %s. Unlike decode, no "
             "cross-GPU prefill comparison on this page confounds architecture "
             "with a different build."
             % ", ".join("%s %s (%d rows)" % (r.gpu, r.commit, r.rows)
                         for r in prov.itertuples()))
    note("Timer: all %d prefill rows use block_bench, and launch is eager only, "
         "so no CUDA-graph arm with its own timer is mixed in. Cold-cache rows "
         "(%d, H100 only) are excluded from the panels below and belong to the "
         "cache page." % (len(p), int((p.cch == "cold").sum())))

    capped = p.groupby("gpu").power_capped.mean()
    if st.toggle("Exclude power-capped rows (throttle bit 0x4: %s)"
                 % ", ".join("%s %.0f%%" % (g, v * 100)
                             for g, v in capped.items()),
                 value=False, key="prefill_capped",
                 help="Kept by default -- only the erratic 0xF8 group was "
                      "rejected upstream -- but the cap lands unevenly by GPU "
                      "and size, so cross-GPU panels can be re-read without it."):
        p = p[~p.power_capped]

    warm = p[p.cch == "warm"]

    _latency(warm)
    _throughput(warm)
    _speedup(warm)
    _memory(warm)
    _scaling(D)
    _asymmetry(D)


def _latency(warm):
    st.subheader("(a) Latency vs sequence length")
    f = filter_bar(warm, key="prefill_a", fields=("gpu", "dt", "D", "B", "md"))
    g = ok(f)
    if not len(g):
        return _empty()
    g = (g.groupby(["gpu", "implementation", "N"], as_index=False)
         .median_us.median().sort_values("N"))
    st.plotly_chart(_tidy(px.line(
        g, x="N", y="median_us", color="implementation", facet_col="gpu",
        markers=True, log_x=True, log_y=True, color_discrete_map=COLOURS,
        category_orders=_orders(g),
        labels={"median_us": "median latency (us)", "N": "sequence length"})),
        use_container_width=True)
    note("Log-log. Each point is the median over whatever the filters leave open "
         "(batch, head dim, dtype, mode, GQA ratio, causal flag): widen them and "
         "the lines get blunter, not wrong.")


def _throughput(warm):
    st.subheader("(b) Achieved TFLOP/s vs sequence length")
    st.caption("fwd and fwd_bwd are separate figures on purpose. tflops divides "
               "an analytic FLOP count by measured time, and for fwd_bwd that "
               "count is the forward count times an ASSUMED 3.5. The two are not "
               "the same quantity, so a gap between them is the assumption "
               "showing, not a measurement.")
    f = filter_bar(warm, key="prefill_b", fields=("gpu", "dt", "D", "B", "Hkv"))
    g = ok(f)
    if not len(g):
        return _empty()
    for md in [m for m in ("fwd", "fwd_bwd") if m in set(g.md)]:
        sub = (g[g.md == md]
               .groupby(["gpu", "implementation", "N"], as_index=False)
               .tflops.median().sort_values("N"))
        st.plotly_chart(_tidy(px.line(
            sub, x="N", y="tflops", color="implementation", facet_col="gpu",
            markers=True, log_x=True, color_discrete_map=COLOURS,
            category_orders=_orders(sub), title="mode = %s" % md,
            labels={"tflops": "TFLOP/s (%s FLOP model)" % md,
                    "N": "sequence length"})), use_container_width=True)


def _speedup(warm):
    st.subheader("(c) Speedup over %s" % BASELINE)
    g = ok(warm)
    gpus = [x for x in GPU_ORDER if x in set(g.gpu)]
    if not gpus:
        return _empty()
    c1, c2 = st.columns([1, 4])
    gpu = c1.selectbox("GPU", gpus, index=len(gpus) - 1, key="prefill_c_gpu")
    with c2:
        g = filter_bar(g[g.gpu == gpu], key="prefill_c",
                       fields=("dt", "D", "md", "Hkv"))
    base = (g[g.implementation == BASELINE]
            .groupby("cell", as_index=False).median_us.median()
            .rename(columns={"median_us": "base_us"}))
    m = g.merge(base, on="cell")
    if not len(m):
        return _empty()
    m = (m.assign(speedup=m.base_us / m.median_us)
         .groupby(["implementation", "N", "B"], as_index=False)
         .agg(speedup=("speedup", "median")))
    m["Nc"], m["Bc"] = m.N.astype(str), m.B.astype(str)
    st.plotly_chart(_tidy(px.density_heatmap(
        m, x="Nc", y="Bc", z="speedup", histfunc="avg", facet_col="implementation",
        facet_col_wrap=4, text_auto=".2f", color_continuous_scale="RdBu",
        color_continuous_midpoint=1.0,
        category_orders={"Nc": [str(x) for x in sorted(m.N.unique())],
                         "Bc": [str(x) for x in sorted(m.B.unique())],
                         "implementation": sorted(m.implementation.unique())},
        labels={"Nc": "sequence length", "Bc": "batch",
                "speedup": "x vs baseline"})), use_container_width=True)

    b = g[g.implementation == BASELINE].pivot_table(
        index="B", columns="N", values="median_us", aggfunc="median")
    st.caption("Absolute baseline latency, %s on %s (median us). A speedup "
               "without the time it multiplies hides that 2x at N=256 is worth "
               "microseconds while 1.2x at N=8192 is worth milliseconds."
               % (BASELINE, gpu))
    st.dataframe(b.round(1), use_container_width=True)
    note(">1 is faster than %s. Tiles take the median over anything the filters "
         "leave open; a blank tile is a configuration that was not run "
         "(UNSUPPORTED, or an allocation refused in advance), never a zero."
         % BASELINE)


def _memory(warm):
    st.subheader("(d) Peak memory and the OOM frontier")
    f = filter_bar(warm, key="prefill_d", fields=("gpu", "dt", "D", "B", "md"))
    if not len(f):
        return _empty()
    meas, pred = f[f.status == "OK"], f[f.status == "OOM_PREDICTED"]
    parts = []
    if len(meas):
        parts.append(meas.assign(mb=meas.peak_allocated_mb,
                                 mark="peak_allocated (measured)"))
    if len(pred):
        parts.append(pred.assign(mb=pred.predicted_peak_gb * 1024,
                                 mark="OOM_PREDICTED (computed, not attempted)"))
    if not parts:
        return _empty()
    m = (pd.concat(parts)
         .groupby(["gpu", "implementation", "mark", "N"], as_index=False)
         .mb.median().sort_values("N"))
    st.plotly_chart(_tidy(px.line(
        m, x="N", y="mb", color="implementation", symbol="mark", line_dash="mark",
        facet_col="gpu", markers=True, log_x=True, log_y=True,
        color_discrete_map=COLOURS, category_orders=_orders(m),
        labels={"mb": "peak memory (MB)", "N": "sequence length"})),
        use_container_width=True)
    note("Two marks because these are two different facts, not one failure "
         "surface. Solid: peak_allocated, the high-water mark of a run that "
         "finished. Dashed: OOM_PREDICTED (%d rows here) -- %s -- drawn on the "
         "same axis only to place the frontier. OOM (%s) is a third thing "
         "entirely: %d rows in this selection. UNSUPPORTED (%s) is %d rows and "
         "carries no memory number at all."
         % (len(pred), STATUS_MEANING["OOM_PREDICTED"], STATUS_MEANING["OOM"],
            int((f.status == "OOM").sum()), STATUS_MEANING["UNSUPPORTED"],
            int((f.status == "UNSUPPORTED").sum())))


def _scaling(D):
    st.subheader("(e) Empirical scaling exponent")
    sc = D["scaling"]
    if not len(sc):
        return not_collected("Scaling exponents",
                             "scaling.parquet is empty -- run "
                             "`python -m akp.analysis` first.")
    s = sc[sc.regime == "prefill"].copy()
    if not len(s):
        return not_collected("Scaling exponents",
                             "scaling.parquet holds no prefill fits.")
    q = s.series.str.split("|", expand=True)     # series is the cell key minus N
    s["gpu"] = s.gpu_name.map(short_gpu)
    s["B"] = q[1].str[1:].astype(int)
    s["Hkv"] = q[3].str[3:].astype(int)
    s["D"] = q[4].str[1:].astype(int)
    s["dt"], s["md"] = q[5], q[6]
    s = filter_bar(s, key="prefill_e", fields=("gpu", "dt", "D", "B", "md"))
    if not len(s):
        return _empty()
    full = int(s.n_points.max())
    st.caption("alpha is the slope of log(median_us) on log(N); quadratic "
               "attention gives 2. %d of %d fits use fewer than %d points, "
               "because the larger N were refused in advance or unsupported -- "
               "a low alpha there may be a short lever arm rather than better "
               "scaling. Sorted shortest fit first."
               % (int((s.n_points < full).sum()), len(s), full))
    st.dataframe(
        s[["gpu", "implementation", "B", "D", "Hkv", "dt", "md", "alpha", "r2",
           "n_points", "n_min", "n_max"]]
        .sort_values(["n_points", "gpu", "implementation"])
        .round({"alpha": 3, "r2": 4}),
        use_container_width=True, hide_index=True,
        column_config={"n_points": st.column_config.NumberColumn(
            "points in fit",
            help="fewer than %d is a truncated series" % full)})


def _asymmetry(D):
    st.subheader("(f) Forward vs forward+backward, measured")
    c = D["cells"]
    c = c[c.regime == "prefill"] if len(c) else c
    if not len(c):
        return not_collected("Forward/backward asymmetry",
                             "cells.parquet holds no prefill rows.")
    # Pair within (gpu, implementation, cell-with-mode-stripped): the ratio has
    # to be one configuration timed twice, never two configurations compared.
    base = (c.cell.str.replace("|fwd_bwd|", "|<md>|", regex=False)
            .str.replace("|fwd|", "|<md>|", regex=False))
    w = (c.assign(base=base)
         .pivot_table(index=["gpu", "implementation", "base"], columns="md",
                      values="median_us"))
    if not {"fwd", "fwd_bwd"} <= set(w.columns):
        return not_collected("Forward/backward asymmetry",
                             "only one mode was measured for prefill.")
    w = w.dropna(subset=["fwd", "fwd_bwd"]).reset_index()
    w["ratio"] = w.fwd_bwd / w.fwd
    w["N"] = w.base.str.split("|").str[5].str[1:].astype(int)
    st.plotly_chart(_tidy(px.box(
        w.sort_values("implementation"), x="implementation", y="ratio",
        color="implementation", facet_col="gpu", points="all",
        color_discrete_map=COLOURS, category_orders=_orders(w),
        labels={"ratio": "fwd_bwd / fwd median_us"})), use_container_width=True)
    st.plotly_chart(_tidy(px.line(
        (w.groupby(["gpu", "implementation", "N"], as_index=False)
         .ratio.median().sort_values("N")),
        x="N", y="ratio", color="implementation", facet_col="gpu", markers=True,
        log_x=True, color_discrete_map=COLOURS, category_orders=_orders(w),
        labels={"ratio": "fwd_bwd / fwd median_us", "N": "sequence length"})),
        use_container_width=True)
    note("%d matched pairs. This is a ratio of measured median_us within one "
         "(GPU, configuration, implementation) -- not a ratio of tflops, whose "
         "fwd_bwd numerator already assumes 3.5x the forward FLOPs, so the same "
         "ratio taken there would just return the assumption. Each box is a "
         "distribution ACROSS configurations: block_bench keeps only the median "
         "of its 30 block times, so no within-cell distribution exists." % len(w))
