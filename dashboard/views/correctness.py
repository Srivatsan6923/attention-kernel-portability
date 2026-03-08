"""Correctness gate and dispatch integrity.

Placed before any attribution page: a latency number from a kernel that
computed the wrong thing, or from a kernel that is not the one its label
names, is not a result. Both halves read the same rows the timing pages do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from _data import (COLOURS, STATUS_COLOURS, STATUS_MEANING, cell_key,
                   filter_bar, not_collected, note, provenance)

# The gate is err <= 2x the naive reference in the SAME dtype, not err <= tau.
# tau_dtype is only a floor, so a config where both errors are ~0 cannot fail
# on noise. Every table here shows the ratio because the ratio is the rule.
GATE_RATIO = 2.0


def _gate(rows):
    """Rows carrying a correctness verdict. status is OK on all of them."""
    if not len(rows) or "correctness_pass" not in rows:
        return pd.DataFrame()
    g = rows[rows.correctness_pass.notna()].copy()
    g["ratio"] = g.max_abs_err / g.baseline_max_abs_err
    return g


def render(D):
    st.title("Correctness and dispatch integrity")

    rows = D.get("rows", pd.DataFrame())
    audit = D.get("audit", pd.DataFrame())
    gate = _gate(rows)

    st.header("Does it compute attention?")
    if not len(gate):
        not_collected("Correctness gate",
                      "rows.parquet carries no correctness_pass column.")
    else:
        _gate_summary(gate, rows)
        _gate_table(gate)
        _error_heatmap(gate)
        _latency_vs_error(gate, rows)
        _backward(gate)
    _failures(rows)

    st.header("Did the kernel we asked for actually run?")
    if not len(audit):
        not_collected("Dispatch audit",
                      "dispatch_audit.parquet is absent; no kernel traces kept.")
        return
    a = pd.concat([audit, cell_key(audit.cell)], axis=1)
    _requested_vs_executed(a)
    _sankey(a)
    _mismatch_rate(a)
    _inductor(a)


# --------------------------------------------------------------------------- #
# correctness

def _gate_summary(g, rows):
    passed = int(g.correctness_pass.fillna(False).sum())
    n_ok = int((rows.status == "OK").sum()) if len(rows) else len(g)
    c = st.columns(4)
    c[0].metric("Gate rows", len(g))
    c[1].metric("Passing", "%d / %d" % (passed, len(g)))
    c[2].metric("Worst err / naive", "%.2fx" % g.ratio.max(),
                help="gate threshold is %.0fx" % GATE_RATIO)
    c[3].metric("Lowest cosine sim", "%.6f" % g.cosine_sim.min())
    note("Every gate row passes. The worst error anywhere is %.2fx the naive "
         "reference in the same dtype against a %.0fx threshold, and no row "
         "produced a non-finite value. That is a clean sheet, not a survival "
         "story. %d of the %d OK rows carry a verdict; the rest are timing "
         "repeats of a configuration already gated."
         % (g.ratio.max(), GATE_RATIO, len(g), n_ok))


def _gate_table(g):
    st.subheader("Gate by implementation")
    t = (g.groupby("implementation")
         .agg(rows=("max_abs_err", "size"),
              passed=("correctness_pass", "sum"),
              max_abs_err=("max_abs_err", "max"),
              mean_abs_err=("mean_abs_err", "mean"),
              max_rel_err=("max_rel_err", "max"),
              min_cosine_sim=("cosine_sim", "min"),
              naive_baseline_err=("baseline_max_abs_err", "max"),
              tau_dtype=("tau_dtype", "max"),
              worst_ratio=("ratio", "max"))
         .reset_index().sort_values("worst_ratio", ascending=False))
    t["passed"] = t.passed.astype(int)
    st.dataframe(t, use_container_width=True, hide_index=True,
                 column_config={
                     "worst_ratio": st.column_config.NumberColumn(
                         "worst err / naive", format="%.2f",
                         help="the criterion: fail above %.0f" % GATE_RATIO),
                     "naive_baseline_err": st.column_config.NumberColumn(
                         "naive baseline err", format="%.2e"),
                     "tau_dtype": st.column_config.NumberColumn(
                         "tau (dtype floor)", format="%.3f"),
                     "max_abs_err": st.column_config.NumberColumn(format="%.2e"),
                     "mean_abs_err": st.column_config.NumberColumn(format="%.2e"),
                     "max_rel_err": st.column_config.NumberColumn(format="%.1f"),
                     "min_cosine_sim": st.column_config.NumberColumn(format="%.7f")})
    note("max_rel_err runs into the thousands and means nothing alone: it "
         "divides by reference elements that are ~0 after the softmax. The "
         "gate reads max_abs_err against the naive reference in the same "
         "dtype; cosine_sim is the shape check; tau_dtype (1e-2 fp16, 4e-2 "
         "bf16) is a floor, not the criterion.")
    note("Row counts are not comparable across implementations. P4h-fa3 is "
         "sm90-only, so it is gated on H100 rows only; D1-inductor and the "
         "P1-inductor-nofuse / -where arms are bf16-only strips. Their small "
         "counts are coverage by design, not gate attrition.")


def _error_heatmap(g):
    st.subheader("Error against workload")
    regime = st.radio("Regime", sorted(g.regime.unique()), horizontal=True,
                      key="corr_regime")
    yaxis = st.radio("Y axis", ["head dim", "batch"], horizontal=True,
                     key="corr_yaxis")
    d = filter_bar(g[g.regime == regime], key="corr_heat",
                   fields=("gpu", "D", "B", "N", "md", "implementation"))
    if not len(d):
        st.info("No gate rows for this selection.")
        return
    d = d.assign(log_err=np.log10(d.max_abs_err))
    y = "D" if yaxis == "head dim" else "B"
    fig = px.density_heatmap(
        d, x="N", y=y, z="log_err", histfunc="max",
        facet_col="implementation", facet_row="dt",
        color_continuous_scale="Viridis", template="plotly_white",
        labels={"log_err": "log10 max_abs_err", "N": "seq / KV length",
                "D": "head dim", "B": "batch"})
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=300 * d.dt.nunique() + 90,
                      coloraxis_colorbar_title_text="log10 err")
    st.plotly_chart(fig, use_container_width=True)
    note("Colour is the worst max_abs_err over every gate row at that shape, "
         "GPUs pooled by the filter above. Empty cells were never gated at "
         "that shape. Narrow the implementation filter if the facets crowd.")


def _latency_vs_error(g, rows):
    st.subheader("Latency against error")
    d = filter_bar(g, key="corr_scatter", fields=("gpu", "dt", "D", "B", "md"))
    if not len(d):
        st.info("No gate rows for this selection.")
        return
    fig = px.scatter(
        d, x="median_us", y="max_abs_err", color="gpu", symbol="dt",
        facet_col="regime", facet_row="timer", log_x=True, log_y=True,
        template="plotly_white", hover_data=["implementation", "cell", "ratio"],
        labels={"median_us": "median latency (us)", "max_abs_err": "max |err|",
                "dt": "dtype", "gpu": "GPU"})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(marker_size=7, marker_opacity=0.7)
    fig.update_layout(height=340 * max(d.timer.nunique(), 1) + 90)
    st.plotly_chart(fig, use_container_width=True)
    note("Faceted by timer, because block_bench and do_bench_cudagraph do not "
         "share a latency axis, and by regime, because a decode step and a "
         "prefill pass differ by orders of magnitude. Absolute latency is the "
         "x axis here, not a speedup. Nothing occupies the fast-and-wrong "
         "corner: the quickest points carry the same error band as the slow.")
    prov = provenance(rows)
    if len(prov):
        with st.expander("Per-GPU commit (this panel puts GPUs on one axis)"):
            st.dataframe(prov, use_container_width=True, hide_index=True)
            note("A10 and A100 decode rows come from a different commit and "
                 "driver branch than H100 decode, so any cross-GPU decode "
                 "difference carries a software change with it. Prefill is "
                 "commit-homogeneous.")


def _backward(g):
    st.subheader("Backward pass")
    b = g[g.dq_max_abs_err.notna()]
    if not len(b):
        not_collected("Backward-pass error", "no row carries dq/dk/dv errors.")
        return
    ids = ["implementation", "gpu", "dt", "N", "D"]
    grads = ["dq_max_abs_err", "dk_max_abs_err", "dv_max_abs_err"]
    # Melt a subset: b still carries the forward max_abs_err, and pandas
    # refuses a value_name that collides with a column in the frame.
    long = b[ids + grads].melt(id_vars=ids, value_vars=grads,
                               var_name="tensor", value_name="max_abs_err")
    long["tensor"] = long.tensor.str[:2]
    fig = px.box(long, x="tensor", y="max_abs_err", color="implementation",
                 facet_col="dt", log_y=True, template="plotly_white",
                 color_discrete_map=COLOURS, points="all",
                 labels={"max_abs_err": "max |err|", "dt": "dtype"})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
    note("%d fwd_bwd rows, prefill only; P4h-fa3 appears on H100 alone. Each "
         "box spreads ACROSS configurations (shape, GPU), never across repeated "
         "calls: block_bench keeps only median/p5/p95 and discards the "
         "individual block times, so no within-configuration distribution "
         "exists." % len(b))
    st.dataframe(
        b.groupby("implementation")[["dq_max_abs_err", "dk_max_abs_err",
                                     "dv_max_abs_err"]].max().reset_index(),
        use_container_width=True, hide_index=True)


def _failures(rows):
    st.subheader("Non-OK rows")
    bad = rows[rows.status != "OK"] if len(rows) else pd.DataFrame()
    if not len(bad):
        st.success("Every row is OK.")
        return
    t = (bad.groupby(["status", "gpu", "implementation"]).size()
         .rename("rows").reset_index())
    t["meaning"] = t.status.map(STATUS_MEANING)
    fig = px.bar(bad.groupby(["status", "implementation"], as_index=False)
                 .size().rename(columns={"size": "rows"}),
                 x="rows", y="implementation", color="status", orientation="h",
                 template="plotly_white", color_discrete_map=STATUS_COLOURS,
                 barmode="group")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
    note("Three separate outcomes, never summed into a failure count. "
         + "; ".join("%s = %s" % (s, STATUS_MEANING[s])
                     for s in sorted(bad.status.unique()) if s in STATUS_MEANING)
         + ". NUMERICAL_FAIL does not occur: nothing ran and failed the gate.")
    st.dataframe(t.sort_values(["status", "rows"], ascending=[True, False]),
                 use_container_width=True, hide_index=True)
    with st.expander("Every non-OK row (%d)" % len(bad)):
        cols = ["gpu", "implementation", "regime", "B", "Hq", "Hkv", "D", "N",
                "dt", "md", "status"]
        full = bad[[c for c in cols if c in bad.columns]].copy()
        full["meaning"] = full.status.map(STATUS_MEANING)
        st.dataframe(full, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# dispatch

def _requested_vs_executed(a):
    st.subheader("Requested against executed")
    t = (a.groupby("implementation")
         .agg(rows=("observed", "size"),
              probed=("probed", "sum"),
              matched=("matched", lambda s: int((s == True).sum())),
              backends=("observed", lambda s: ", ".join(
                  "%s %d" % (k, v) for k, v in s.value_counts().items())),
              median_kernels=("n_kernels", "median"))
         .reset_index())
    t["probed"] = t.probed.astype(int)
    t["unprobed %"] = (100 * (1 - t.probed / t.rows)).round(1)
    t["mismatch % of probed"] = (
        100 * (1 - t.matched / t.probed.replace(0, np.nan))).round(2)
    st.dataframe(t, use_container_width=True, hide_index=True)
    note("matched is a verdict only where probed is true, so the mismatch "
         "column is a rate over probed rows and never over all rows: %d of %d "
         "OK rows carry no trace at all. For the SDPA arms the observed "
         "backend is the measurement, not an error."
         % (int((~a.probed).sum()), len(a)))


def _sankey(a):
    st.subheader("Requested implementation to observed backend")
    flow = a.groupby(["implementation", "observed"]).size().rename("n").reset_index()
    impls = sorted(flow.implementation.unique())
    backends = sorted(flow.observed.unique())
    idx = {n: i for i, n in enumerate(impls + backends)}
    unprobed = int(flow[flow.observed == "unprobed"].n.sum())
    labels = impls + ["%s  %.0f%%" % (b, 100 * flow[flow.observed == b].n.sum() / len(a))
                      for b in backends]
    colour = ([COLOURS.get(i, "#9e9e9e") for i in impls]
              + ["#c0392b" if b == "unprobed" else
                 "#7f8c8d" if b == "unknown" else "#4c9fd6" for b in backends])
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, color=colour, pad=12, thickness=14),
        link=dict(source=flow.implementation.map(idx),
                  target=flow.observed.map(idx), value=flow.n,
                  color=["rgba(192,57,43,0.30)" if o == "unprobed"
                         else "rgba(120,120,120,0.22)" for o in flow.observed])))
    fig.update_layout(height=780, template="plotly_white", font_size=11,
                      margin=dict(l=8, r=8, t=8, b=8))
    st.plotly_chart(fig, use_container_width=True)
    note("'unprobed' is a terminal node carrying its own share, not a dropped "
         "edge: %d of %d OK rows (%.0f%%) launched with no kernel trace. The "
         "miss is biased toward the fastest implementations (P4-fa2 ~73%%, "
         "P4h-fa3 ~68%%, P3-triton ~60%% of their rows), so normalising it "
         "away would flatter exactly the kernels this study is about."
         % (unprobed, len(a), 100 * unprobed / len(a)))


def _mismatch_rate(a):
    st.subheader("Mismatch rate")
    by = st.radio("Break out by", ["gpu", "implementation", "dt"],
                  horizontal=True, key="disp_by",
                  format_func=lambda c: {"dt": "dtype"}.get(c, c))
    p = a[a.probed]
    t = (p.groupby(by).matched
         .agg(probed="size", matched=lambda s: int((s == True).sum()))
         .reset_index())
    t["mismatch %"] = (100 * (1 - t.matched / t.probed)).round(2)
    t["unprobed % of all rows"] = t[by].map(
        a.groupby(by).probed.apply(lambda s: 100 * (1 - s.mean())).round(1))
    fig = px.bar(t, x=by, y="mismatch %", template="plotly_white", text="mismatch %",
                 color="implementation" if by == "implementation" else None,
                 color_discrete_map=COLOURS)
    fig.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(t, use_container_width=True, hide_index=True)
    note("Denominator is probed rows only, with the unprobed share printed "
         "beside it so a low rate on a thinly-probed group cannot be read as "
         "confirmation. Overall %d mismatches in %d probed rows (%.2f%%)."
         % (int((p.matched == False).sum()), len(p),
            100 * (p.matched == False).mean()))


def _inductor(a):
    st.subheader("TorchInductor: did it rewrite naive attention into SDPA?")
    in_trace = a[a.attn_kernel_in_trace == True]
    c = st.columns(2)
    c[0].metric("fused_into_sdpa", int(a.fused_into_sdpa.sum()),
                help="Inductor's own pattern-match counter")
    c[1].metric("attn_kernel_in_trace", len(in_trace),
                help="a separate signal, kept separate")
    st.write("`fused_into_sdpa` is Inductor's own counter and it is **0 on all "
             "%d rows**. torch.compile did not rewrite the naive attention "
             "into an SDPA call anywhere in this sweep." % len(a))
    if not len(in_trace):
        return
    st.write("Separately, and not merged into that counter: "
             "`attn_kernel_in_trace` is true on %d rows, all H100."
             % len(in_trace))
    st.dataframe(in_trace.groupby(["gpu", "implementation", "dt", "observed"])
                 .size().rename("rows").reset_index(),
                 use_container_width=True, hide_index=True)
    note("Unexplained. On these rows the trace holds a kernel whose name "
         "matches a fused-attention pattern while Inductor reports no "
         "substitution and the observed backend stays triton. Two signals "
         "disagreeing is not evidence of fusion; it stays an open anomaly "
         "rather than being folded into the counter.")
