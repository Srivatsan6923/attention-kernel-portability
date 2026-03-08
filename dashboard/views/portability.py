"""Page 6 -- cross-architecture portability and the backend selector.

Nothing here computes a metric. The ratios drawn are the ones analysis.py
already stored (inversions.r1/r2, portability.p_ratio) or a ratio of two stored
medians; the selector numbers are read verbatim out of selector.json. Where a
panel the page wants would need a quantity analysis.py never wrote, it says so
instead of re-deriving it here, because a dashboard that re-fits the tree can
disagree with the report that quotes it.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from _data import (COLOURS, filter_bar, gpu_meta, not_collected, note,
                   provenance, short_gpu)

# Arms that are a deliberate strip, not a defeat: showing them beside a real
# backend without this makes an unsupported configuration look like a loss.
STRIPPED = {
    "P4h-fa3": "sm90 only -- UNSUPPORTED on every A100 prefill row, never ran on A10",
    "D1-inductor": "~96% UNSUPPORTED",
    "P1-inductor-nofuse": "~91% UNSUPPORTED (bf16-only strip)",
    "P1-inductor-where": "~91% UNSUPPORTED (bf16-only strip)",
}


def _prov(D, key):
    """Trap 6. Every cross-GPU panel carries this: A10/A100 decode rows come
    from commit 468b9a55 and H100 decode from 91959d34, so a cross-GPU decode
    delta is architecture plus software. Prefill is commit-homogeneous."""
    rows = D.get("rows")
    if rows is None or not len(rows):
        return
    # rows carries 'regime' twice (native column + cell_key parse); the groupby
    # inside provenance() rejects the duplicate.
    p = provenance(rows.loc[:, ~rows.columns.duplicated()])
    if not len(p):
        return
    with st.expander("provenance -- which commit produced which rows", expanded=False):
        st.dataframe(p, use_container_width=True, hide_index=True)
        for regime, n in p.groupby("regime").commit.nunique().items():
            st.caption(":grey[%s: %s]" % (
                regime,
                "commit-homogeneous" if n == 1 else
                "**%d commits** -- every cross-GPU %s delta carries a software "
                "difference alongside the architecture one" % (n, regime)))


def _pair_label(a, b):
    return "%s / %s" % (short_gpu(a), short_gpu(b))


def _lookup(lut, gpu, cell, impl):
    return lut.reindex(pd.MultiIndex.from_arrays([gpu, cell, impl])).values


# --------------------------------------------------------------------------- #
# (a) matched-cell speedup scatter

def _scatter(D):
    st.subheader("(a) Matched-cell speedup, GPU against GPU")
    inv = D.get("inversions")
    if inv is None or not len(inv):
        not_collected("Matched-cell speedup",
                      "inversions.parquet is empty -- no GPU pair had cells on "
                      "both sides.")
        return
    inv = inv.assign(regime=inv.cell.str.split("|").str[0],
                     pair=[_pair_label(a, b) for a, b in zip(inv.gpu_a, inv.gpu_b)])
    pair = st.selectbox("GPU pair", sorted(inv.pair.unique()), key="port_a_pair")
    d = inv[inv.pair == pair].copy()

    # Constraint 16: the ratio never travels without the latencies behind it.
    lut = D["cells"].set_index(["gpu_name", "cell", "implementation"]).median_us
    for side, g in (("1", "gpu_a"), ("2", "gpu_b")):
        d["a_us" + side] = _lookup(lut, d[g], d.cell, d.a)
        d["b_us" + side] = _lookup(lut, d[g], d.cell, d.b)

    g1, g2 = short_gpu(d.gpu_a.iloc[0]), short_gpu(d.gpu_b.iloc[0])
    fig = px.scatter(
        d, x="r1", y="r2", color="practical", facet_col="regime",
        log_x=True, log_y=True, opacity=0.7,
        category_orders={"regime": ["prefill", "decode"]},
        color_discrete_map={True: "#c0392b", False: "#9e9e9e"},
        hover_data={"cell": True, "a": True, "b": True, "sig": True,
                    "a_us1": ":.1f", "b_us1": ":.1f",
                    "a_us2": ":.1f", "b_us2": ":.1f"},
        labels={"r1": "%s: median_us(A) / median_us(B)" % g1,
                "r2": "%s: median_us(A) / median_us(B)" % g2,
                "practical": ">=10% both sides"})
    fig.add_vline(x=1, line_dash="dot", line_color="#888")
    fig.add_hline(y=1, line_dash="dot", line_color="#888")
    fig.update_layout(height=430, legend_title_text="practical")
    st.plotly_chart(fig, use_container_width=True)

    fam = int(d.pairs_examined.max())
    note("inversions.parquet stores only the pairs whose ordering already flips, so "
         "every point here is off-diagonal by construction -- the agreeing majority "
         "of the %d (cell, A, B) comparisons on this pair is not in the frame and "
         "cannot be drawn. %d flips stored, %d of them practical."
         % (fam, len(d), int(d.practical.sum())))
    note("Prefill and decode are faceted, never pooled: they are different cell "
         "families and a shared axis invites reading one against the other.")
    _prov(D, "a")


# --------------------------------------------------------------------------- #
# (b) rank correlation

def _rank_matrix(D):
    st.subheader("(b) Rank correlation between GPUs")
    rc = (D.get("summary") or {}).get("rank_correlation") or {}
    if not rc:
        not_collected("Rank correlation",
                      "summary.json has no 'rank_correlation' block.")
        return
    names = sorted({short_gpu(g) for k in rc for g in k.split(" vs ")})
    stat = st.radio("statistic", ["spearman_median", "kendall_median"],
                    horizontal=True, key="port_b_stat")
    m = pd.DataFrame(np.nan, index=names, columns=names)
    for k, v in rc.items():
        a, b = (short_gpu(g) for g in k.split(" vs "))
        m.loc[a, b] = m.loc[b, a] = v.get(stat)
    for n in names:
        m.loc[n, n] = 1.0                     # a GPU ranks identically to itself

    fig = px.imshow(m, text_auto=".3f", zmin=0, zmax=1,
                    color_continuous_scale="Blues", aspect="equal")
    fig.update_layout(height=380, coloraxis_colorbar_title=stat.split("_")[0])
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        pd.DataFrame([{"pair": k, "spearman": v.get("spearman_median"),
                       "kendall": v.get("kendall_median"),
                       "cells": v.get("n_cells"),
                       "practical inversion rate": v.get("practical_inversion_rate")}
                      for k, v in rc.items()]),
        use_container_width=True, hide_index=True)
    note("Median over per-cell correlations, so no FLOP axis is shared between "
         "regimes and no single cell dominates.")
    not_collected(
        "Rank correlation split by prefill / batch-1 decode / batched decode",
        "summary['rank_correlation'] is keyed by GPU pair only -- analysis.py "
        "medians over every shared cell of both regimes and writes one number per "
        "pair. Three separate matrices would need a per-regime, per-batch Spearman "
        "that was never written, and re-deriving it here would let the dashboard "
        "and the report disagree.")
    _prov(D, "b")


# --------------------------------------------------------------------------- #
# (c) inversion rate

def _inversion_rate(D):
    st.subheader("(c) Inversion rate -- practical against statistical")
    inv = D.get("inversions")
    if inv is None or not len(inv):
        not_collected("Inversion rate", "inversions.parquet is empty.")
        return
    inv = inv.assign(pair=[_pair_label(a, b) for a, b in zip(inv.gpu_a, inv.gpu_b)])
    g = inv.groupby("pair").agg(family=("pairs_examined", "max"),
                                flips=("cell", "size"),
                                statistical=("sig", "sum"),
                                practical=("practical", "sum")).reset_index()
    g["practical rate"] = g.practical / g.family
    g["statistical rate"] = g.statistical / g.family

    for c, (_, r) in zip(st.columns(len(g)), g.iterrows()):
        c.metric("%s -- practical" % r["pair"], int(r.practical),
                 "%.2f%% of %d compared pairs" % (100 * r["practical rate"], r.family),
                 delta_color="off")
        c.caption("statistical: %d (%.1f%%)"
                  % (r.statistical, 100 * r["statistical rate"]))
    st.dataframe(g, use_container_width=True, hide_index=True)

    note("Practical = both ratios past 1.10 (analysis.PRACTICAL). It leads because "
         "at a family of ~4000 comparisons per pair a nominal alpha manufactures "
         "inversions out of noise.")
    st.warning(
        "The stored `sig` column is **not** multiplicity-corrected. It is 'both "
        "bootstrap CIs exclude 1' at the nominal level; analysis.inversions takes an "
        "`fdr=0.05` argument but never applies it, so no FDR-corrected count exists "
        "in the processed data. Read the statistical column against the family size "
        "printed beside it, not as a corrected count.",
        icon=":material/warning:")
    _prov(D, "c")


# --------------------------------------------------------------------------- #
# (d) performance portability ratio

def _portability(D):
    st.subheader("(d) Performance portability P per implementation")
    p = D.get("portability")
    if p is None or not len(p):
        not_collected("Performance portability",
                      "portability.parquet is empty -- stage-A portability was "
                      "not run.")
        return
    p = p.assign(gpu=p.gpu_name.map(short_gpu))
    view = st.radio("view", ["box", "ECDF"], horizontal=True, key="port_d_view")
    for regime in ("prefill", "decode"):
        d = p[p.regime == regime]
        if not len(d):
            continue
        order = sorted(d.implementation.unique())
        if view == "box":
            fig = px.box(d, x="implementation", y="p_ratio", color="implementation",
                         color_discrete_map=COLOURS, points="outliers",
                         category_orders={"implementation": order},
                         hover_data={"gpu": True, "median_us": ":.1f",
                                     "best_us": ":.1f", "cell": True},
                         labels={"p_ratio": "P = best_us / median_us"})
        else:
            fig = px.ecdf(d, x="p_ratio", color="implementation",
                          color_discrete_map=COLOURS,
                          category_orders={"implementation": order},
                          labels={"p_ratio": "P = best_us / median_us"})
        fig.update_layout(height=400, showlegend=(view == "ECDF"),
                          title="%s -- %d (gpu, cell, implementation) points"
                                % (regime, len(d)))
        st.plotly_chart(fig, use_container_width=True)

    note("P = 1 means this backend was the fastest measured one on that GPU and "
         "cell; P = 0.5 means it cost twice the best. The distribution is ACROSS "
         "cells and GPUs -- block_bench keeps only median/p5/p95 per cell, so no "
         "within-cell latency distribution exists anywhere in this data.")
    note("Absolute latency travels with the ratio: median_us and best_us are on "
         "every hover point in the box view.")
    cov = p.groupby("implementation").size()
    strip = [i for i in STRIPPED if i in cov.index]
    if strip:
        st.caption(":grey[Coverage caveat -- %s]" % "; ".join(
            "**%s** %d cells, %s" % (i, cov[i], STRIPPED[i]) for i in strip))
    _prov(D, "d")


# --------------------------------------------------------------------------- #
# (e) normalized hardware scaling

def _hw_scaling(D):
    st.subheader("(e) Observed speedup against bandwidth and SM ratios")
    cells = D.get("cells")
    summary = D.get("summary") or {}
    # gpu_meta sorts on a column it only creates when 'environment' is populated.
    meta = gpu_meta(summary) if summary.get("environment") else pd.DataFrame()
    if cells is None or not len(cells) or not len(meta):
        not_collected("Hardware scaling",
                      "cells.parquet or the environment manifest is missing.")
        return
    c = filter_bar(cells, key="port_e", fields=("dt", "D", "md", "N"))
    c = c[["gpu", "cell", "implementation", "median_us", "regime"]]
    m = c.merge(c, on=["cell", "implementation"], suffixes=("_a", "_b"))
    m = m[m.gpu_a < m.gpu_b]                  # each unordered pair once
    if not len(m):
        st.info("No cell survives the filter on two GPUs at once.")
        return
    m["speedup"] = m.median_us_a / m.median_us_b
    m["pair"] = m.gpu_a + " -> " + m.gpu_b

    mt = meta.set_index("gpu")
    ratios = []
    for pair, sub in m.groupby("pair"):
        a, b = pair.split(" -> ")
        if a not in mt.index or b not in mt.index:
            continue
        ratios.append({"pair": pair, "n": len(sub),
                       "bandwidth ratio": mt.measured_bw_gbs[b] / mt.measured_bw_gbs[a],
                       "SM ratio": mt.sm_count[b] / mt.sm_count[a],
                       "median speedup": float(sub.speedup.median()),
                       "median us, first": float(sub.median_us_a.median()),
                       "median us, second": float(sub.median_us_b.median())})
    ratios = pd.DataFrame(ratios)

    regimes = [r for r in ("prefill", "decode") if (m.regime_a == r).any()]
    fig = px.box(m, x="pair", y="speedup", color="implementation",
                 facet_col="regime_a", log_y=True, color_discrete_map=COLOURS,
                 category_orders={"regime_a": regimes,
                                  "pair": sorted(m.pair.unique())},
                 hover_data={"median_us_a": ":.1f", "median_us_b": ":.1f",
                             "cell": True},
                 labels={"speedup": "median_us(first GPU) / median_us(second)",
                         "regime_a": "regime"})
    if len(ratios):
        for j in range(len(regimes)):
            for col, colour in (("bandwidth ratio", "#1f6fb2"), ("SM ratio", "#c98a3a")):
                fig.add_trace(go.Scatter(
                    x=ratios.pair, y=ratios[col], mode="markers", name=col,
                    marker=dict(symbol="line-ew", size=26,
                                line=dict(width=3, color=colour)),
                    showlegend=(j == 0), legendgroup=col), row=1, col=j + 1)
    fig.update_layout(height=460)
    st.plotly_chart(fig, use_container_width=True)
    if len(ratios):
        st.dataframe(ratios, use_container_width=True, hide_index=True)

    note("A box sitting on the bandwidth marker means the workload tracks memory "
         "bandwidth on that pair; one sitting on the SM marker means it tracks "
         "compute width. measured_peak_bw_gbs is an achieved 512MB device-to-device "
         "copy figure, not a theoretical peak -- theoretical_bw_gbs is null on all "
         "three GPUs -- so the bandwidth marker is itself an underestimate.")
    note("Absolute medians on both sides are in the hover and in the table; the "
         "speedup never appears alone.")
    note("Cells aggregate the rows usable() kept, which deliberately includes "
         "SwPowerCap rows (~20%, unevenly distributed by GPU and size). A pair's "
         "speedup carries that unevenness.")
    _prov(D, "e")


# --------------------------------------------------------------------------- #
# selector half

def _selector_table(sel):
    st.subheader("(f) Selector against fixed-backend baselines")
    a, b, c, d = st.columns(4)
    a.metric("median regret", "%.3f" % sel["median_regret"])
    b.metric("p95 regret", "%.3f" % sel["p95_regret"])
    c.metric("held-out cells", sel["n_heldout"])
    d.metric("top-1 accuracy", "%.1f%%" % (100 * sel["accuracy"]))
    note("Regret is the headline. Top-1 accuracy of %.1f%% against a median regret "
         "of %.3f says the tree usually picks a backend that is a near-tie with the "
         "winner, not that it picks badly -- accuracy punishes a 2%% miss exactly as "
         "hard as a 3x one."
         % (100 * sel["accuracy"], sel["median_regret"]))

    vs = sel.get("vs_fixed") or {}
    if not vs:
        st.info("selector.json has no vs_fixed block.")
        return
    t = pd.DataFrame(
        [{"policy": "oracle (per-cell best)", "median": 1.0, "p95": 1.0,
          "coverage": 1.0, "units": "regret vs oracle"},
         {"policy": "the rule (tree)", "median": sel["median_regret"],
          "p95": sel["p95_regret"], "coverage": 1.0, "units": "regret vs oracle"}]
        + [{"policy": "always " + k, "median": v["median_speedup"],
            "p95": v["p95_speedup"], "coverage": v["coverage"],
            "units": "rule's speedup over this fixed policy"}
           for k, v in sorted(vs.items())])
    st.dataframe(t, use_container_width=True, hide_index=True)
    st.warning(
        "The two blocks are in **different units** and must not be read down one "
        "column. analysis.selector stores the rule's regret against the oracle, but "
        "for a fixed policy it stores only median(fixed / chosen) -- the fixed "
        "policies' own regret against the oracle was never written, and a ratio of "
        "medians is not the median of a ratio, so it cannot be recovered here.",
        icon=":material/warning:")
    note("vs_fixed lists decode backends only. analysis.selector drops any policy "
         "that is NaN on more than half the held-out cells, and the held-out device "
         "is the A10 (335 cells, 191 decode / 144 prefill), so every prefill policy "
         "lands at coverage 0.43 and is dropped. The prefill arms were measured; "
         "they failed a coverage rule, they are not missing. 'Always-FA2' here is "
         "the decode D3-fa-kvcache, and 'always-SDPA' the decode D2-sdpa; there is "
         "no prefill P4-fa2 or P2c-sdpa-flash row in this table.")


def _regret_ecdf(sel):
    st.subheader("(g) Regret over held-out cells")
    not_collected(
        "Regret ECDF",
        "selector.json stores only the median (%.3f) and p95 (%.3f) of "
        "chosen/oracle over the %d held-out cells -- analysis.selector never writes "
        "the per-cell regret vector. An ECDF cannot be drawn from two quantiles, and "
        "re-fitting the tree in the dashboard to regenerate them would let this page "
        "and the report quote different numbers."
        % (sel["median_regret"], sel["p95_regret"], sel["n_heldout"]))
    fig = go.Figure(go.Scatter(x=[sel["median_regret"], sel["p95_regret"]],
                               y=[0.5, 0.95], mode="markers+text",
                               text=["median", "p95"], textposition="top center",
                               marker=dict(size=12, color="#1f6fb2")))
    fig.update_layout(height=280, xaxis_title="regret (chosen / oracle latency)",
                      yaxis_title="fraction of held-out cells",
                      xaxis_range=[0.98, 1.45], yaxis_range=[0, 1])
    fig.add_vline(x=1.0, line_dash="dot", line_color="#888")
    st.plotly_chart(fig, use_container_width=True)
    note("The two stored quantiles, drawn as the only two points of the ECDF that "
         "exist. The curve between and beyond them is unknown.")


def _decision_regions(D, sel):
    st.subheader("(h) Decision regions over the held-out GPU")
    cells = D.get("cells")
    if cells is None or not len(cells):
        not_collected("Decision regions", "cells.parquet is empty.")
        return
    # The held-out GPU is whichever one analysis.FIT_GPUS did not match. It is
    # identifiable from the processed data alone: n_heldout is its cell count.
    counts = cells.groupby("gpu").cell.nunique()
    hold = counts[counts == sel.get("n_heldout")]
    if not len(hold):
        not_collected("Decision regions",
                      "no GPU has exactly n_heldout=%s cells, so the held-out "
                      "device cannot be identified from the processed data."
                      % sel.get("n_heldout"))
        return
    g = hold.index[0]
    d = cells[cells.gpu == g]
    win = d.loc[d.groupby("cell").median_us.idxmin()]

    fig = px.scatter(win, x="N", y="B", color="implementation", facet_col="regime",
                     log_x=True, log_y=True, color_discrete_map=COLOURS,
                     category_orders={"regime": ["prefill", "decode"]},
                     hover_data={"cell": True, "median_us": ":.1f", "dt": True,
                                 "md": True},
                     labels={"N": "seq / KV length", "B": "batch",
                             "median_us": "winner median_us"})
    fig.update_traces(marker=dict(size=13, opacity=0.75))
    fig.update_layout(height=440)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This is the **actual** per-cell winner on the held-out %s, not the tree's "
        "selection. selector.json stores the rule text and the aggregate scores but "
        "not its per-cell prediction, so neither the selected-implementation "
        "colouring nor the disagreement markers can be drawn without re-fitting the "
        "tree here. Read this map against the rule printed verbatim below." % g,
        icon=":material/info:")
    note("%d cells on %s. FA3 never ran on this device and D1-inductor is ~96%% "
         "UNSUPPORTED, so their absence from the map is a strip, not a defeat. "
         "Winner latency in us is on every hover point." % (len(win), g))
    note("Cells differing only in dtype or mode overplot at the same (N, B); the "
         "hover carries both.")


def _rule(sel):
    st.subheader("(i) The rule, verbatim")
    st.code(sel.get("rule", ""), language="text")
    note("sklearn export_text at max_depth=4, fit on the A100 and H100 cells and "
         "tested on the held-out device. Several splits lead to the same class on "
         "both sides -- the tree kept them because the split was pure enough on the "
         "training GPUs, and they are where the top-1 accuracy leaks.")


# --------------------------------------------------------------------------- #

def render(D):
    st.title("Cross-architecture portability and the selector")
    st.caption("Does a ranking measured on one GPU hold on another, and can a rule "
               "pick the backend for a device it never saw?")

    _scatter(D)
    st.divider()
    _rank_matrix(D)
    st.divider()
    _inversion_rate(D)
    st.divider()
    _portability(D)
    st.divider()
    _hw_scaling(D)
    st.divider()

    st.header("Selector")
    sel = D.get("selector") or {}
    if "median_regret" not in sel:
        not_collected("Backend selector",
                      "selector.json holds %r -- the tree needs at least one fit "
                      "GPU and one held-out GPU."
                      % sel.get("error", "no result"))
        return
    _selector_table(sel)
    st.divider()
    _regret_ecdf(sel)
    st.divider()
    _decision_regions(D, sel)
    st.divider()
    _rule(sel)
