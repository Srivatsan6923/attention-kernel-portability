"""Page 1: the results, and what they are not.

Every number here is either read from summary.json/selector.json or recomputed
from cells.parquet the way akp.analysis computes it. The one thing this page
derives that analysis.py does not write is the inversion rate split by regime --
summary.json carries a single pooled rate per GPU pair, and prefill inverts
about three times as often as decode, so the pooled figure buries the finding.
_inversion_rates() reconstructs analysis.inversions()' own denominator and the
self-check at the bottom asserts it reproduces the pooled rate.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from _data import (COLOURS, STATUS_MEANING, filter_bar, not_collected, note,
                   provenance)

BASE = "P2c-sdpa-flash"          # the ranking baseline the study is written against
MHA, GQA = 32, 8                 # Hkv values; Hq is 32 everywhere

# Coverage that a winner map or a regret table would otherwise misread as a
# defeat. Each of these is a deliberate strip or an architecture gate.
STRIPS = {
    "P4h-fa3": "sm90 only, UNSUPPORTED on every A100 prefill row and never ran on A10",
    "D1-inductor": "~96% UNSUPPORTED, a deliberate strip",
    "P1-inductor-nofuse": "~91% UNSUPPORTED, bf16-only arm",
    "P1-inductor-where": "~91% UNSUPPORTED, bf16-only arm",
}


def _flat(df):
    """load() concatenates the parsed cell fields onto rows, which already ships
    a regime column, so rows carries it twice and any groupby on it raises."""
    return df.loc[:, ~df.columns.duplicated()]


def _winners(cells):
    return cells.loc[cells.groupby(["gpu", "cell"]).median_us.idxmin()]


def _inversion_rates(cells, inv):
    """Practical inversions over pairs examined, per regime, pooled over GPU pairs.

    Denominator counted as analysis.inversions() counts it: every unordered pair
    of implementations that both GPUs ran inside a shared cell.
    """
    if not len(inv) or not len(cells):
        return {}
    per_gpu = cells.groupby(["cell", "gpu_name"]).implementation.agg(set).unstack()
    den = {"prefill": 0, "decode": 0}
    for ga, gb in inv[["gpu_a", "gpu_b"]].drop_duplicates().itertuples(index=False):
        if ga not in per_gpu or gb not in per_gpu:
            continue
        k = per_gpu[[ga, gb]].dropna()
        n = pd.Series([len(a & b) for a, b in zip(k[ga], k[gb])], index=k.index)
        pairs = n * (n - 1) // 2
        reg = pd.Series(k.index.str.split("|").str[0], index=k.index)
        for r in den:
            den[r] += int(pairs[reg == r].sum())
    num = inv.groupby(inv.cell.str.split("|").str[0]).practical.sum()
    return {r: (float(num.get(r, 0)) / den[r], den[r]) for r in den if den[r]}


def _max_speedup(cells):
    """Best non-baseline prefill implementation over the baseline, same cell.

    Pairing inside one cell holds mode, launch and therefore the timer fixed:
    the eager and cudagraph arms are timed by different harnesses and must not
    be divided into each other.
    """
    p = cells[cells.regime == "prefill"]
    b = p[p.implementation == BASE].set_index(["gpu", "cell"]).median_us
    o = p[p.implementation != BASE]
    if not len(b) or not len(o):
        return None
    o = o.loc[o.groupby(["gpu", "cell"]).median_us.idxmin()].set_index(["gpu", "cell"])
    j = o.join(b.rename("base_us"), how="inner")
    if not len(j):
        return None
    return j.assign(sp=j.base_us / j.median_us).sort_values("sp").iloc[-1]


def _flip_cuts(cells):
    """Winner-change rate over the cells every GPU ran, cut four ways.

    Restricted to the fully shared cells for the same reason winner_flips() is:
    a cell only two GPUs reached cannot disagree three ways.
    """
    w = _winners(cells)
    shared = w.groupby("cell").gpu.nunique()
    w = w[w.cell.isin(shared[shared == shared.max()].index)]
    k = w.groupby("cell").implementation.nunique()
    f = cells.drop_duplicates("cell").set_index("cell").loc[k.index]
    f = f.assign(flip=(k > 1).values)
    dec = f[f.regime == "decode"]
    groups = [
        ("regime", "prefill", f[f.regime == "prefill"]),
        ("regime", "decode", f[f.regime == "decode"]),
        ("decode batch", "batch 1", dec[dec.B == 1]),
        ("decode batch", "batch > 1", dec[dec.B > 1]),
        ("KV heads", "MHA (Hkv 32)", f[f.Hkv == MHA]),
        ("KV heads", "GQA (Hkv 8)", f[f.Hkv == GQA]),
        ("dtype", "bf16", f[f.dt == "bf16"]),
        ("dtype", "fp16", f[f.dt == "fp16"]),
    ]
    return pd.DataFrame([{"cut": c, "group": g, "rate": float(d.flip.mean()),
                          "cells": len(d)} for c, g, d in groups if len(d)])


def _label(r):
    return "B%-2d N%-5d D%d %s %s %s %s %s" % (
        r.B, r.N, r.D, "MHA" if r.Hkv == MHA else "GQA", r.dt, r.md,
        "causal" if r.csl else "full", r.lnch)


def _winner_map(cells, regime):
    w = _winners(cells[cells.regime == regime])
    if not len(w):
        st.info("No %s cells in this selection." % regime)
        return
    f = cells.drop_duplicates("cell").set_index("cell")
    order = (f.loc[sorted(set(w.cell))]
             .sort_values(["md", "dt", "D", "Hkv", "lnch", "B", "N"]))
    piv = w.pivot(index="cell", columns="gpu", values="implementation").loc[order.index]
    names = [n for n in COLOURS if n in set(w.implementation)]
    code = {n: i for i, n in enumerate(names)}
    z = piv.replace(code).where(piv.notna()).astype(float).values
    steps = [[b, COLOURS[n]] for i, n in enumerate(names)
             for b in (i / len(names), (i + 1) / len(names))]
    fig = go.Figure(go.Heatmap(
        z=z, x=list(piv.columns), y=[_label(r) for r in order.itertuples()],
        customdata=piv.fillna("not run").values, colorscale=steps, showscale=False,
        zmin=-0.5, zmax=len(names) - 0.5, xgap=1, ygap=1,
        hovertemplate="%{y}<br>%{x}: %{customdata}<extra></extra>"))
    for n in names:                      # a heatmap carries no categorical legend
        fig.add_scatter(x=[None], y=[None], mode="markers", name=n,
                        marker=dict(color=COLOURS[n], size=10, symbol="square"))
    fig.update_layout(
        height=min(2400, 120 + 13 * len(piv)),
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed", tickfont=dict(family="monospace", size=9)),
        legend=dict(orientation="h", y=-0.02, yanchor="top"))
    st.plotly_chart(fig, use_container_width=True)


def render(D):
    cells, rows, summary = D["cells"], _flat(D["rows"]), D["summary"] or {}
    sel = D["selector"] or {}
    if not len(cells):
        return not_collected("Executive results",
                             "cells.parquet is empty. Run a sweep, then "
                             "`python -m akp.analysis`, and start the dashboard "
                             "from the repository root or set AKP_PROCESSED.")
    n_rows = summary.get("n_rows", len(rows))
    n_cells = summary.get("n_cells", len(cells))

    st.title("Does the fastest attention implementation stay fastest?")
    st.write("Three GPUs, %s measured rows, %s (GPU, workload, implementation) "
             "cells. It does not, and how badly it fails depends on the regime."
             % (format(n_rows, ","), format(n_cells, ",")))

    # Provenance first. A reader who meets the cross-GPU decode numbers before
    # the commit split reads a software difference as an architectural one.
    st.subheader("Provenance")
    prov = provenance(rows)
    if len(prov):
        st.dataframe(prov, use_container_width=True, hide_index=True)
    st.warning(
        "Decode is **not** commit-homogeneous. A10 and A100 decode rows come from "
        "468b9a55 on driver 595.71.05; H100 decode is 91959d34 on 580.126.09. "
        "Every cross-GPU decode comparison on this page carries a software "
        "difference alongside the architecture one. Prefill is clean: all three "
        "GPUs ran 91959d34.", icon=":material/warning:")

    st.subheader("Fastest backend per GPU")
    w = _winners(cells)
    for regime, sub, cap in (
            ("prefill", w[w.regime == "prefill"],
             "prefill, fwd and fwd_bwd cells, won on median latency"),
            ("decode", w[(w.regime == "decode") & (w.B == 1)],
             "decode at batch 1, won on median decode-step latency")):
        cols = st.columns(max(1, cells.gpu.nunique()))
        for c, g in zip(cols, sorted(cells.gpu.unique())):
            d = sub[sub.gpu == g]
            if not len(d):
                c.metric("%s - %s" % (g, regime), "-")
                continue
            v = d.implementation.value_counts()
            c.metric("%s - %s" % (g, regime), v.index[0],
                     "%d of %d cells" % (v.iloc[0], len(d)), delta_color="off")
        note(cap)

    st.subheader("Portability and selection")
    inv = _inversion_rates(cells, D["inversions"])
    sp = _max_speedup(cells)
    c = st.columns(6)
    for col, r in zip(c[:2], ("prefill", "decode")):
        if r in inv:
            col.metric("%s inversions" % r, "%.1f%%" % (100 * inv[r][0]),
                       help="implementation pairs whose order flips across a GPU "
                            "pair by more than 10%% in both directions, over %d "
                            "pairs examined" % inv[r][1])
        else:
            col.metric("%s inversions" % r, "n/a")
    mm = summary.get("dispatch_mismatch_rate")
    c[2].metric("Dispatch mismatch", "n/a" if mm is None else "%.2f%%" % (100 * mm),
                help="label and executed kernel disagree; probed rows only")
    if sp is not None:
        c[3].metric("Max prefill speedup over %s" % BASE, "%.2fx" % sp.sp,
                    "%.1f -> %.1f us" % (sp.base_us, sp.median_us), delta_color="off")
    c[4].metric("Selector median regret", "%.3fx" % sel.get("median_regret", float("nan")),
                help="chosen implementation's latency over the oracle's, held-out cells")
    c[5].metric("Selector p95 regret", "%.2fx" % sel.get("p95_regret", float("nan")))

    if sp is not None:
        note("Largest speedup: %s over %s on %s, %s -- %.1f us against %.1f us. "
             "Both arms are the same cell, so mode, launch and timer are fixed."
             % (sp.implementation, BASE, sp.name[0], sp.name[1],
                sp.median_us, sp.base_us))
    note("Selector top-1 accuracy is %.1f%% over %d held-out cells, but accuracy "
         "counts label agreement and most cells are near-ties: the median "
         "mistake costs %.1f%%. Regret is the number that matters here."
         % (100 * sel.get("accuracy", float("nan")), sel.get("n_heldout", 0),
            100 * (sel.get("median_regret", 1) - 1)))
    note("Dispatch mismatch covers probed rows only. %s of %s OK rows carry no "
         "CUDA-kernel trace, and the miss is biased toward the fastest "
         "implementations (P4-fa2 ~73%%, P4h-fa3 ~68%%, P3-triton ~60%%), so the "
         "true rate is unknown rather than small."
         % (format(summary.get("dispatch_probe_failures", 0), ","),
            format((summary.get("status") or {}).get("OK", 0), ",")))

    st.subheader("Winner map")
    st.caption("Fastest implementation per workload cell, by GPU. One row per "
               "cell, grouped by regime; blank means that GPU never ran it.")
    sub = filter_bar(cells, key="ovw")
    if not len(sub):
        st.info("No cells for this selection.")
    else:
        for regime in [r for r in ("prefill", "decode") if (sub.regime == r).any()]:
            st.markdown("**%s**" % regime)
            _winner_map(sub, regime)
    note("An implementation missing from a column is often a strip, not a "
         "defeat: " + "; ".join("%s is %s" % (k, v) for k, v in STRIPS.items()) + ".")

    st.subheader("How often the winner changes")
    cuts = _flip_cuts(cells)
    if not len(cuts):
        st.info("No cell is shared by every GPU.")
    else:
        fig = px.bar(cuts, x="group", y="rate", color="cut", text="cells",
                     labels={"rate": "cells whose winner differs across GPUs",
                             "group": ""})
        fig.update_traces(texttemplate="n=%{text}", textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%", legend_title_text="",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        wf = summary.get("winner_flips") or {}
        note("Restricted to the %d cells all %d GPUs ran; pooled rate %.1f%%. The "
             "decode bars inherit the commit split above, so part of that change "
             "rate is software and not architecture."
             % (wf.get("n_cells", 0), wf.get("n_gpus", 0),
                100 * wf.get("flip_rate", float("nan"))))

    st.subheader("What this dataset is, and is not")
    a, b = st.columns(2)
    with a:
        st.markdown("**Is**")
        st.markdown(
            "- %s rows on three devices: A10 (sm86), A100-SXM4-80GB (sm80), "
            "H100 80GB HBM3 (sm90).\n"
            "- %s cells: prefill and decode, fwd and fwd_bwd, bf16 and fp16, "
            "head dim 64 and 128, Hkv 32 and 8, batch 1-64, length 256-16384.\n"
            "- Per-cell medians over five processes. block_bench keeps median, "
            "p5 and p95 and discards the block times, so there is no "
            "within-cell latency distribution to plot."
            % (format(n_rows, ","), format(n_cells, ",")))
        st.dataframe(
            pd.DataFrame({"status": list((summary.get("status") or {}).keys()),
                          "rows": list((summary.get("status") or {}).values())})
            .assign(meaning=lambda d: d.status.map(STATUS_MEANING)),
            use_container_width=True, hide_index=True)
    with b:
        st.markdown("**Is not**")
        st.markdown(
            "- No sm89 or sm120 device. The environment manifest carries RTX "
            "4090, L40 and L40S entries from a probe, but they contributed zero "
            "measured rows, so nothing here speaks to Ada or Blackwell.\n"
            "- No paging axis: page_size always equals seq_len, one page per "
            "sequence, so D3 against D4 compares kernels and not layouts.\n"
            "- No profiler data anywhere, so nothing on this page is attributed "
            "to occupancy, DRAM traffic or launch overhead.\n"
            "- Not a reliability report: UNSUPPORTED, OOM_PREDICTED and OOM are "
            "three different outcomes and are never pooled.")
        not_collected(
            "Kernel-level attribution",
            "No Nsight Compute or Systems capture exists: no .ncu-rep, no "
            ".nsys-rep, and results/profile/ was never created. Counter "
            "collection needs GPU performance-counter permission, which the "
            "rented hosts withheld.")


if __name__ == "__main__":
    # The regime split is the only number this page derives rather than reads,
    # so it is the only one worth a check: summed back over regimes it has to
    # reproduce analysis.py's pooled inversion rate.
    #     PYTHONPATH=dashboard python -m views.overview      (from the repo root)
    from _data import load

    d = load()
    r = _inversion_rates(d["cells"], d["inversions"])
    num = sum(v[0] * v[1] for v in r.values())
    den = sum(v[1] for v in r.values())
    assert abs(num - d["inversions"].practical.sum()) < 1e-6
    pooled = [v["practical_inversion_rate"]
              for v in d["summary"]["rank_correlation"].values()
              if "practical_inversion_rate" in v]
    # analysis.py divides by a cumulative counter sampled at the last logged
    # inversion, so its denominator can run a pair or two short of the true one.
    assert abs(num / den - np.mean(pooled)) < 0.02, (num / den, pooled)
    print("ok: pooled %.4f, per regime %s"
          % (num / den, {k: round(v[0], 4) for k, v in r.items()}))
