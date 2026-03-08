"""Page 7 -- methodology and reproducibility.

Everything the other pages assume: what the hardware was, what software wrote
the rows, which implementation x regime x dtype x GPU cells were even attempted,
how a number was timed, and how much of it is noise. Nothing here is a result;
it is the ledger against which the results are read.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from _data import (COLOURS, STATUS_COLOURS, STATUS_MEANING, cell_key,
                   filter_bar, gpu_meta, not_collected, note, provenance,
                   short_gpu)

# Compute capability is what the runtime reports; the marketing name is not in
# any artefact, so it is written down here once rather than guessed per panel.
ARCH = {"sm80": "Ampere GA100", "sm86": "Ampere GA102",
        "sm89": "Ada Lovelace", "sm90": "Hopper GH100"}

# Where each hardware figure came from. The distinction matters because none of
# them is a vendor datasheet number: theoretical_bw_gbs is null on every device.
SOURCE = {
    "cc": "CUDA device query", "arch": "written down here from cc",
    "sm_count": "CUDA device query", "memory_gb": "CUDA device query",
    "l2_mb": "CUDA device query", "measured_bw_gbs": "measured here",
    "theoretical_bw_gbs": "vendor specification (not recorded)",
    "event_overhead_us": "measured here", "driver": "NVML",
    "cuda": "torch build", "torch": "package version",
}

# block_bench's constants. They are arguments in akp/bench.py, not columns in
# the parquet, so a reader cannot recover them from the data.
WARMUP, TARGET_US, COPY_MB = 25, 200.0, 512

NAIVE_LIKE = ("P0-naive", "P1-inductor", "P1-inductor-nofuse",
              "P1-inductor-where")

CAT_COLOUR = {"never ran": "#f2f2f2", "measured": "#dcefe2",
              "partial": "#fbecd2", "declined by design": "#e6e6e6"}


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """rows.parquet already has a 'regime' column and cell_key() adds a second
    one, so df.regime is 2-D and every groupby on it raises. Keep the first."""
    return df.loc[:, ~df.columns.duplicated()]


def _cat(ok, unsup, oomp, oom):
    if ok + unsup + oomp + oom == 0:
        return "never ran"
    if ok == 0:
        return "declined by design" if unsup else "partial"
    return "measured" if unsup + oomp + oom == 0 else "partial"


def _label(ok, unsup, oomp, oom):
    if ok + unsup + oomp + oom == 0:
        return "never ran"
    bits = []
    if ok:
        bits.append("%d measured" % ok)
    if unsup:
        bits.append("%d declined" % unsup)
    if oomp:
        bits.append("%d OOM-pred" % oomp)
    if oom:
        bits.append("%d OOM" % oom)
    return " / ".join(bits)


def render(D):
    st.title("Methodology and reproducibility")
    st.write("What the numbers on the other pages rest on: the devices, the "
             "software that produced each row, the cells that were attempted, "
             "and the size of the measurement noise.")

    rows = _flat(D["rows"])
    # gpu_meta() sorts on a column it only creates per device, so it raises on
    # an environment-less summary rather than returning an empty frame.
    env = (D.get("summary") or {}).get("environment") or {}
    meta = gpu_meta(D["summary"]) if env else pd.DataFrame()

    _hardware(meta, rows)
    _software(meta, rows)
    _support(rows)
    _sweep(rows)
    _protocol(meta, rows)
    _reliability(D["stability"], rows)
    _cache(D["cache_sensitivity"])
    _feasibility(D["oom_calibration"], rows)
    _power(rows)


# --------------------------------------------------------------------------- #
# (a) hardware
# --------------------------------------------------------------------------- #

def _hardware(meta, rows):
    st.header("Hardware")
    if not len(meta):
        not_collected("Hardware", "summary.json carries no environment block.")
        return

    m = meta.copy()
    m["arch"] = m.cc.map(ARCH).fillna("unknown")
    counted = rows.gpu.value_counts() if len(rows) else pd.Series(dtype=int)
    m["rows"] = m.gpu.map(counted).fillna(0).astype(int)
    m["l2_mb"] = m.l2_mb.round(1)

    cols = ["gpu", "arch", "cc", "sm_count", "memory_gb", "l2_mb",
            "measured_bw_gbs", "theoretical_bw_gbs", "event_overhead_us",
            "driver", "cuda", "rows"]
    st.dataframe(m[cols].rename(columns={
        "gpu": "GPU", "arch": "architecture", "cc": "compute capability",
        "sm_count": "SMs", "memory_gb": "memory (GB)", "l2_mb": "L2 (MB)",
        "measured_bw_gbs": "measured BW (GB/s)",
        "theoretical_bw_gbs": "spec BW (GB/s)",
        "event_overhead_us": "event overhead (us)", "driver": "driver",
        "cuda": "CUDA", "rows": "rows in this dataset"}),
        use_container_width=True, hide_index=True)

    idle = m[m.rows == 0].gpu.tolist()
    if idle:
        note("%s appear in the environment manifest because preflight probed "
             "them, but contributed no rows to this dataset. Every result page "
             "covers %s only."
             % (", ".join(idle), ", ".join(m[m.rows > 0].gpu)))

    st.subheader("Where each figure comes from")
    st.dataframe(pd.DataFrame({"field": list(SOURCE), "source": list(SOURCE.values())}),
                 use_container_width=True, hide_index=True)

    if m.theoretical_bw_gbs.isna().all():
        st.warning(
            "**theoretical_bw_gbs is null on every device.** The bandwidth "
            "column is a *measured* %d MB device-to-device copy (median of 20 "
            "reps), which is a floor on achievable bandwidth, not a peak: a "
            "copy both reads and writes, so it under-measures a read-dominated "
            "stream. It is the denominator of `bw_util`, which is why ~546 rows "
            "exceed 100%%. Never label it a theoretical or vendor peak."
            % COPY_MB, icon=":material/warning:")

    note("Event overhead is one cudaEventRecord pair around an empty region, "
         "median of 50. It is the reason block_bench times k calls per pair "
         "rather than one: at 5-10 us it would otherwise dominate a decode step.")


# --------------------------------------------------------------------------- #
# (b) software manifest and provenance
# --------------------------------------------------------------------------- #

def _software(meta, rows):
    st.header("Software")
    if not len(meta):
        not_collected("Software manifest", "no environment block in summary.json.")
        return

    sw = ["gpu", "torch", "triton", "flash_attn", "flash_attn_3", "flashinfer",
          "cuda", "driver", "git_sha"]
    st.dataframe(meta[sw].rename(columns={"gpu": "GPU", "git_sha": "manifest commit"}),
                 use_container_width=True, hide_index=True)
    note("flash_attn_3 records only presence, not a version: the interface "
         "package exposes none. It is present on the sm90 device alone.")

    prov = provenance(rows)
    if not len(prov):
        not_collected("Provenance", "rows.parquet carries no git_sha column.")
        return

    st.subheader("Which commit produced which rows")
    wide = prov.pivot_table(index=["regime", "commit"], columns="gpu",
                            values="rows", aggfunc="sum", fill_value=0).astype(int)
    st.dataframe(wide.reset_index(), use_container_width=True, hide_index=True)

    per_regime = prov.groupby("regime").commit.nunique()
    split = per_regime[per_regime > 1].index.tolist()
    if split:
        st.warning(
            "**%s rows are not commit-homogeneous.** The manifest above records "
            "one commit per GPU (the last run on it); the table is the truth per "
            "row. Any cross-GPU %s comparison confounds architecture with a "
            "software difference -- different commit, and on these devices a "
            "different driver branch too. Say so wherever such a delta is quoted."
            % (" and ".join(split), split[0]), icon=":material/warning:")
    homogeneous = per_regime[per_regime == 1].index.tolist()
    if homogeneous:
        note("%s rows all come from a single commit, so cross-GPU %s deltas are "
             "architecture-only." % (", ".join(homogeneous), ", ".join(homogeneous)))


# --------------------------------------------------------------------------- #
# (c) support matrix
# --------------------------------------------------------------------------- #

def _support(rows):
    st.header("Implementation support matrix")
    if not len(rows):
        not_collected("Support matrix", "rows.parquet is empty.")
        return

    st.caption("Every implementation x GPU x dtype cell carries an explicit "
               "count. Regime is a property of the implementation (P* prefill, "
               "D* decode), not an independent axis, so it labels the row "
               "instead of crossing it -- crossing would manufacture 50% "
               "structurally empty cells.")

    ct = (rows.groupby(["implementation", "regime", "gpu", "dtype"]).status
          .value_counts().unstack(fill_value=0))
    for s in ("OK", "UNSUPPORTED", "OOM_PREDICTED", "OOM"):
        if s not in ct:
            ct[s] = 0
    ct = ct.reset_index()

    impls = sorted(rows.implementation.unique())
    regime_of = dict(rows.groupby("implementation").regime.first())
    gpus = sorted(rows.gpu.unique())
    dts = sorted(rows.dtype.unique())
    have = {(r.implementation, r.gpu, r.dtype): r for r in ct.itertuples()}

    index = ["%s  (%s)" % (i, regime_of[i]) for i in impls]
    columns = ["%s %s" % (g, d) for g in gpus for d in dts]
    text = pd.DataFrame("", index=index, columns=columns)
    code = pd.DataFrame("never ran", index=index, columns=columns)
    for i, lab in zip(impls, index):
        for g in gpus:
            for d in dts:
                r = have.get((i, g, d))
                n = (0, 0, 0, 0) if r is None else (r.OK, r.UNSUPPORTED,
                                                    r.OOM_PREDICTED, r.OOM)
                text.loc[lab, "%s %s" % (g, d)] = _label(*n)
                code.loc[lab, "%s %s" % (g, d)] = _cat(*n)

    # A Styler would be the obvious way to colour this, but pandas' .style
    # needs jinja2, which this environment does not have. A heatmap carrying
    # its own cell text costs the same and drops the dependency.
    order = ["never ran", "declined by design", "partial", "measured"]
    n = len(order)
    scale = []
    for i, c in enumerate(order):
        scale += [[i / n, CAT_COLOUR[c]], [(i + 1) / n, CAT_COLOUR[c]]]
    fig = go.Figure(go.Heatmap(
        z=code.map(order.index).values, x=list(columns), y=list(index),
        text=text.values, texttemplate="%{text}", hoverinfo="text",
        colorscale=scale, zmin=-0.5, zmax=n - 0.5, showscale=False,
        xgap=2, ygap=2))
    fig.update_traces(textfont=dict(size=11, color="#111"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=90 + 34 * len(index),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(" · ".join("%s = %s" % (k, v) for k, v in
                          [("measured", "every attempted row ran"),
                           ("partial", "some rows declined or refused"),
                           ("declined by design", "zero measured rows"),
                           ("never ran", "cell never entered the sweep")]))

    st.subheader("Deliberate strips")
    st.caption("A grey cell is a configuration the implementation never claimed "
               "to support. It is a boundary of the design, not a defeat, and "
               "must not be shown in a winner map or a regret table without "
               "this annotation.")
    strips = []
    for i in impls:
        sub = ct[ct.implementation == i]
        if not sub.UNSUPPORTED.sum():
            continue
        tot = sub[["OK", "UNSUPPORTED", "OOM_PREDICTED", "OOM"]].values.sum()
        dead_dt = [d for d in dts if sub[sub.dtype == d].OK.sum() == 0]
        dead_gpu = [g for g in gpus if sub[sub.gpu == g].OK.sum() == 0]
        strips.append({
            "implementation": i,
            "declined (%)": round(100 * sub.UNSUPPORTED.sum() / tot, 1),
            "measured rows": int(sub.OK.sum()),
            "dtypes with zero measured rows": ", ".join(dead_dt) or "-",
            "GPUs with zero measured rows": ", ".join(dead_gpu) or "-"})
    if strips:
        st.dataframe(pd.DataFrame(strips).sort_values("declined (%)", ascending=False),
                     use_container_width=True, hide_index=True)
    note("P4h-fa3 is sm90-only: it declined every A10 and A100 configuration, so "
         "no cross-architecture statement about FA3 is possible from this "
         "dataset. D1-inductor and the P1-inductor-nofuse / -where arms are bf16 "
         "strips: they were never asked to compile an fp16 kernel.")
    note("Status is not a severity scale. " + " · ".join(
        "%s = %s" % (k, v) for k, v in STATUS_MEANING.items() if k in
        set(rows.status.unique())))


# --------------------------------------------------------------------------- #
# (d) sweep definition
# --------------------------------------------------------------------------- #

def _sweep(rows):
    st.header("Workload sweep")
    if not len(rows):
        not_collected("Sweep definition", "rows.parquet is empty.")
        return

    axes = [("B", "batch"), ("N", "seq / KV length"), ("Hq", "query heads"),
            ("Hkv", "KV heads"), ("D", "head dim"), ("dt", "dtype"),
            ("md", "mode"), ("csl", "causal"), ("lnch", "launch"),
            ("cch", "cache")]
    out = []
    for regime, g in rows.groupby("regime"):
        for col, label in axes:
            vals = sorted(g[col].dropna().unique().tolist(), key=str)
            out.append({"regime": regime, "axis": label,
                        "n": len(vals),
                        "values": ", ".join(str(v) for v in vals)})
    st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)

    grid = (rows.groupby(["regime", "gpu"])
            .agg(cells=("cell", "nunique"), rows=("cell", "size"),
                 implementations=("implementation", "nunique")).reset_index())
    tot = rows.groupby("regime").cell.nunique()
    st.subheader("Grid size")
    st.dataframe(grid, use_container_width=True, hide_index=True)
    note("Distinct cells across all GPUs: " +
         ", ".join("%s %d" % (k, v) for k, v in tot.items()) +
         ". A cell is the composite key regime|B|Hq|Hkv|D|N|dtype|mode|launch|"
         "cache|causal; a row is one implementation x cell x process repeat.")

    if "page_size" in rows and rows.page_size.notna().any():
        same = (rows.page_size == rows.seq_len)
        if same[rows.page_size.notna()].all():
            note("page_size equals seq_len on every row that has one: one page "
                 "per sequence, by design. There is no paging axis in this "
                 "sweep, so no chart here compares page layouts.")


# --------------------------------------------------------------------------- #
# (e) timing protocol
# --------------------------------------------------------------------------- #

def _protocol(meta, rows):
    st.header("Timing protocol")
    if not len(rows):
        not_collected("Timing protocol", "rows.parquet is empty.")
        return
    st.markdown(
        "- **Warmup** %d untimed calls, then 10 more to size the inner loop.\n"
        "- **Inner loop** k calls per cudaEventRecord pair, k chosen so a block "
        "exceeds %.0f us. That keeps the event pair under ~1%% of the "
        "measurement even for a 5-15 us decode kernel.\n"
        "- **Reps** %s timed blocks per row; the row keeps median, p5 and p95 "
        "and discards the individual block times. There is therefore no "
        "per-call latency distribution anywhere in this dataset.\n"
        "- **Process repeats** the whole grid is re-run in a fresh process; "
        "`repeat` is part of the config hash, so repeats never overwrite each "
        "other.\n"
        "- **Interleaving** configurations are shuffled once with a fixed seed, "
        "and the implementation order is reshuffled per configuration with a "
        "per-repeat seed, so clock and thermal drift lands on every "
        "implementation rather than on whichever ran last.\n"
        "- **L2 policy** warm by default. Cold rows allocate a buffer twice L2 "
        "and zero it once per *block* of k calls, so only the first of the k is "
        "truly cold."
        % (WARMUP, TARGET_US,
           ", ".join(str(int(v)) for v in sorted(rows.reps.dropna().unique()))))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rows per timer")
        t = (rows.timer.fillna("(never timed)").value_counts()
             .rename("rows").rename_axis("timer").reset_index())
        st.dataframe(t, use_container_width=True, hide_index=True)
        st.caption("`(never timed)` is the non-OK rows: declined, refused or "
                   "failed, so no timer ran. The two real timers are different "
                   "instruments -- block_bench times a k-call inner loop, "
                   "do_bench_cudagraph times a captured graph -- and a latency "
                   "axis must never be shared across them without faceting.")
    with c2:
        st.subheader("Rows per process repeat")
        st.dataframe(rows.pivot_table(index="repeat", columns="gpu",
                                      values="cell", aggfunc="size",
                                      fill_value=0).reset_index(),
                     use_container_width=True, hide_index=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Inner loop k")
        k = (rows.inner_k.dropna().astype(int).value_counts().sort_index()
             .rename("rows").rename_axis("k").reset_index())
        st.dataframe(k, use_container_width=True, hide_index=True)
        st.caption("k=1 means one call already exceeded the block target.")
    with c4:
        st.subheader("L2 flush")
        # Cast before grouping: a bool column with a string sentinel mixed in
        # is object-typed and Arrow refuses to serialise it.
        flushed = rows.l2_flushed.map({True: "cold (flushed)", False: "warm"}) \
            .fillna("(never timed)").astype(str)
        st.dataframe(rows.groupby(flushed).size()
                     .rename("rows").rename_axis("L2 state").reset_index(),
                     use_container_width=True, hide_index=True)
        if len(meta):
            st.caption("Measured event overhead: " + ", ".join(
                "%s %.2f us" % (r.gpu, r.event_overhead_us)
                for r in meta[meta.event_overhead_us.notna()].itertuples()))

    note("The cudagraph arm is not a launch-overhead measurement. It is a "
         "different timer on a captured graph, and its delta against eager is "
         "far larger than the few microseconds of measured event overhead. "
         "Present it as 'eager vs CUDA-graph capture, different timers'.")


# --------------------------------------------------------------------------- #
# (f) timing reliability
# --------------------------------------------------------------------------- #

def _reliability(stab, rows):
    st.header("Timing reliability")
    if not len(stab):
        not_collected("Timing reliability",
                      "stability.parquet is empty -- the sweep ran a single "
                      "process repeat, so there is no between-process spread.")
    else:
        s = stab.assign(gpu=stab.gpu_name.map(short_gpu))
        single = int((s.n_repeats < 2).sum())
        s = s[s.n_repeats >= 2]
        note("%d of %d (implementation, cell) groups ran in only one process; "
             "their CI width is identically zero and they are excluded from "
             "both panels below." % (single, len(stab)))

        st.subheader("Relative CI width across process repeats")
        fig = px.histogram(s, x="rel_ci_width", color="implementation",
                           facet_col="regime", nbins=60, log_y=True,
                           color_discrete_map=COLOURS)
        fig.update_layout(barmode="overlay", legend_title_text="")
        fig.update_traces(opacity=0.65)
        st.plotly_chart(fig, use_container_width=True)
        med = s.rel_ci_width.median()
        p95 = s.rel_ci_width.quantile(0.95)
        st.caption("Median %.1f%%, p95 %.1f%%, worst %.0f%%. A speedup smaller "
                   "than the CI width of either arm is not a result."
                   % (100 * med, 100 * p95, 100 * s.rel_ci_width.max()))

        st.subheader("Between-process CV")
        fig = px.box(s.sort_values("implementation"), x="implementation", y="cv",
                     color="implementation", facet_col="gpu",
                     color_discrete_map=COLOURS, points=False)
        fig.update_yaxes(title="CV of the median across repeats")
        fig.update_xaxes(title="", showticklabels=False)
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Coefficient of variation of the per-process median, over "
                   "the %s repeats that ran. It is process-to-process spread, "
                   "not within-run jitter -- block_bench keeps no per-call "
                   "times to build the latter from."
                   % "/".join(str(int(v)) for v in sorted(s.n_repeats.unique())))

    st.subheader("Trial-order drift")
    if "timestamp" not in rows or rows.timestamp.isna().all():
        not_collected("Trial-order drift", "rows.parquet has no timestamp.")
        return

    d = rows[rows.status == "OK"].copy()
    d = filter_bar(d, key="repro_drift", fields=("gpu", "regime", "dt", "md"),
                   default_gpu="H100")
    if not len(d):
        st.info("No rows for this selection.")
        return
    d["elapsed_min"] = ((d.timestamp - d.groupby(["gpu", "repeat"])
                         .timestamp.transform("min")) / 60.0)
    d["ratio"] = d.median_us / d.groupby(["gpu", "cell", "implementation"]) \
        .median_us.transform("median")

    y = st.radio("y axis", ["median_us", "ratio to the cell's own median"],
                 horizontal=True, key="repro_drift_y")
    ycol, logy = ("median_us", True) if y == "median_us" else ("ratio", False)
    fig = px.scatter(d, x="elapsed_min", y=ycol, color="implementation",
                     facet_col="gpu", facet_row="repeat", opacity=0.45,
                     color_discrete_map=COLOURS,
                     labels={"elapsed_min": "minutes into the repeat"})
    fig.update_traces(marker_size=3)
    if logy:
        fig.update_yaxes(type="log")
    else:
        fig.add_hline(y=1.0, line_dash="dot", line_width=1)
    fig.update_layout(legend_title_text="", height=200 + 180 * d.repeat.nunique())
    st.plotly_chart(fig, use_container_width=True)
    st.caption("x is wall-clock position inside one process repeat. On the raw "
               "axis the vertical banding is the workload grid, not drift -- "
               "the cells themselves span three orders of magnitude. The ratio "
               "axis is the one that shows drift: because implementation order "
               "is reshuffled per configuration, a clock or thermal excursion "
               "moves every colour together rather than penalising whichever "
               "implementation happened to run late.")


# --------------------------------------------------------------------------- #
# (g) warm vs cold
# --------------------------------------------------------------------------- #

def _cache(cs):
    st.header("Warm vs cold L2")
    if not len(cs):
        not_collected("Warm vs cold L2",
                      "cache_sensitivity.parquet is empty -- no cold-cache rows "
                      "were paired with a warm counterpart.")
        return

    k1 = cs[(cs.inner_k == 1) & cs.sensitivity.notna()].copy()
    st.caption("%d of %d pairs survive the inner_k == 1 filter. The flush buffer "
               "is zeroed once per *block* of k calls, so at k > 1 only the "
               "first call of the block is genuinely cold and the rest are "
               "measuring a warmed cache under a cold label. Only k = 1 is "
               "interpretable." % (len(k1), len(cs)))
    if not len(k1):
        st.info("No inner_k == 1 pairs.")
        return

    k1["gpu"] = k1.gpu_name.map(short_gpu)
    fig = px.scatter(k1, x="us_warm", y="us_cold", color="implementation",
                     symbol="gpu", color_discrete_map=COLOURS,
                     hover_data=["cell_cold", "reps_cold", "reps_warm"],
                     labels={"us_warm": "warm (us)", "us_cold": "cold (us)"})
    lo = float(min(k1.us_warm.min(), k1.us_cold.min()))
    hi = float(max(k1.us_warm.max(), k1.us_cold.max()))
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(dash="dot", width=1))
    fig.update_traces(marker_size=10)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(k1.groupby(["gpu", "implementation"])
                 .agg(pairs=("sensitivity", "size"),
                      median_sensitivity=("sensitivity", "median"),
                      median_cold_us=("us_cold", "median"),
                      median_warm_us=("us_warm", "median")).round(3).reset_index(),
                 use_container_width=True, hide_index=True)
    note("Cold rows exist on %s only, across %d distinct cells. This is a "
         "sanity check on the flush mechanism, not a cache-sensitivity result: "
         "the sample is too small and too narrow to rank implementations by it. "
         "Points below the diagonal are cold measuring *faster* than warm, "
         "which is the block-zeroing artefact above, not a cache effect."
         % (", ".join(sorted(k1.gpu.unique())), k1.cell_cold.nunique()))


# --------------------------------------------------------------------------- #
# (h) OOM and feasibility
# --------------------------------------------------------------------------- #

def _feasibility(oom, rows):
    st.header("Memory prediction and feasibility")
    if not len(oom):
        not_collected("OOM calibration", "oom_calibration.parquet is empty.")
        return

    o = oom.copy()
    o["gpu"] = o.gpu_name.map(short_gpu)
    o = pd.concat([o, cell_key(o.cell)], axis=1)

    st.warning(
        "**The frontier below belongs to %s only.** The allocation is predicted "
        "in advance solely for the score-matrix implementations, because only "
        "they materialise an N x N tensor whose size is known ahead of the call. "
        "Every fused implementation was simply attempted, so its memory frontier "
        "does not appear here at all -- absence of an OOM_PREDICTED point is not "
        "evidence that a configuration fits."
        % ", ".join(sorted(set(NAIVE_LIKE) & set(o.implementation))),
        icon=":material/warning:")

    st.subheader("Predicted vs actual peak allocation")
    ran = o[o.status == "OK"]
    fig = px.scatter(ran, x="predicted_gb", y="actual_gb", color="implementation",
                     symbol="gpu", color_discrete_map=COLOURS,
                     hover_data=["cell"],
                     labels={"predicted_gb": "predicted peak (GB)",
                             "actual_gb": "measured peak (GB)"})
    lo = float(max(1e-3, min(ran.predicted_gb.min(), ran.actual_gb.min())))
    hi = float(max(ran.predicted_gb.max(), ran.actual_gb.max()))
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(dash="dot", width=1))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("%d rows ran and can be compared; %d were refused and have no "
               "measured value, so they are absent from this panel by "
               "construction -- the calibration is only visible where the "
               "prediction said yes. Median predicted/actual ratio %.2f; the "
               "predictor is deliberately conservative, and a ratio above 1 is "
               "headroom, not error."
               % (len(ran), int((o.status == "OOM_PREDICTED").sum()),
                  ran.ratio.median()))

    st.subheader("Feasibility map")
    fig = px.scatter(o, x="N", y="B", color="status", facet_col="implementation",
                     facet_row="gpu", color_discrete_map=STATUS_COLOURS,
                     hover_data=["predicted_gb", "actual_gb"],
                     labels={"N": "sequence length", "B": "batch"})
    fig.update_traces(marker_size=9, marker_line_width=0)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.update_layout(legend_title_text="", height=680)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(" · ".join("%s = %s" % (k, STATUS_MEANING[k])
                          for k in o.status.unique() if k in STATUS_MEANING))

    actual = rows[rows.status == "OOM"] if len(rows) else rows
    if len(actual):
        st.subheader("Allocations that actually failed")
        st.dataframe(actual.groupby(["gpu", "regime", "implementation", "dtype"])
                     .size().rename("rows").reset_index(),
                     use_container_width=True, hide_index=True)
        note("These %d rows are not in oom_calibration -- that table only holds "
             "the predicted arm. They are real failures on implementations the "
             "predictor never guards, which is exactly the blind spot the "
             "warning above describes." % len(actual))


# --------------------------------------------------------------------------- #
# (i) power capping
# --------------------------------------------------------------------------- #

def _power(rows):
    st.header("Power capping")
    if "power_capped" not in rows or not len(rows):
        not_collected("Power capping", "no throttle telemetry in rows.parquet.")
        return

    st.caption("The usability filter deliberately KEEPS software power-cap rows "
               "(throttle bit 0x4) and rejects only the erratic 0xF8 group. A "
               "power cap is the steady state of a real deployment; discarding "
               "it would bias every number toward an unloaded machine. But the "
               "rate is very uneven across GPUs and grows with size, so it is a "
               "confound in any cross-GPU comparison and has to be visible.")

    per_gpu = (rows.groupby("gpu").power_capped
               .agg(rows="size", capped="sum").reset_index())
    per_gpu["rate"] = (per_gpu.capped / per_gpu.rows).round(3)
    st.dataframe(per_gpu, use_container_width=True, hide_index=True)

    g = (rows.groupby(["gpu", "regime", "N"]).power_capped
         .agg(rate="mean", rows="size").reset_index())
    fig = px.bar(g, x="N", y="rate", color="gpu", barmode="group",
                 facet_col="regime", hover_data=["rows"],
                 labels={"N": "sequence / KV length",
                         "rate": "fraction of rows power-capped"})
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    note("The cap rate rises monotonically with sequence length on every "
         "device, so it is not noise: longer kernels hold the SMs busy for "
         "longer and reach the cap. A cross-GPU ratio taken at large N compares "
         "a mostly-capped device with a mostly-uncapped one.")

    not_collected(
        "Clock, occupancy and DRAM counters",
        "No Nsight capture exists in this repository -- no .ncu-rep, no "
        ".nsys-rep, and results/profile/ was never created. Occupancy, warp "
        "stall reasons, DRAM traffic and launch-overhead fractions cannot be "
        "reported, and nothing on any page infers them. The only clock evidence "
        "collected is the NVML throttle bitmask summarised above.")
