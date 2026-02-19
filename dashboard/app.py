"""Results dashboard.

Reads only results/processed/, which akp.analysis writes. Nothing is recomputed
here, so a number in the dashboard and the same number in the report cannot
disagree.

    python -m akp.analysis
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

PROCESSED = os.environ.get("AKP_PROCESSED", "results/processed")

# Categorical slots, validated for both surfaces (see dataviz palette).
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
         "#4a3aa7", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
        "#9085e9", "#e66767"]

# Colour follows the implementation, not its rank, so filtering the chart never
# repaints the series that remain.
SLOT_ORDER = {
    "prefill": ["P4-fa2", "P3-triton", "P2c-sdpa-flash", "P2d-sdpa-cudnn",
                "P2b-sdpa-mem-eff", "P0-naive", "P1-inductor", "P4h-fa3",
                "P1-inductor-nofuse", "P1-inductor-where"],
    "decode": ["D3-fa-kvcache", "D4-flashinfer", "D2-sdpa", "D0-naive-kv",
               "D6-fa-prefill-at-1", "D1-inductor"],
}
DEFAULTS = {"prefill": 6, "decode": 6}
MAX_SERIES = 8          # past eight the palette cannot stay CVD-separable

st.set_page_config(page_title="Attention Kernel Portability", layout="wide")
DARK_MODE = st.get_option("theme.base") == "dark"
PALETTE = DARK if DARK_MODE else LIGHT
TEMPLATE = "plotly_dark" if DARK_MODE else "plotly_white"


@st.cache_data
def load():
    def parquet(name):
        p = os.path.join(PROCESSED, name)
        return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()

    def js(name):
        p = os.path.join(PROCESSED, name)
        return json.load(open(p, encoding="utf8")) if os.path.exists(p) else {}

    return (parquet("rows.parquet"), parquet("cells.parquet"),
            parquet("dispatch_audit.parquet"), parquet("inversions.parquet"),
            js("summary.json"), js("selector.json"))


rows, cells, audit, inv, summary, sel = load()

if rows.empty:
    st.error(f"No results in {PROCESSED}. Run a sweep, then `python -m akp.analysis`.")
    st.stop()


def colours(regime):
    return {n: PALETTE[i % len(PALETTE)]
            for i, n in enumerate(SLOT_ORDER[regime])}


def series_picker(df, regime, key):
    """One filter row above the chart; capped so the palette stays separable."""
    present = [n for n in SLOT_ORDER[regime] if n in set(df.implementation)]
    chosen = st.multiselect("Implementations", present,
                            default=present[:DEFAULTS[regime]], key=key)
    if len(chosen) > MAX_SERIES:
        st.caption(f"Showing the first {MAX_SERIES}: beyond that the colours "
                   f"stop being reliably distinguishable.")
        chosen = chosen[:MAX_SERIES]
    return df[df.implementation.isin(chosen)]


def line(df, x, y, regime, title, log_y=True):
    if df.empty:
        st.info("No rows for this selection.")
        return
    fig = px.line(df.sort_values(x), x=x, y=y, color="implementation",
                  markers=True, template=TEMPLATE, title=title,
                  color_discrete_map=colours(regime),
                  facet_col="gpu_name" if df.gpu_name.nunique() > 1 else None)
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    # One series needs no legend box: the title already names it.
    fig.update_layout(hovermode="x unified", legend_title_text="",
                      showlegend=df.implementation.nunique() > 1)
    st.plotly_chart(fig, use_container_width=True)


def filters(df, regime, key):
    """Shape filters, in one row, above the chart."""
    c = st.columns(5)
    out = df[df.regime == regime]
    for col, name, label in zip(c, ["gpu_name", "dtype", "head_dim", "batch", "hkv"],
                                ["GPU", "dtype", "head dim", "batch", "KV heads"]):
        vals = sorted(out[name].unique())
        pick = col.selectbox(label, vals, key=f"{key}_{name}")
        out = out[out[name] == pick]
    return out


PAGES = ["Hardware & environment", "Prefill", "Decode", "Dispatch & fallbacks",
         "Ranking inversions", "Nsight & roofline", "Backend selector"]
page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption(f"{summary.get('n_rows', 0)} rows · "
                   f"{len(summary.get('gpus', []))} GPU(s)")

# --------------------------------------------------------------------------- #

if page == "Hardware & environment":
    st.title("Attention kernel portability")
    st.write("Does the fastest attention implementation stay the fastest when "
             "the GPU or the workload regime changes?")

    a, b, c = st.columns(3)
    a.metric("Measured rows", summary.get("n_rows", 0))
    b.metric("Configurations", summary.get("n_cells", 0))
    mm = summary.get("dispatch_mismatch_rate")
    c.metric("Dispatch mismatch", "n/a" if mm is None else f"{mm:.1%}")

    st.subheader("Devices")
    env = summary.get("environment", {})
    if env:
        st.dataframe(pd.DataFrame([
            {"GPU": g, "CC": f"{m['cc_major']}.{m['cc_minor']}",
             "SMs": m["sm_count"], "Memory (GB)": m["total_memory_gb"],
             "L2 (MB)": round(m["l2_bytes"] / 2**20, 1),
             "Peak BW (GB/s)": m.get("theoretical_bw_gbs"),
             "Measured BW (GB/s)": m.get("measured_peak_bw_gbs"),
             "Event overhead (us)": m.get("event_overhead_us"),
             "Driver": m.get("driver"), "torch": m["versions"].get("torch"),
             "triton": m["versions"].get("triton"),
             "flash_attn": m["versions"].get("flash_attn"),
             "flashinfer": m["versions"].get("flashinfer")}
            for g, m in env.items()]), use_container_width=True, hide_index=True)

    st.subheader("Cell status")
    st.dataframe(pd.Series(summary.get("status", {}), name="cells")
                 .rename_axis("status").reset_index(),
                 use_container_width=True, hide_index=True)

elif page in ("Prefill", "Decode"):
    regime = page.lower()
    st.title(f"{page} results")
    df = filters(rows[rows.status == "OK"], regime, regime)
    df = series_picker(df, regime, f"{regime}_impls")

    if regime == "prefill":
        mode = st.radio("Pass", sorted(df["mode"].unique()), horizontal=True) \
            if df["mode"].nunique() > 1 else None
        if mode:
            df = df[df["mode"] == mode]
        line(df, "seq_len", "median_us", regime, "Latency vs sequence length (us)")
        line(df, "seq_len", "tflops", regime, "Throughput (TFLOP/s)", log_y=False)
        line(df, "seq_len", "peak_allocated_mb", regime, "Peak allocated (MB)")
    else:
        launch = st.radio("Launch", sorted(df.launch.unique()), horizontal=True) \
            if df.launch.nunique() > 1 else None
        if launch:
            df = df[df.launch == launch]
        line(df, "seq_len", "us_per_token", regime, "Latency per token (us)")
        line(df, "seq_len", "tokens_per_s", regime, "Tokens/s", log_y=False)
        line(df, "seq_len", "eff_bw_gbs", regime,
             "Effective KV bandwidth (GB/s)", log_y=False)

    st.subheader("Table")
    keep = [c for c in ["implementation", "seq_len", "batch", "head_dim", "dtype",
                        "median_us", "p5_us", "p95_us", "tflops", "eff_bw_gbs",
                        "peak_allocated_mb", "max_abs_err", "correctness_pass"]
            if c in df.columns]
    st.dataframe(df[keep].sort_values(["implementation", "seq_len"]),
                 use_container_width=True, hide_index=True)

elif page == "Dispatch & fallbacks":
    st.title("Did the requested backend actually run?")
    st.write("Every cell records the CUDA kernels the call launched. A mismatch "
             "means the label and the executed kernel disagree.")
    if audit.empty:
        st.info("No dispatch records yet.")
    else:
        per = (audit.groupby("implementation")
               .agg(cells=("matched", "size"),
                    matched=("matched", "mean"),
                    kernels=("n_kernels", "median")).reset_index())
        per["matched"] = (per["matched"] * 100).round(1)
        st.dataframe(per.rename(columns={"matched": "matched (%)",
                                         "kernels": "kernels launched (median)"}),
                     use_container_width=True, hide_index=True)

        fused = audit[audit.fused_into_sdpa]
        st.metric("Cells where TorchInductor substituted SDPA", len(fused))

    st.subheader("Status by implementation")
    st.dataframe(pd.crosstab(rows.implementation, rows.status),
                 use_container_width=True)

elif page == "Ranking inversions":
    st.title("Does the ranking transfer across GPUs?")
    gpus = summary.get("gpus", [])
    if len(gpus) < 2:
        st.info(f"Needs at least two GPUs; results so far cover {gpus}.")
    else:
        st.subheader("Rank correlation")
        st.dataframe(pd.DataFrame(summary["rank_correlation"]).T
                     .rename_axis("pair").reset_index(),
                     use_container_width=True, hide_index=True)
        if not inv.empty:
            a, b = st.columns(2)
            a.metric("Practical inversions", int(inv.practical.sum()))
            b.metric("Statistically separated", int(inv.sig.sum()))
            st.dataframe(inv, use_container_width=True, hide_index=True)

    st.subheader("Fastest implementation per configuration")
    st.dataframe(pd.Series(summary.get("winners", {}), name="cells won")
                 .rename_axis("implementation").reset_index(),
                 use_container_width=True, hide_index=True)

elif page == "Nsight & roofline":
    st.title("Attribution")
    ncu = os.path.join(PROCESSED, "ncu.parquet")
    if not os.path.exists(ncu):
        st.info("No Nsight counters collected yet. Profiling covers 8-12 "
                "representative cells and needs GPU performance-counter "
                "permission, which shared clusters often withhold.")
    else:
        st.dataframe(pd.read_parquet(ncu), use_container_width=True)

elif page == "Backend selector":
    st.title("Hardware-aware backend selection")
    st.write("A rule fitted on the principal GPUs and tested on hardware it "
             "never saw, against fixed-backend policies and the oracle.")
    if "error" in sel:
        st.info(sel["error"])
    else:
        a, b, c = st.columns(3)
        a.metric("Held-out accuracy", f"{sel['accuracy']:.1%}")
        b.metric("Median regret vs oracle", f"{sel['median_regret']:.2f}x")
        c.metric("p95 regret", f"{sel['p95_regret']:.2f}x")
        if sel.get("vs_fixed"):
            st.subheader("Against fixed policies")
            st.dataframe(pd.DataFrame(sel["vs_fixed"]).T
                         .rename_axis("always use").reset_index(),
                         use_container_width=True, hide_index=True)
        st.subheader("The rule")
        st.code(sel["rule"], language="text")
