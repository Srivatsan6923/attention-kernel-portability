"""Decode and KV cache.

Decode is memory-bound, so the axes that matter are KV length, batch and
achieved bandwidth -- not FLOP/s, which analysis.py computes for decode as
4*B*Hq*N*D and which lands two to three orders of magnitude under prefill.
Nothing on this page shares an axis with prefill.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from _data import (COLOURS, filter_bar, gpu_meta, not_collected, note, ok,
                   provenance)

PEAK = "measured peak (512MB copy)"
CMAP = dict(COLOURS, **{PEAK: "#c0392b"})


def _med(df, keys, val="median_us"):
    return df.groupby(keys, as_index=False)[val].median()


def _facets(fig):
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(legend_title_text="", hovermode="x unified")
    return fig


def render(D):
    rows = D["rows"]
    # cell_key() re-emits 'regime', which rows.parquet already carries, so the
    # loaded frame has it twice and rows.regime resolves to a DataFrame. Same
    # content either way; keep the first and let attribute access work.
    if len(rows):
        rows = rows.loc[:, ~rows.columns.duplicated()]
    if not len(rows) or "regime" not in rows:
        not_collected("Decode", "rows.parquet is empty or has no regime column.")
        return
    dec_all = rows[rows.regime == "decode"]
    dec = ok(dec_all)
    if not len(dec):
        not_collected("Decode", "no decode rows reached status OK.")
        return

    st.title("Decode and KV cache")
    st.write("One generated token against a KV cache of length N, batch B. "
             "Every number below is a single attention call on a single layer.")

    # Trap 6: the decode rows are not commit-homogeneous, and the split runs
    # exactly along the GPU axis every cross-GPU panel here uses.
    meta = gpu_meta(D["summary"])
    with st.expander("Provenance -- read before comparing GPUs", expanded=False):
        prov = provenance(dec)
        if len(prov):
            st.dataframe(prov[prov.regime == "decode"], hide_index=True,
                         use_container_width=True)
        st.markdown(
            "A10 and A100 decode rows come from a different commit and driver "
            "branch than the H100 ones. Any cross-GPU decode delta on this page "
            "therefore carries a software difference alongside the architectural "
            "one. (The prefill rows are commit-homogeneous; these are not.)")
        cap = (dec.groupby("gpu").power_capped.mean() * 100).round(1)
        st.markdown("SwPowerCap (bit 0x4) share of OK decode rows per GPU: "
                    + ", ".join("%s %.1f%%" % (g, v) for g, v in cap.items())
                    + ". These rows are kept deliberately -- only the erratic "
                      "0xF8 throttle group is rejected -- but the share is "
                      "uneven across GPUs and sizes.")
        st.dataframe(meta[meta.gpu.isin(dec.gpu.unique())][
            ["gpu", "cc", "memory_gb", "measured_bw_gbs", "theoretical_bw_gbs",
             "event_overhead_us", "driver", "git_sha"]],
            hide_index=True, use_container_width=True)

    if st.checkbox("Exclude power-capped rows", key="decode_nocap"):
        dec = dec[~dec.power_capped]

    # Trap 14: two timers live in this regime. Panels (a)-(f) are block_bench
    # eager only; the cudagraph arm gets its own panel and its own axis.
    base = dec[(dec.timer == "block_bench") & (dec.launch == "eager")]
    warm = base[base.cache == "warm"]
    n_cold = int((base.cache == "cold").sum())

    st.divider()
    f = filter_bar(warm, key="decode", fields=("gpu", "dt", "D", "Hkv"))
    note("Filters cover GPU, dtype, head dim and KV heads. Batch and KV length "
         "are axes on the panels below, so they are deliberately not filtered "
         "here. Warm cache only: %d cold-cache decode rows exist in this arm "
         "(H100 only), too few to face against the warm ones here." % n_cold)
    if not len(f):
        st.warning("No rows for this selection.")
        return

    # ---------------------------------------------------------------- (a)
    st.header("a. Time per decode step vs KV length")
    st.caption("y is the latency of one decode step for the whole batch "
               "(median_us, verbatim). It is NOT per-sequence time and it is "
               "not divided by batch.")
    bs = sorted(f.B.unique())
    for b in ([bs[0], bs[-1]] if len(bs) > 1 else bs):
        sub = _med(f[f.B == b], ["gpu", "implementation", "N"])
        fig = px.line(sub.sort_values("N"), x="N", y="median_us",
                      color="implementation", facet_col="gpu", markers=True,
                      log_x=True, log_y=True, color_discrete_map=CMAP,
                      labels={"median_us": "step latency (us)",
                              "N": "KV length"},
                      title="batch %d -- time per decode step (us)" % b)
        st.plotly_chart(_facets(fig), use_container_width=True,
                        key="dec_a_%d" % b)
    note("Both charts are the same quantity at two batch sizes. The gap "
         "between them is the batch factor, not a per-sequence slowdown.")

    # ---------------------------------------------------------------- (b)
    st.header("b. Step latency and aggregate throughput vs batch")
    st.error("**us_per_token and tokens_per_s are not reciprocals.** "
             "`us_per_token` is the time for one decode step of the whole batch "
             "(identical to median_us, never divided by B). `tokens_per_s` is "
             "B / step_time -- an AGGREGATE rate across the batch. They differ "
             "by the batch factor, up to 64x here. Reading tokens_per_s as a "
             "per-sequence rate overstates single-stream speed by exactly B.",
             icon=":material/warning:")
    ns = sorted(f.N.unique())
    n_pick = st.select_slider("KV length", ns, value=ns[len(ns) // 2],
                              key="decode_b_n")
    fb = f[f.N == n_pick]
    c1, c2 = st.columns(2)
    for col, y, lab, ttl in (
            (c1, "us_per_token", "time per decode step (us)",
             "Time per decode step, whole batch"),
            (c2, "tokens_per_s", "aggregate tokens/s across the batch",
             "Aggregate tokens/s across the batch")):
        sub = _med(fb, ["gpu", "implementation", "B"], y).sort_values("B")
        fig = px.line(sub, x="B", y=y, color="implementation", facet_col="gpu",
                      markers=True, log_x=True, log_y=True,
                      color_discrete_map=CMAP,
                      labels={y: lab, "B": "batch"},
                      title="%s (N=%d)" % (ttl, n_pick))
        col.plotly_chart(_facets(fig), use_container_width=True,
                         key="dec_b_%s" % y)
    note("Left: what one request waits. Right: what the server bills. The same "
         "measurement seen two ways, a factor of B apart.")

    # ---------------------------------------------------------------- (c)
    st.header("c. Effective KV bandwidth vs KV length")
    peak = dict(zip(meta.gpu, meta.measured_bw_gbs))
    sub = _med(f, ["gpu", "implementation", "B", "N"], "eff_bw_gbs")
    # The per-GPU roof is drawn as an ordinary flat series rather than a shape,
    # so it lands in the right facet without depending on px's facet ordering.
    roof = (sub[["gpu", "B", "N"]].drop_duplicates()
            .assign(implementation=PEAK,
                    eff_bw_gbs=lambda x: x.gpu.map(peak)).dropna())
    plot = pd.concat([sub, roof])
    plot["batch"] = "B=" + plot.B.astype(str)
    fig = px.line(plot.sort_values(["B", "N"]), x="N", y="eff_bw_gbs",
                  color="implementation", facet_col="gpu", facet_row="batch",
                  markers=True, log_x=True, color_discrete_map=CMAP, height=900,
                  labels={"eff_bw_gbs": "effective KV bandwidth (GB/s)",
                          "N": "KV length"})
    st.plotly_chart(_facets(fig), use_container_width=True, key="dec_c")
    st.caption(
        "The red flat line is that GPU's `measured_peak_bw_gbs` -- an achieved "
        "512MB device-to-device copy, not a theoretical figure "
        "(`theoretical_bw_gbs` is null on all three GPUs). "
        "eff_bw_gbs = kv_bytes / step_time. For D0-naive-kv at Hkv=8, kv_bytes "
        "counts the GQA-expanded cache, 4x the native footprint, so its "
        "bandwidth number is not comparable with the native-layout kernels.")

    # ---------------------------------------------------------------- (d)
    st.header("d. Bandwidth utilisation")
    u = _med(f, ["gpu", "implementation", "B", "N"], "bw_util")
    u["bw_util_pct"] = u.bw_util * 100
    u["batch"] = "B=" + u.B.astype(str)
    fig = px.line(u.sort_values(["B", "N"]), x="N", y="bw_util_pct",
                  color="implementation", facet_col="gpu", facet_row="batch",
                  markers=True, log_x=True, color_discrete_map=CMAP, height=900,
                  labels={"bw_util_pct": "bandwidth utilisation (%)",
                          "N": "KV length"})
    fig.add_hline(y=100, line_dash="dash", line_color="#c0392b")
    st.plotly_chart(_facets(fig), use_container_width=True, key="dec_d")
    over = dec[dec.bw_util > 1]
    st.warning(
        "**%d of %d OK decode rows sit above 100%%** (%s). Nothing has been "
        "clipped. The denominator is the measured 512MB copy benchmark, which "
        "sits BELOW the true read-stream peak -- a pure-read kernel can beat a "
        "copy. Above 100%% means the copy benchmark under-measures the roof, "
        "not that a kernel broke physics. This is not utilisation against a "
        "theoretical peak: no theoretical peak was recorded on any of the three "
        "GPUs." % (len(over), len(dec),
                   ", ".join("%s %d" % (g, n)
                             for g, n in over.groupby("gpu").size().items())
                   or "none in this selection"),
        icon=":material/warning:")

    # ---------------------------------------------------------------- (e)
    st.header("e. MHA vs GQA (Hkv=32 vs Hkv=8)")
    gq = warm[warm.Hq == 32]
    if gq.Hkv.nunique() < 2:
        not_collected("MHA vs GQA",
                      "only one Hkv value survives the current filter.")
    else:
        cnt = dec_all[dec_all.Hq == 32].groupby("Hkv").size()
        st.caption("Hq is 32 throughout, so Hkv=32 is MHA and Hkv=8 is 4:1 GQA. "
                   "This is the solid version of the comparison: %s decode rows "
                   "across %d GPUs, the same shapes on both arms."
                   % (" vs ".join("%d at Hkv=%d" % (v, k)
                                  for k, v in cnt.items()), gq.gpu.nunique()))
        e1, e2 = st.columns(2)
        gpu_e = e1.selectbox("GPU", sorted(gq.gpu.unique()), key="decode_e_gpu")
        be = sorted(gq.B.unique())
        b_e = e2.select_slider("batch", be, value=be[-1], key="decode_e_b")
        ge = gq[(gq.gpu == gpu_e) & (gq.B == b_e)]
        for y, lab in (("median_us", "step latency (us)"),
                       ("eff_bw_gbs", "effective KV bandwidth (GB/s)"),
                       ("kv_bytes", "KV cache bytes (per layer)")):
            s = _med(ge, ["implementation", "Hkv", "N"], y)
            s["Hkv"] = "Hkv=" + s.Hkv.astype(str)
            fig = px.line(s.sort_values("N"), x="N", y=y, color="implementation",
                          facet_col="Hkv", markers=True, log_x=True,
                          log_y=(y != "eff_bw_gbs"), color_discrete_map=CMAP,
                          labels={y: lab, "N": "KV length"},
                          title="%s -- %s, batch %d" % (lab, gpu_e, b_e))
            st.plotly_chart(_facets(fig), use_container_width=True,
                            key="dec_e_%s" % y)
        note("kv_bytes is what each row recorded, so D0-naive-kv reports the "
             "expanded cache at Hkv=8 while the other kernels report the native "
             "one. That 4x gap is a layout choice inside the implementation, "
             "not a measurement of the cache a model must hold.")

    # ---------------------------------------------------------------- (f)
    st.header("f. Decode regime map: which implementation is fastest where")
    grid = _med(f, ["gpu", "implementation", "B", "N"])
    win = grid.loc[grid.groupby(["gpu", "B", "N"]).median_us.idxmin()].copy()
    second = (grid.sort_values("median_us").groupby(["gpu", "B", "N"]).nth(1)
              [["gpu", "B", "N", "median_us"]]
              .rename(columns={"median_us": "runner_up"}))
    win = win.merge(second, on=["gpu", "B", "N"], how="left")
    win["margin"] = win.runner_up / win.median_us
    win["N_"] = win.N.astype(str)
    win["B_"] = win.B.astype(str)
    fig = px.scatter(win, x="N_", y="B_", color="implementation",
                     facet_col="gpu", color_discrete_map=CMAP,
                     symbol_sequence=["square"],
                     category_orders={
                         "N_": [str(v) for v in sorted(f.N.unique())],
                         "B_": [str(v) for v in sorted(f.B.unique())]},
                     hover_data={"median_us": ":.1f", "runner_up": ":.1f",
                                 "margin": ":.2f"},
                     labels={"N_": "KV length", "B_": "batch"}, height=420)
    fig.update_traces(marker_size=34, marker_line_width=0)
    st.plotly_chart(_facets(fig), use_container_width=True, key="dec_f")
    st.caption("Hover gives the winner's absolute step latency in us, the "
               "runner-up's, and the ratio between them -- a tile whose margin "
               "is near 1.00 is a tie, not a win. A crossover reads as a colour "
               "boundary; the same data as lines buries it in overlap.")
    d1 = dec_all[dec_all.implementation == "D1-inductor"]
    if len(d1):
        st.caption("Coverage caveat: D1-inductor is UNSUPPORTED on %.0f%% of its "
                   "attempted decode configurations (%d of %d) -- a deliberate "
                   "bf16-only strip, not a defeat. Where it is missing from a "
                   "tile it never ran there."
                   % (100 * (d1.status == "UNSUPPORTED").mean(),
                      int((d1.status == "UNSUPPORTED").sum()), len(d1)))

    # ---------------------------------------------------------------- (g)
    st.header("g. Eager vs CUDA-graph capture")
    cg = dec[dec.launch == "cudagraph"]
    eg = dec[dec.launch == "eager"]
    if not len(cg):
        not_collected("Eager vs CUDA-graph",
                      "no cudagraph decode rows in this selection.")
    else:
        a = _med(cg.assign(k=cg.cell.str.replace("|cudagraph|", "|eager|",
                                                 regex=False)),
                 ["gpu", "implementation", "k"]).rename(
            columns={"median_us": "cudagraph_us"})
        b = _med(eg.assign(k=eg.cell), ["gpu", "implementation", "k"]).rename(
            columns={"median_us": "eager_us"})
        m = a.merge(b, on=["gpu", "implementation", "k"])
        st.warning(
            "**The two arms use different timers.** The eager arm is "
            "`block_bench` with an inner loop of k calls; the cudagraph arm is "
            "`do_bench_cudagraph`. The delta is %s of eager latency against a "
            "measured event overhead of about %s us, so it cannot be read as "
            "launch overhead: it is a capture-plus-timer difference, and no "
            "Nsight measurement exists anywhere in this repo to separate the "
            "two contributions."
            % ("%.1f%%" % (100 * (1 - (m.cudagraph_us / m.eager_us).median()))
               if len(m) else "n/a",
               "/".join("%g" % v
                        for v in meta.event_overhead_us.dropna().unique()[:3])),
            icon=":material/warning:")
        if not len(m):
            st.info("No cell matched on both arms under the current filters.")
        else:
            m["ratio"] = m.eager_us / m.cudagraph_us
            long = m.melt(["gpu", "implementation", "k"],
                          ["eager_us", "cudagraph_us"],
                          var_name="arm", value_name="us")
            fig = px.box(long, x="implementation", y="us", color="arm",
                         facet_col="gpu", log_y=True, points="all",
                         labels={"us": "median_us -- separate timers, do not "
                                       "read across arms as one axis"},
                         title="Absolute latency on each arm, %d matched cells"
                               % m.k.nunique())
            st.plotly_chart(_facets(fig), use_container_width=True, key="dec_g")
            st.dataframe(
                m.groupby(["gpu", "implementation"]).agg(
                    cells=("k", "nunique"),
                    eager_us=("eager_us", "median"),
                    cudagraph_us=("cudagraph_us", "median"),
                    ratio=("ratio", "median")).round(2).reset_index(),
                hide_index=True, use_container_width=True)
            note("Absolute microseconds on both arms, never a ratio alone. The "
                 "cudagraph arm covers %d cells on %s only."
                 % (m.k.nunique(), ", ".join(sorted(m.gpu.unique()))))

    # ------------------------------------------------------- memory model
    st.divider()
    st.header("KV-cache memory model")
    st.warning(
        "`kv_bytes` here is **per layer, for one attention call**: "
        "2 (K and V) x B x Hkv x N x D x bytes-per-element. Extrapolating it to "
        "a whole model means multiplying by a layer count, and **the layer count "
        "is your assumption sitting on top of our measured per-layer number.** "
        "We measured one layer. Everything past that is your arithmetic.",
        icon=":material/warning:")
    c = st.columns(5)
    layers = c[0].number_input("layers (your assumption)", 1, 256, 32,
                               key="decode_L")
    bmm = c[1].number_input("batch", 1, 4096, 32, key="decode_mB")
    ctx = c[2].number_input("context (KV length)", 1, 1_048_576, 8192,
                            key="decode_mN")
    hkv_m = c[3].number_input("Hkv", 1, 128, 8, key="decode_mH")
    dm = c[4].number_input("head dim", 8, 512, 128, key="decode_mD")
    per_layer = 2 * bmm * hkv_m * ctx * dm * 2      # both dtypes here are 2 bytes
    total = per_layer * layers
    m1, m2, m3 = st.columns(3)
    m1.metric("Per layer (the measured formula)", "%.3f GB" % (per_layer / 1e9))
    m2.metric("x %d layers (your assumption)" % layers, "%.2f GB" % (total / 1e9))
    m3.metric("Per sequence, all layers", "%.3f GB" % (total / bmm / 1e9))
    fit = meta[meta.gpu.isin(dec.gpu.unique())][["gpu", "memory_gb"]].copy()
    fit["KV as % of device memory"] = (100 * total / 1e9 / fit.memory_gb).round(1)
    st.dataframe(fit, hide_index=True, use_container_width=True)
    st.caption("Weights, activations and allocator fragmentation are not in "
               "this figure; it is the KV cache alone. Both dtypes measured in "
               "this sweep are 2 bytes per element.")

    with st.expander("Measured kv_bytes rows behind the formula"):
        st.dataframe(
            dec[["gpu", "implementation", "B", "Hkv", "N", "D", "dt",
                 "gqa_mode", "kv_bytes", "median_us", "eff_bw_gbs", "bw_util"]]
            .drop_duplicates(subset=["gpu", "implementation", "B", "Hkv", "N",
                                     "D", "dt"])
            .sort_values(["gpu", "B", "N"]), hide_index=True,
            use_container_width=True, height=320)

    # Trap 9: page_size equals seq_len on every decode row by construction --
    # one page per sequence -- so D3-vs-D4 compares kernels, never layouts.
    not_collected(
        "Paging overhead",
        "There is no paging axis in this sweep. `page_size` equals `seq_len` on "
        "every decode row by design (one page per sequence), so D3-fa-kvcache "
        "against D4-flashinfer compares two kernels, not paged against "
        "contiguous KV. The paging ablation was never run, and no chart on this "
        "page should be read as measuring it.")
