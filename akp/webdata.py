"""Aggregate results/processed into one compact JSON the report page embeds.

    python -m akp.webdata results/processed results/figures/web.json

The page renders its charts from this file, so every number on the site comes
from the same parquet analysis.py wrote. Nothing here defines a metric: it
selects slices, rounds, and reshapes. Slices are chosen by coverage rather than
hardcoded, and the chosen slice travels with the data so a caption can state it.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from akp.analysis import (PRACTICAL, paired_ratio_ci, per_cell_median, usable)

SHORT = {"NVIDIA A10": "A10", "NVIDIA A100-SXM4-80GB": "A100",
         "NVIDIA H100 80GB HBM3": "H100", "NVIDIA L40": "L40",
         "NVIDIA L40S": "L40S", "NVIDIA GeForce RTX 5090": "RTX 5090",
         "NVIDIA GeForce RTX 4090": "RTX 4090"}
# Compute capability orders the devices everywhere on the page. Alphabetical
# would put the Ada parts between the two Ampere ones and imply nothing.
CC_ORDER = ["A100", "A10", "L40", "L40S", "H100", "RTX 5090"]


def short(g):
    return SHORT.get(g, str(g).replace("NVIDIA ", ""))


def order(gs):
    return [g for g in CC_ORDER if g in set(gs)] + \
           sorted(set(gs) - set(CC_ORDER))


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else round(float(x), 3)


def best_slice(df, keys):
    """The slice with the widest device and implementation coverage.

    Picked from the data rather than hardcoded: which batch and head dim are
    best covered changes as devices finish, and a caption that names the slice
    has to name the one actually plotted.
    """
    g = (df.groupby(keys)
           .agg(gpus=("gpu_name", "nunique"), impls=("implementation", "nunique"),
                lens=("seq_len", "nunique"), n=("median_us", "size"))
           .sort_values(["gpus", "impls", "lens", "n"], ascending=False))
    return dict(zip(keys, g.index[0])) if len(g) else {}


def series(df, slice_, value="median_us"):
    d = df.copy()
    for k, v in slice_.items():
        d = d[d[k] == v]
    out = {}
    for (g, impl), s in d.groupby(["gpu_name", "implementation"]):
        s = s.groupby("seq_len")[value].median().sort_index()
        out.setdefault(short(g), {})[impl] = {
            "x": [int(x) for x in s.index], "y": [r3(v) for v in s.values]}
    return out


def winner_grid(df, axis_y="batch"):
    """Fastest implementation over (seq_len, batch) per GPU."""
    w = df.loc[df.groupby(["gpu_name", "seq_len", axis_y]).median_us.idxmin()]
    out = {}
    for g, s in w.groupby("gpu_name"):
        out[short(g)] = [{"N": int(r.seq_len), "y": int(getattr(r, axis_y)),
                          "impl": r.implementation, "us": r3(r.median_us)}
                         for r in s.itertuples()]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", nargs="?", default="results/processed")
    ap.add_argument("out", nargs="?", default="results/figures/web.json")
    a = ap.parse_args(argv)
    P = a.processed

    rows = pd.read_parquet(os.path.join(P, "rows.parquet"))
    summ = json.load(open(os.path.join(P, "summary.json"), encoding="utf8"))
    sel = json.load(open(os.path.join(P, "selector.json"), encoding="utf8"))
    audit = pd.read_parquet(os.path.join(P, "dispatch_audit.parquet"))
    ok = rows[rows.status == "OK"]
    cells = per_cell_median(usable(rows))
    cells["regime"] = cells.cell.str.split("|").str[0]

    out = {"generated_from": P}

    # ---- devices ---------------------------------------------------------
    env = summ.get("environment", {})
    out["devices"] = [{
        "gpu": short(g), "cc": "sm%s%s" % (m.get("cc_major"), m.get("cc_minor")),
        "arch": {"80": "Ampere", "86": "Ampere", "89": "Ada", "90": "Hopper",
                 "120": "Blackwell"}.get("%s%s" % (m.get("cc_major"), m.get("cc_minor")), "?"),
        "sm": m.get("sm_count"), "mem_gb": r3(m.get("total_memory_gb")),
        "l2_mb": r3((m.get("l2_bytes") or 0) / 1e6) or None,
        "bw_gbs": r3(m.get("measured_peak_bw_gbs")),
        "event_us": r3(m.get("event_overhead_us")),
        "driver": m.get("driver"), "sha": (m.get("git_sha") or "")[:8],
    } for g, m in env.items() if short(g) in set(rows.gpu_name.map(short))]
    # A manifest exists for every device a pod ever started on, including one
    # that produced no rows. Listing it would claim a measurement we do not have.
    # Resolve the ordering before sorting: a key that reads the list being
    # sorted sees it half-permuted.
    _ord = order([x["gpu"] for x in out["devices"]])
    out["devices"].sort(key=lambda d: _ord.index(d["gpu"]))
    out["software"] = (list(env.values())[0].get("versions") if env else {}) or {}

    # ---- headline --------------------------------------------------------
    aud = audit[audit.probed]
    out["headline"] = {
        "cells": int(cells.groupby(["gpu_name", "cell"]).ngroups),
        "rows": int(len(rows)),
        "ok_rows": int(len(ok)),
        "devices": len(out["devices"]),
        "dispatch_mismatch_pct": r3(100 * (~aud.matched.fillna(True).astype(bool)).mean()),
        "probed_pct": r3(100 * audit.probed.mean()),
        "selector_median_regret": r3(sel.get("median_regret")),
        "selector_p95_regret": r3(sel.get("p95_regret")),
    }

    # ---- winner separation and flip rates --------------------------------
    win = []
    for (g, c), sub in cells.groupby(["gpu_name", "cell"]):
        s = sub.sort_values(["median_us", "implementation"])
        w = s.iloc[0]
        rec = {"gpu": short(g), "cell": c, "regime": c.split("|")[0],
               "winner": w.implementation, "us": r3(w.median_us), "sep": False,
               "ratio": None}
        if len(s) > 1:
            ru = s.iloc[1]
            ratio = float(ru.median_us / w.median_us)
            kw = (dict(ids_a=ru.repeat_ids, ids_b=w.repeat_ids)
                  if "repeat_ids" in sub.columns else {})
            _ratio, lo, hi, _n = paired_ratio_ci(ru.samples, w.samples, **kw)
            rec["ratio"] = r3(ratio)
            rec["runner_up"] = ru.implementation
            rec["sep"] = bool((lo > 1 or hi < 1) and ratio >= PRACTICAL)
        win.append(rec)
    W = pd.DataFrame(win)
    out["separation"] = {
        "overall": r3(W.sep.mean()),
        "by_regime": {reg: {"n": int(len(s)), "separated": r3(s.sep.mean()),
                            "median_ratio": r3(s.ratio.median())}
                      for reg, s in W.groupby("regime")},
        "practical_threshold": PRACTICAL}

    wmap = {(r.gpu, r.cell): (r.winner, r.sep) for r in W.itertuples()}
    pairs = []
    gs = order(W.gpu.unique())
    for reg in sorted(W.regime.unique()):
        for i, g1 in enumerate(gs):
            for g2 in gs[i + 1:]:
                shared = [c for c in W[(W.gpu == g1) & (W.regime == reg)].cell
                          if (g2, c) in wmap]
                if len(shared) < 5:
                    continue
                flip = sum(1 for c in shared if wmap[(g1, c)][0] != wmap[(g2, c)][0])
                sep = [c for c in shared if wmap[(g1, c)][1] and wmap[(g2, c)][1]]
                sflip = sum(1 for c in sep if wmap[(g1, c)][0] != wmap[(g2, c)][0])
                rc = (summ.get("rank_correlation") or {}).get(
                    "%s vs %s" % (g1, g2), {})
                pairs.append({"regime": reg, "a": g1, "b": g2,
                              "n": len(shared), "flip": r3(flip / len(shared)),
                              "n_sep": len(sep),
                              "flip_sep": r3(sflip / len(sep)) if sep else None,
                              "spearman": r3(rc.get("spearman"))})
    out["pairs"] = pairs

    # ---- winners per device ---------------------------------------------
    out["winners"] = {reg: {g: s.winner.value_counts().to_dict()
                            for g, s in d.groupby("gpu")}
                      for reg, d in W.groupby("regime")}

    # ---- coverage / status ----------------------------------------------
    cov = (rows.assign(gpu=rows.gpu_name.map(short))
               .groupby(["gpu", "regime", "implementation", "status"]).size()
               .rename("n").reset_index())
    out["coverage"] = cov.to_dict("records")

    # ---- dispatch --------------------------------------------------------
    ad = audit.assign(gpu=audit.gpu_name.map(short))
    out["dispatch"] = {
        "flows": [{"impl": i, "observed": o, "n": int(n)} for (i, o), n
                  in ad.groupby(["implementation", "observed"]).size().items()],
        "by_gpu": {g: {"probed": r3(100 * s.probed.mean()),
                       "mismatch": r3(100 * (~s[s.probed].matched.fillna(True).astype(bool)).mean())}
                   for g, s in ad.groupby("gpu")},
        "inductor_counter_fired": int(ad.fused_into_sdpa.sum()),
        "inductor_rows": int(ad.implementation.str.startswith(("P1-", "D1-")).sum()),
        "attn_kernel_in_trace": int(ad.get("attn_kernel_in_trace", pd.Series(dtype=bool)).sum()),
    }

    # ---- correctness -----------------------------------------------------
    gate = ok[ok.max_abs_err.notna()]
    out["correctness"] = [{
        "impl": i,
        "n": int(len(s)),
        "max_abs": r3(s.max_abs_err.max()),
        "max_rel": r3(s.max_rel_err.max()),
        "min_cos": r3(s.cosine_sim.min()),
        "vs_baseline": r3((s.max_abs_err / s.baseline_max_abs_err.replace(0, np.nan)).max()),
        "fail": int((s.pass_relative == False).sum()),  # noqa: E712
    } for i, s in gate.groupby("implementation")]

    # ---- prefill ---------------------------------------------------------
    pre = ok[ok.regime == "prefill"]
    sl = best_slice(pre, ["batch", "head_dim", "dtype", "mode"])
    out["prefill_slice"] = {k: (int(v) if isinstance(v, (np.integer, int)) else v)
                            for k, v in sl.items()}
    out["prefill_latency"] = series(pre, sl)
    out["prefill_tflops"] = series(pre, sl, "tflops")
    out["prefill_memory"] = series(pre, sl, "peak_allocated_mb")

    base = "P2c-sdpa-flash"
    sp = pre[(pre.head_dim == sl["head_dim"]) & (pre.dtype == sl["dtype"])
             & (pre["mode"] == sl["mode"])]
    piv = sp.groupby(["gpu_name", "batch", "seq_len", "implementation"]).median_us.median()
    out["prefill_speedup"] = []
    for (g, b, n), s in piv.groupby(level=[0, 1, 2]):
        s = s.droplevel([0, 1, 2])
        if base not in s:
            continue
        for impl, v in s.items():
            if impl == base:
                continue
            out["prefill_speedup"].append(
                {"gpu": short(g), "B": int(b), "N": int(n), "impl": impl,
                 "speedup": r3(s[base] / v), "us": r3(v), "base_us": r3(s[base])})

    # ---- decode ----------------------------------------------------------
    dec = ok[(ok.regime == "decode") & (ok.launch == "eager")]
    ds = best_slice(dec, ["hkv", "head_dim", "dtype"])
    out["decode_slice"] = {k: (int(v) if isinstance(v, (np.integer, int)) else v)
                           for k, v in ds.items()}
    for b in (1, 32):
        d = dec[dec.batch == b]
        out["decode_latency_b%d" % b] = series(d, ds)
        out["decode_bw_b%d" % b] = series(d, ds, "eff_bw_gbs")
    out["decode_winner_map"] = winner_grid(
        dec[(dec.hkv == ds["hkv"]) & (dec.head_dim == ds["head_dim"])
            & (dec.dtype == ds["dtype"])])

    # MHA vs GQA at the same shape: Hkv 32 against Hkv 8.
    mg = dec[(dec.head_dim == ds["head_dim"]) & (dec.dtype == ds["dtype"])
             & dec.hkv.isin([8, 32])]
    out["mha_vs_gqa"] = [{
        "gpu": short(g), "hkv": int(h), "N": int(n), "impl": i,
        "us": r3(s.median_us.median()), "kv_mb": r3(s.kv_bytes.median() / 1e6)}
        for (g, h, n, i), s in mg.groupby(["gpu_name", "hkv", "seq_len", "implementation"])]

    # ---- roofline (analytic traffic; no profiler data exists) ------------
    rf = ok[ok.arith_intensity.notna() & ok.tflops.notna()] if "arith_intensity" in ok else ok.iloc[:0]
    if len(rf):
        s = rf.groupby(["gpu_name", "regime", "implementation"]).agg(
            ai=("arith_intensity", "median"), tf=("tflops", "median"))
        out["roofline"] = [{"gpu": short(g), "regime": reg, "impl": i,
                            "ai": r3(row.ai), "tflops": r3(row.tf)}
                           for (g, reg, i), row in s.iterrows()]
    out["roofline_is_modelled"] = True

    # ---- selector --------------------------------------------------------
    out["selector"] = {
        "accuracy": r3(sel.get("accuracy")), "n": sel.get("n_heldout"),
        "median_regret": r3(sel.get("median_regret")),
        "p95_regret": r3(sel.get("p95_regret")),
        "rule": sel.get("rule", ""),
        "vs_fixed": {k: {"median": r3(v.get("median_speedup")),
                         "p95": r3(v.get("p95_speedup")),
                         "coverage": r3(v.get("coverage"))}
                     for k, v in (sel.get("vs_fixed") or {}).items()}}

    out["spread"] = summ.get("spread_by_regime", {})
    out["status_counts"] = summ.get("status", {})
    out["not_collected"] = [
        "Nsight Compute and Nsight Systems: profiling is blocked by pod security "
        "policy on the cluster and declined by the cloud host, so no hardware "
        "counters exist. The roofline here uses analytic traffic, not measured.",
        "Paged versus contiguous KV layout: page_size equals the KV length in "
        "every decode cell by construction, so the paged and contiguous entries "
        "differ by library, not by layout. The page-size sweep was not run.",
    ]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf8"), separators=(",", ":"))
    print("wrote %s (%.0f kB)" % (a.out, os.path.getsize(a.out) / 1024))
    print("devices:", ", ".join(d["gpu"] for d in out["devices"]))
    print("prefill slice:", out["prefill_slice"], "| decode slice:", out["decode_slice"])
    print("separation overall: %.3f" % out["separation"]["overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
