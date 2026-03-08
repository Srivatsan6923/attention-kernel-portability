"""Loading and shared conventions for the dashboard.

Every page reads through here, so a filter or a colour means the same thing on
all of them. Nothing in this module defines a metric: analysis.py owns every
number and this only reshapes what it wrote. The one exception is cell_key(),
which splits the composite key analysis.py built -- parsing is not computing.
"""
import json
import os

import pandas as pd
import streamlit as st

PROCESSED = os.environ.get("AKP_PROCESSED", "results/processed")

# Sixteen distinct colours because the prefill registry has ten implementations
# and the decode registry six. The old eight-slot palette wrapped, so
# P1-inductor-nofuse drew in P4-fa2's colour on any chart spanning the registry.
COLOURS = {
    "P0-naive": "#9e9e9e",
    "P1-inductor": "#c98a3a",
    "P1-inductor-nofuse": "#e0b877",
    "P1-inductor-where": "#8c6b3f",
    "P2b-sdpa-mem-eff": "#4c9fd6",
    "P2c-sdpa-flash": "#1f6fb2",
    "P2d-sdpa-cudnn": "#6fc7c1",
    "P3-triton": "#7e57c2",
    "P4-fa2": "#2e9e5b",
    "P4h-fa3": "#17643a",
    "D0-naive-kv": "#9e9e9e",
    "D1-inductor": "#c98a3a",
    "D2-sdpa": "#1f6fb2",
    "D3-fa-kvcache": "#2e9e5b",
    "D4-flashinfer": "#d4547a",
    "D6-fa-prefill-at-1": "#b0a04a",
}

# Status is not a severity scale: UNSUPPORTED means the implementation declined
# a configuration it never claimed, OOM_PREDICTED means we refused to attempt an
# allocation we computed would fail, and OOM means one actually failed. Pooling
# them into "failures" would read as a reliability problem where there is none.
STATUS_COLOURS = {
    "OK": "#2e9e5b",
    "UNSUPPORTED": "#9e9e9e",
    "OOM_PREDICTED": "#c98a3a",
    "OOM": "#c0392b",
    "NUMERICAL_FAIL": "#8e44ad",
    "ERROR": "#c0392b",
    "CAPTURE_FAIL": "#7f8c8d",
}

STATUS_MEANING = {
    "OK": "measured",
    "UNSUPPORTED": "implementation declined this configuration by design",
    "OOM_PREDICTED": "allocation computed in advance to exceed memory; not attempted",
    "OOM": "allocation attempted and failed",
    "NUMERICAL_FAIL": "ran, but failed the 2x-naive correctness gate",
    "ERROR": "raised",
}


def short_gpu(name: str) -> str:
    return (str(name).replace("NVIDIA ", "")
            .replace("-SXM4-80GB", "").replace(" 80GB HBM3", ""))


def _read(name):
    p = os.path.join(PROCESSED, name)
    if not os.path.exists(p):
        return pd.DataFrame() if name.endswith(".parquet") else {}
    if name.endswith(".json"):
        return json.load(open(p, encoding="utf8"))
    return pd.read_parquet(p)


# Keyed on the directory's mtime so a re-analysis invalidates the cache. The
# previous zero-argument cache never expired, so a rebuilt parquet showed the
# old numbers and looked like a data bug.
@st.cache_data(show_spinner=False)
def _load(_stamp):
    d = {
        "rows": _read("rows.parquet"),
        "cells": _read("cells.parquet"),
        "spread": _read("spread.parquet"),
        "inversions": _read("inversions.parquet"),
        "audit": _read("dispatch_audit.parquet"),
        "scaling": _read("scaling.parquet"),
        "portability": _read("portability.parquet"),
        "stability": _read("stability.parquet"),
        "cache_sensitivity": _read("cache_sensitivity.parquet"),
        "oom_calibration": _read("oom_calibration.parquet"),
        "summary": _read("summary.json"),
        "selector": _read("selector.json"),
    }
    for k in ("rows", "cells", "spread", "inversions", "audit"):
        if len(d[k]) and "gpu_name" in d[k]:
            d[k] = d[k].assign(gpu=d[k].gpu_name.map(short_gpu))
    for k in ("rows", "cells"):
        if len(d[k]):
            d[k] = _with_key(d[k])
    return d


def load():
    try:
        stamp = max(os.path.getmtime(os.path.join(PROCESSED, f))
                    for f in os.listdir(PROCESSED))
    except (OSError, ValueError):
        stamp = 0
    return _load(stamp)


def _with_key(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the parsed cell key, minus any field the frame already carries.

    rows.parquet ships its own 'regime' column, so a plain concat left the
    label duplicated and df.regime resolved to a two-column frame -- every
    groupby or comparison against it raised. The parsed value is identical to
    the stored one, so the stored column wins and nothing is recomputed.
    """
    k = cell_key(df.cell)
    return pd.concat([df, k.drop(columns=k.columns.intersection(df.columns))],
                     axis=1)


def cell_key(s: pd.Series) -> pd.DataFrame:
    """Split analysis.py's composite cell key back into its fields.

    regime|B<b>|Hq<q>|Hkv<k>|D<d>|N<n>|dtype|mode|launch|cache|c<0|1>
    """
    p = s.str.split("|", expand=True)
    return pd.DataFrame({
        "regime": p[0],
        "B": p[1].str[1:].astype(int),
        "Hq": p[2].str[2:].astype(int),
        "Hkv": p[3].str[3:].astype(int),
        "D": p[4].str[1:].astype(int),
        "N": p[5].str[1:].astype(int),
        "dt": p[6],
        "md": p[7],
        "lnch": p[8],
        "cch": p[9],
        "csl": p[10].str[1:] == "1",
    }, index=s.index)


def gpu_meta(summary: dict) -> pd.DataFrame:
    """Per-GPU manifest as a frame, one row per device."""
    env = (summary or {}).get("environment") or {}
    out = []
    for g, m in env.items():
        v = m.get("versions") or {}
        out.append({
            "gpu": short_gpu(g), "gpu_name": g,
            "cc": "sm%s%s" % (m.get("cc_major"), m.get("cc_minor")),
            "sm_count": m.get("sm_count"),
            "memory_gb": m.get("total_memory_gb"),
            "l2_mb": (m.get("l2_bytes") or 0) / 1e6 or None,
            "measured_bw_gbs": m.get("measured_peak_bw_gbs"),
            "theoretical_bw_gbs": m.get("theoretical_bw_gbs"),
            "event_overhead_us": m.get("event_overhead_us"),
            "driver": m.get("driver"),
            "cuda": m.get("cuda"),
            "torch": v.get("torch"),
            "triton": v.get("triton"),
            "flash_attn": v.get("flash_attn"),
            "flash_attn_3": v.get("flash_attn_interface"),
            "flashinfer": v.get("flashinfer"),
            "git_sha": (m.get("git_sha") or "")[:8],
        })
    return pd.DataFrame(out).sort_values("cc").reset_index(drop=True)


def provenance(rows: pd.DataFrame) -> pd.DataFrame:
    """Which commit produced which rows, per GPU and regime.

    Not decoration: the decode rows for A10 and A100 come from a different
    commit and driver branch than the H100 ones, so every cross-GPU decode
    delta carries a software difference alongside the architecture one. The
    prefill rows are all from one commit. A reader has to be able to see that.
    """
    if not len(rows) or "git_sha" not in rows:
        return pd.DataFrame()
    g = (rows.groupby(["gpu", "regime", rows.git_sha.str[:8]])
         .size().rename("rows").reset_index()
         .rename(columns={"git_sha": "commit"}))
    return g.sort_values(["regime", "gpu"])


def ok(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows.status == "OK"] if len(rows) else rows


def note(msg: str):
    st.caption(":grey[%s]" % msg)


def not_collected(title: str, why: str):
    """An honest placeholder. A panel that silently disappears reads as though
    the question was never asked."""
    st.subheader(title)
    st.info("**Not collected.** %s" % why, icon=":material/block:")


def filter_bar(df: pd.DataFrame, key: str, fields=("gpu", "dt", "D", "B", "N", "md"),
               default_gpu=None):
    """Shared filter widgets. Returns the filtered frame.

    Every page uses the same widget set so a reader carries one mental model
    across pages, and every widget key is namespaced by page.
    """
    if not len(df):
        return df
    labels = {"gpu": "GPU", "dt": "dtype", "D": "head dim", "B": "batch",
              "N": "seq / KV length", "md": "mode", "Hkv": "KV heads",
              "lnch": "launch", "cch": "cache", "implementation": "implementation"}
    cols = st.columns(len(fields))
    out = df
    for c, f in zip(cols, fields):
        if f not in out.columns:
            continue
        vals = sorted(out[f].dropna().unique().tolist())
        if len(vals) <= 1:
            c.caption("%s: %s" % (labels.get(f, f), vals[0] if vals else "-"))
            continue
        pre = vals
        if f == "gpu" and default_gpu and default_gpu in vals:
            pre = [default_gpu]
        sel = c.multiselect(labels.get(f, f), vals, default=pre, key="%s_%s" % (key, f))
        if sel:
            out = out[out[f].isin(sel)]
    return out
