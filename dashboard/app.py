"""Results dashboard.

Reads only results/processed/, which akp.analysis writes. Nothing is recomputed
here, so a number in the dashboard and the same number in the report cannot
disagree. The single exception is documented where it lives: overview.py splits
the pooled inversion rate by regime, because summary.json stores only the pooled
one, and its self-check asserts the split reproduces analysis.py's figure.

    python -m akp.analysis
    streamlit run dashboard/app.py

This module is a router and nothing else: every page lives in views/ and is
handed the one loaded dict, so there is a single place where the data is read.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys

# streamlit runs this file as a script, so dashboard/ is not importable by
# default and the views' `from _data import ...` would fail when the server is
# started from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(page_title="Attention Kernel Portability", layout="wide")

import _data  # noqa: E402
from views import (attribution, correctness, decode, overview,  # noqa: E402
                   portability, prefill, reproducibility)

PAGES = {
    "Overview": overview,
    "Prefill": prefill,
    "Decode & KV cache": decode,
    "Correctness & dispatch": correctness,
    "Attribution": attribution,
    "Portability & selector": portability,
    "Methodology": reproducibility,
}


def _freshness():
    """Row count, devices and how old the processed set is."""
    try:
        stamp = max(os.path.getmtime(os.path.join(_data.PROCESSED, f))
                    for f in os.listdir(_data.PROCESSED))
        when = _dt.datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError):
        when = "missing"
    return when


D = _data.load()
page = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")

rows = D["rows"]
gpus = (sorted(rows.gpu.unique()) if len(rows) and "gpu" in rows
        else (D["summary"].get("gpus") or []))
st.sidebar.divider()
st.sidebar.caption(
    "**%s** measured rows\n\n%s\n\n`%s` written %s"
    % (f"{len(rows):,}", ", ".join(map(str, gpus)) or "no devices",
       _data.PROCESSED, _freshness()))

PAGES[page].render(D)
