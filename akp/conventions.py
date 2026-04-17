"""Shared presentation conventions: colours, status meanings, key parsing.

Nothing here defines a metric. analysis.py owns every number; this module only
names and reshapes what it wrote, so the paper and the website cannot drift on
which colour or which status label means what.

cell_key() is the one function that touches data, and it parses rather than
computes: analysis.py builds the composite key and this splits it back.
"""
from __future__ import annotations

import pandas as pd

# Sixteen distinct colours because the prefill registry has ten implementations
# and the decode registry six. An eight-slot palette wraps, which drew
# P1-inductor-nofuse in P4-fa2's colour on any chart spanning the registry.
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
# a configuration it never claimed, OOM_PREDICTED means an allocation computed
# in advance to exceed memory was not attempted, and OOM means one actually
# failed. Pooling them into "failures" would read as a reliability problem
# where most of it is a documented refusal.
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
    "ERROR": "raised at run time",
}


def short_gpu(name: str) -> str:
    return (str(name).replace("NVIDIA ", "")
            .replace("-SXM4-80GB", "").replace(" 80GB HBM3", ""))


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
