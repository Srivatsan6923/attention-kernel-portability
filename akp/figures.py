"""Regenerate every paper figure from results/processed, deterministically.

    python -m akp.figures results/processed

Nothing here defines a metric. analysis.py owns the numbers and this only draws
what it wrote, with two exceptions that are parsing rather than computing: the
cell key (split by dashboard._data.cell_key) and results_live/provenance.txt.
The one genuine computation is the winner/runner-up separation used to hatch
fig1, and it reuses analysis.py's own bootstrap and practical threshold rather
than inventing a second definition of "separated".

Determinism: the bootstrap seed is analysis.py's fixed one, every ordering is an
explicit sort, and savefig writes no CreationDate, so re-running on unchanged
input reproduces the PDFs byte for byte.

Every figure facets by regime (or is single-regime by construction): decode
tflops sit two to three orders below prefill, and fwd and fwd_bwd never share an
axis because the 3.5x backward factor in tflops is an assumption, not a
measurement.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import NullFormatter

from akp.analysis import (PRACTICAL, paired_ratio_ci, per_cell_median, usable)
# ponytail: the dashboard already owns the colour map and the cell-key parser,
# so the paper and the dashboard cannot drift. Cost is a streamlit import.
from dashboard._data import COLOURS, cell_key, short_gpu

DPI = 300

# Okabe-Ito, for the categorical scales this module invents (provenance status,
# inversion class). The per-implementation colours come from the dashboard and
# are disambiguated for colour-blind and greyscale readers by hatching, which
# is the only channel that survives a black-and-white print.
OKABE = ["#0072b2", "#e69f00", "#009e73", "#d55e00",
         "#cc79a7", "#56b4e9", "#f0e442", "#000000"]

# One pattern per implementation, in registry order. No empty pattern: an
# ambiguous cell is drawn as hatch-on-white, and a blank one would read as
# missing data.
PATTERNS = ["///", "\\\\\\", "xxx", "...", "+++", "ooo", "***", "|||", "---", "OO"]
HATCH = {impl: PATTERNS[i % len(PATTERNS)]
         for i, impl in enumerate([k for k in COLOURS if k.startswith("P")])}
HATCH.update({impl: PATTERNS[i % len(PATTERNS)]
              for i, impl in enumerate([k for k in COLOURS if k.startswith("D")])})

REGIMES = ("prefill", "decode")


def _style():
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embed TrueType for LaTeX
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "axes.axisbelow": True, "hatch.linewidth": 0.4,
        "figure.constrained_layout.use": True,
    })


def save(fig, out, name):
    p = os.path.join(out, name)
    # metadata=CreationDate:None -> byte-reproducible PDFs.
    fig.savefig(p, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print("wrote %s (%.0f kB)" % (p, os.path.getsize(p) / 1e3))
    return p


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load(proc: str) -> dict:
    def read(name):
        p = os.path.join(proc, name)
        if not os.path.exists(p):
            return {} if name.endswith(".json") else pd.DataFrame()
        if name.endswith(".json"):
            return json.load(open(p, encoding="utf8"))
        return pd.read_parquet(p)

    d = {n: read(n + ".parquet") for n in
         ("rows", "cells", "spread", "inversions", "portability")}
    d["summary"] = read("summary.json")
    d["selector"] = read("selector.json")
    for k in ("rows", "cells", "spread", "inversions"):
        if len(d[k]) and "gpu_name" in d[k]:
            d[k] = d[k].assign(gpu=d[k].gpu_name.map(short_gpu))
    for k in ("rows", "cells", "spread", "inversions"):
        if len(d[k]) and "cell" in d[k]:
            key = cell_key(d[k].cell)
            d[k] = pd.concat(
                [d[k], key.drop(columns=key.columns.intersection(d[k].columns))],
                axis=1)
    return d


def gpu_order(d: dict) -> list:
    """GPUs ordered by compute capability, so a column position means an
    architecture generation rather than an alphabetical accident."""
    env = (d["summary"] or {}).get("environment") or {}
    seen = sorted(set(d["cells"].gpu_name)) if len(d["cells"]) else []
    def cc(g):
        m = env.get(g, {})
        return (m.get("cc_major") or 0) * 10 + (m.get("cc_minor") or 0)
    return sorted(seen, key=lambda g: (cc(g), g))


def peak_bw(d: dict) -> dict:
    """MEASURED peak bandwidth per GPU. Never the theoretical number: the
    manifest records a measured copy rate and only falls back if it is absent,
    so a figure that said "theoretical" would be mislabelling its own line."""
    env = (d["summary"] or {}).get("environment") or {}
    return {g: m.get("measured_peak_bw_gbs") for g, m in env.items()
            if m.get("measured_peak_bw_gbs")}


def shared_cells(cells: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Cells every GPU that ran this regime actually ran.

    Matching matters: an implementation or a device that is absent where it
    would have lost buys portability by not showing up.
    """
    d = cells[cells.regime == regime]
    if not len(d):
        return d
    n = d.groupby("cell").gpu_name.nunique()
    return d[d.cell.isin(n[n == d.gpu_name.nunique()].index)]


# --------------------------------------------------------------------------- #
# Winner separation (the only quantity this module computes)
# --------------------------------------------------------------------------- #

def winner_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (gpu, cell): the fastest implementation, the runner-up, and whether
    the gap clears BOTH thresholds -- a cluster bootstrap CI on the ratio that
    excludes 1, and a ratio of at least PRACTICAL. A winner that clears neither
    is a coin flip dressed as a result, which is what fig1 hatches.
    """
    cells = per_cell_median(usable(rows))
    out = []
    for (g, c), sub in cells.groupby(["gpu_name", "cell"], sort=True):
        sub = sub.sort_values(["median_us", "implementation"])
        w = sub.iloc[0]
        rec = dict(gpu_name=g, cell=c, regime=c.split("|")[0],
                   winner=w.implementation, winner_us=float(w.median_us),
                   n_impls=len(sub))
        if len(sub) > 1:
            r = sub.iloc[1]
            ratio = float(r.median_us / w.median_us)
            # Paired over process launches, with ratio and interval taken
            # from one population, and separation requiring lo > 1.
            kw = (dict(ids_a=r.repeat_ids, ids_b=w.repeat_ids)
                  if "repeat_ids" in sub.columns else {})
            ratio, lo, hi, n_launch = paired_ratio_ci(r.samples, w.samples, **kw)
            rec.update(runner_up=r.implementation, runner_up_us=float(r.median_us),
                       ratio=ratio, n_launch=int(n_launch),
                       sig=bool(np.isfinite(lo) and lo > 1),
                       practical=bool(np.isfinite(ratio) and ratio >= PRACTICAL))
        else:
            rec.update(runner_up=None, runner_up_us=np.nan, ratio=np.nan,
                       sig=False, practical=False)
        out.append(rec)
    d = pd.DataFrame(out)
    d["separated"] = d.sig & d.practical
    return d


def _row_label(r) -> str:
    # r["dt"], not r.dt: on a Series, .dt is pandas' datetime accessor and
    # raises rather than returning the dtype field of the cell key.
    if r.regime == "prefill":
        return "%s %s D%d B%d" % (r["md"], r["dt"], r["D"], r["B"])
    return "%s D%d Hkv%d B%d" % (r["dt"], r["D"], r["Hkv"], r["B"])


SORT_KEYS = ["md", "dt", "D", "Hkv", "B", "N"]
BLOCK_KEYS = ["md", "dt", "D", "Hkv", "B"]


# --------------------------------------------------------------------------- #
# fig1: the winner map
# --------------------------------------------------------------------------- #

def fig1_winner_map(d: dict, wins: pd.DataFrame, out: str) -> str:
    gpus = gpu_order(d)
    panels = []
    for reg in REGIMES:
        sh = shared_cells(d["cells"], reg)
        if not len(sh):
            continue
        order = (sh.drop_duplicates("cell").sort_values(SORT_KEYS + ["cell"])
                 .reset_index(drop=True))
        panels.append((reg, order, [g for g in gpus if g in set(sh.gpu_name)]))
    if not panels:
        return print("skipped fig1: no cell is shared by every GPU")

    nrows = sum(len(o) for _, o, _ in panels)
    height = 1.9 + 0.040 * nrows
    fig = plt.figure(figsize=(7.6, height))
    gs = fig.add_gridspec(len(panels), 1, height_ratios=[len(o) for _, o, _ in panels])
    panel_in = [(height - 1.9) * len(o) / nrows for _, o, _ in panels]
    w = wins.set_index(["gpu_name", "cell"])

    for ax_i, (reg, order, gs_list) in enumerate(panels):
        ax = fig.add_subplot(gs[ax_i])
        used = []
        for y, r in order.iterrows():
            for x, g in enumerate(gs_list):
                try:
                    rec = w.loc[(g, r.cell)]
                except KeyError:
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#f2f2f2",
                                           edgecolor="white", linewidth=0.3))
                    continue
                impl = rec.winner
                used.append(impl)
                col = COLOURS.get(impl, "#333333")
                if rec.separated:
                    # Hatch in white over the fill: the texture is what a
                    # greyscale print has to tell implementations apart, and a
                    # dark hatch at this row height swallows the colour.
                    ax.add_patch(Rectangle(
                        (x, y), 1, 1, facecolor=col, edgecolor="#ffffffcc",
                        hatch=HATCH.get(impl), linewidth=0.25))
                else:
                    # Not separated from #2 by both thresholds: hatch on white,
                    # so the claim "this kernel won here" is visibly weaker.
                    ax.add_patch(Rectangle(
                        (x, y), 1, 1, facecolor="white", edgecolor=col,
                        hatch=HATCH.get(impl), linewidth=0.35))

        ax.set_xlim(0, len(gs_list))
        ax.set_ylim(len(order), 0)
        ax.set_xticks(np.arange(len(gs_list)) + 0.5)
        ax.set_xticklabels([short_gpu(g) for g in gs_list], fontsize=7)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        blocks = order[BLOCK_KEYS].ne(order[BLOCK_KEYS].shift()).any(axis=1)
        idx = list(order.index[blocks])
        for i in idx[1:]:
            ax.axhline(i, color="#999999", linewidth=0.4)
        # Every block gets a separator; only the ones that clear the label
        # height get a label, so ticks never overprint each other.
        rows_per_label = max(1, int(np.ceil(len(order) / (panel_in[ax_i] * 72 / 6.0))))
        keep, last = [], -10 ** 9
        for i in idx:
            if i - last >= rows_per_label:
                keep.append(i)
                last = i
        ax.set_yticks([i + 0.5 for i in keep])
        ax.set_yticklabels([_row_label(order.loc[i]) for i in keep], fontsize=5)
        ax.tick_params(length=1.5, width=0.4)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(0.5)

        n_amb = int((~w.loc[[(g, c) for g in gs_list for c in order.cell
                             if (g, c) in w.index]].separated).sum())
        ax.set_ylabel("%s: %d matched configurations\n(N ascending within each block)"
                      % (reg, len(order)), fontsize=7)
        ax.set_title("%s -- %d configurations x %d GPUs, %.0f%% of cells have a "
                     "winner separated from #2" %
                     (reg, len(order), len(gs_list),
                      100 * (1 - n_amb / max(1, len(order) * len(gs_list)))),
                     fontsize=8, pad=16)

        handles = [Patch(facecolor=COLOURS.get(i, "#333"), edgecolor="#00000055",
                         hatch=HATCH.get(i), label=i)
                   for i in sorted(set(used))]
        handles.append(Patch(facecolor="white", edgecolor="#444444", hatch="///",
                             label="winner NOT separated from #2\n(bootstrap CI "
                                   "on the ratio excludes 1 AND ratio >= %.2f)"
                                   % PRACTICAL))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0, frameon=False, handlelength=2.4,
                  labelspacing=0.35)

    fig.suptitle("Fastest attention implementation per configuration and GPU\n"
                 "(kernel, GPU, regime, software-distribution snapshot: "
                 "torch %s / CUDA %s / flash-attn %s / flashinfer %s)"
                 % _versions(d), fontsize=9)
    return save(fig, out, "fig1_winner_map.pdf")


def _versions(d: dict):
    env = (d["summary"] or {}).get("environment") or {}
    v = {}
    for m in env.values():
        v.update({k: x for k, x in (m.get("versions") or {}).items() if x})
        v.setdefault("cuda", m.get("cuda"))
    return (v.get("torch", "?"), v.get("cuda", "?"),
            v.get("flash_attn", "?"), v.get("flashinfer", "?"))


# --------------------------------------------------------------------------- #
# fig2 / fig3: latency and bandwidth
# --------------------------------------------------------------------------- #

def _densest_slice(d: pd.DataFrame, keys) -> dict:
    """The fixed-parameter slice with the most (gpu, impl, N) points.

    A latency-vs-N line has to hold batch, head dim and dtype fixed or it is a
    plot of three variables at once; picking the best-covered slice from the
    data keeps the choice out of the source.
    """
    g = (d.groupby(list(keys))
         .apply(lambda s: s.groupby(["gpu_name", "implementation", "N"]).ngroups,
                include_groups=False)
         .sort_values(ascending=False))
    top = g.index[0]
    return dict(zip(keys, top if isinstance(top, tuple) else (top,)))


def _slice(d: pd.DataFrame, sel: dict) -> pd.DataFrame:
    m = np.ones(len(d), bool)
    for k, v in sel.items():
        m &= (d[k] == v).values
    return d[m]


def _fixed_label(sel: dict) -> str:
    return ", ".join("%s=%s" % (k, v) for k, v in sel.items())


def fig2_prefill_latency(d: dict, out: str) -> str:
    pre = d["cells"][d["cells"].regime == "prefill"]
    if not len(pre):
        return print("skipped fig2: no prefill cells in this dataset")
    sel = _densest_slice(pre, ["B", "D", "dt"])
    sub = _slice(pre, sel)
    gpus = [g for g in gpu_order(d) if g in set(sub.gpu_name)]
    modes = [m for m in ("fwd", "fwd_bwd") if m in set(sub.md)]

    fig, axes = plt.subplots(len(modes), len(gpus), squeeze=False,
                             figsize=(2.05 * len(gpus) + 1.6, 2.5 * len(modes)),
                             sharex=True)
    for i, md in enumerate(modes):
        # sharey WITHIN a mode row only: fwd_bwd is a different measurement,
        # and putting it on a fwd axis would invite reading the gap as a speedup.
        base = None
        for j, g in enumerate(gpus):
            ax = axes[i][j]
            if base is None:
                base = ax
            else:
                ax.sharey(base)
            s = sub[(sub.md == md) & (sub.gpu_name == g)]
            for impl in sorted(set(s.implementation)):
                t = (s[s.implementation == impl].groupby("N").median_us.median()
                     .sort_index())
                ax.plot(t.index, t.values, marker="o", markersize=2.6,
                        linewidth=1.0, color=COLOURS.get(impl, "#333"), label=impl)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title("%s -- %s" % (short_gpu(g), md), fontsize=8)
            if i == len(modes) - 1:
                ax.set_xlabel("sequence length N (tokens)")
            if j == 0:
                ax.set_ylabel("median latency (µs, log)")
            else:
                ax.tick_params(labelleft=False)

    handles = [Line2D([], [], color=COLOURS.get(i, "#333"), marker="o",
                      markersize=3, linewidth=1.0, label=i)
               for i in sorted(set(sub.implementation))]
    fig.legend(handles=handles, loc="outside center right", frameon=False)
    fig.suptitle("Prefill latency vs sequence length (causal, %s)\n"
                 "log-log; each row of panels has its own y axis -- fwd and "
                 "fwd_bwd are different measurements" % _fixed_label(sel),
                 fontsize=9)
    return save(fig, out, "fig2_prefill_latency.pdf")


def fig3_decode_bandwidth(d: dict, out: str) -> str:
    rows = usable(d["rows"])
    dec = rows[(rows.regime == "decode") & rows.eff_bw_gbs.notna()]
    if not len(dec):
        return print("skipped fig3: no decode rows with effective bandwidth")
    sel = _densest_slice(dec, ["B", "D", "Hkv", "dt"])
    sub = _slice(dec, sel)
    gpus = [g for g in gpu_order(d) if g in set(sub.gpu_name)]
    peaks = peak_bw(d)

    fig, axes = plt.subplots(1, len(gpus), squeeze=False, sharex=True, sharey=True,
                             figsize=(2.15 * len(gpus) + 1.7, 2.9))
    for j, g in enumerate(gpus):
        ax = axes[0][j]
        s = sub[sub.gpu_name == g]
        for impl in sorted(set(s.implementation)):
            t = (s[s.implementation == impl].groupby("N").eff_bw_gbs.median()
                 .sort_index())
            ax.plot(t.index, t.values, marker="o", markersize=2.6, linewidth=1.0,
                    color=COLOURS.get(impl, "#333"), label=impl)
        p = peaks.get(g)
        if p:
            ax.axhline(p, color="#000000", linestyle="--", linewidth=0.9)
            ax.annotate("measured peak %.0f GB/s" % p, (0.02, p), xycoords=
                        ("axes fraction", "data"), va="bottom", fontsize=6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(short_gpu(g), fontsize=8)
        ax.set_xlabel("KV-cache length (tokens)")
        if j == 0:
            ax.set_ylabel("effective KV bandwidth (GB/s, log)")
        else:
            ax.tick_params(labelleft=False)

    handles = [Line2D([], [], color=COLOURS.get(i, "#333"), marker="o",
                      markersize=3, linewidth=1.0, label=i)
               for i in sorted(set(sub.implementation))]
    handles.append(Line2D([], [], color="#000000", linestyle="--",
                          label="measured peak copy bandwidth (this host)"))
    fig.legend(handles=handles, loc="outside center right", frameon=False)
    fig.suptitle("Decode: effective KV-cache bandwidth vs KV length (%s)\n"
                 "bytes = KV actually streamed / measured latency; the dashed "
                 "line is each device's MEASURED peak, not a vendor figure;\n"
                 "points above it mean the KV slice was served from cache "
                 "rather than streamed from HBM" % _fixed_label(sel),
                 fontsize=8.5)
    return save(fig, out, "fig3_decode_bandwidth.pdf")


# --------------------------------------------------------------------------- #
# fig4: binary provenance
# --------------------------------------------------------------------------- #

NATIVE, OLDER, PTX, ABSENT, MISSING, JIT, JITFAIL = (
    "native SASS", "runs older cubin", "PTX only (JIT at load)",
    "no code for this arch", "library not installed",
    "JIT-compiled at run time", "JIT fails (nvcc < 12.9)")

PROV_STYLE = {
    NATIVE:  (OKABE[2], "SASS", "white"),
    OLDER:   (OKABE[1], "", "black"),
    PTX:     (OKABE[5], "PTX", "black"),
    ABSENT:  ("#e8e8e8", "—", "black"),
    MISSING: ("#bdbdbd", "n/a", "black"),
    JIT:     (OKABE[0], "JIT", "white"),
    JITFAIL: (OKABE[3], "fail", "white"),
}


def parse_provenance(path: str) -> tuple[dict, dict]:
    """cuobjdump output as {library: {"sass": [...], "ptx": [...]}}.

    Needs no GPU: it reads what the shipped wheels contain, which is the only
    way to separate "this kernel is slow here" from "this wheel has no code for
    here". Returns (libs, header) where header carries the version comments.
    """
    libs, head = {}, {}
    for line in open(path, encoding="utf8"):
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.match(r"#\s*(\S+)\s+(\S+)$", line)
            if m:
                head[m.group(1)] = m.group(2)
            head.setdefault("_notes", []).append(line.lstrip("# "))
            continue
        name = line.split()[0]
        if "NOT FOUND" in line:
            libs[name] = None
            continue
        def arch(tag):
            m = re.search(tag + r"\[([^\]]*)\]", line)
            return sorted({int(a.split("_")[1])
                           for a in re.findall(r"sm_\d+", m.group(1) if m else "")})
        libs[name] = {"sass": arch("SASS"), "ptx": arch("PTX")}
    return libs, head


def provenance_matrix(libs: dict, targets: list, cuda: str | None) -> pd.DataFrame:
    """Status of each library on each target architecture.

    "runs older cubin" is CUDA minor-version binary compatibility: a cubin built
    for sm_80 loads on sm_86 and sm_89 because they share the major version.
    That is the mechanism the packaging hypothesis is CONSISTENT WITH; nothing
    here proves it caused any latency, only that no native code was shipped.
    """
    # ponytail: nvcc gained SM 12.x in CUDA 12.9, so a run-time JIT under 12.8
    # cannot target Blackwell consumer parts. One threshold, stated once.
    cu = float(re.match(r"(\d+\.\d+)", str(cuda or "0")).group(1)) if cuda else 0.0
    out = {}
    for lib, info in sorted(libs.items()):
        row = {}
        for t in targets:
            if info is None:
                row["sm_%d" % t] = MISSING
            elif info == JIT:
                row["sm_%d" % t] = (JITFAIL if (t >= 120 and 0 < cu < 12.9) else JIT)
            elif t in info["sass"]:
                row["sm_%d" % t] = NATIVE
            elif any(s // 10 == t // 10 and s < t for s in info["sass"]):
                row["sm_%d" % t] = OLDER
            elif t in info["ptx"] or any(p <= t for p in info["ptx"]):
                row["sm_%d" % t] = PTX
            else:
                row["sm_%d" % t] = ABSENT
        out[lib] = row
    return pd.DataFrame(out).T[["sm_%d" % t for t in targets]]


def _older_label(info, t):
    if not info or info == JIT:
        return ""
    c = [s for s in info["sass"] if s // 10 == t // 10 and s < t]
    return "sm_%d" % max(c) if c else ""


def fig4_provenance(d: dict, prov_path: str, out: str) -> tuple[str, pd.DataFrame]:
    libs, head = parse_provenance(prov_path)
    libs = dict(libs)
    if any("flashinfer" in n.lower() for n in head.get("_notes", [])):
        libs["flashinfer (JIT)"] = JIT

    env = (d["summary"] or {}).get("environment") or {}
    have = sorted({(m.get("cc_major") or 0) * 10 + (m.get("cc_minor") or 0)
                   for m in env.values() if m.get("cc_major")})
    newer = sorted({a for i in libs.values() if isinstance(i, dict)
                    for a in i["sass"] + i["ptx"] if have and a > max(have)})
    targets = sorted(set(have) | set(newer))
    cuda = next((m.get("cuda") for m in env.values() if m.get("cuda")), None)
    mat = provenance_matrix(libs, targets, cuda)

    fig, ax = plt.subplots(figsize=(1.05 * len(targets) + 3.4,
                                    0.45 * len(mat) + 2.3))
    for y, lib in enumerate(mat.index):
        for x, col in enumerate(mat.columns):
            st = mat.loc[lib, col]
            face, txt, fg = PROV_STYLE[st]
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face,
                                   edgecolor="white", linewidth=1.2))
            if st == OLDER:
                txt = _older_label(libs.get(lib), int(col.split("_")[1]))
            if txt:
                ax.text(x + 0.5, y + 0.5, txt, ha="center", va="center",
                        fontsize=6.5, color=fg)
    ax.set_xlim(0, len(mat.columns))
    ax.set_ylim(len(mat), 0)
    ax.set_xticks(np.arange(len(mat.columns)) + 0.5)
    ax.set_xticklabels(mat.columns, fontsize=7)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(len(mat)) + 0.5)
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("target compute capability", fontsize=8)
    ax.xaxis.set_label_position("top")
    ax.legend(handles=[Patch(facecolor=PROV_STYLE[s][0], edgecolor="white", label=s)
                       for s in (NATIVE, OLDER, PTX, ABSENT, JIT, JITFAIL, MISSING)],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              borderaxespad=0)
    ax.set_title("Binary provenance of the shipped wheels (cuobjdump, no GPU "
                 "required)\ntorch %s / CUDA %s / flash-attn %s / flashinfer %s "
                 "-- an orange cell runs another architecture's cubin under\n"
                 "CUDA minor-version binary compatibility; this is a packaging "
                 "fact, consistent with (not proof of) the latency it sits beside"
                 % _versions(d), fontsize=8, pad=22)
    return save(fig, out, "fig4_provenance.pdf"), mat


# --------------------------------------------------------------------------- #
# fig5: inversion scatter
# --------------------------------------------------------------------------- #

def examined_pairs(cells: pd.DataFrame, a: str, b: str) -> dict:
    """Implementation pairs comparable on both devices, per regime.

    analysis.inversions records only the pairs whose ordering flipped, and its
    pairs_examined counter runs across both regimes at once, so it cannot be
    sliced. The denominator of a per-regime inversion rate has to be recounted,
    and it is C(k,2) over the implementations each shared cell ran on both.
    """
    m = cells[cells.gpu_name.isin([a, b])]
    out = {}
    for cell, sub in m.groupby("cell"):
        k = len(set(sub[sub.gpu_name == a].implementation)
                & set(sub[sub.gpu_name == b].implementation))
        reg = cell.split("|")[0]
        out[reg] = out.get(reg, 0) + k * (k - 1) // 2
    return out


INV_CLASS = [("statistical and practical", OKABE[3]),
             ("practical only", OKABE[1]),
             ("statistical only", OKABE[0]),
             ("neither threshold", "#b0b0b0")]


def _inv_class(r):
    if r.sig and r.practical:
        return "statistical and practical"
    if r.practical:
        return "practical only"
    if r.sig:
        return "statistical only"
    return "neither threshold"


def _same_arch_pair(d: dict) -> tuple | None:
    """The control pair: two GPUs of the same compute capability. Its inversion
    rate is the noise floor every cross-architecture number is read against."""
    env = (d["summary"] or {}).get("environment") or {}
    inv = d["inversions"]
    if not len(inv):
        return None
    def cc(g):
        m = env.get(g, {})
        return (m.get("cc_major"), m.get("cc_minor"))
    for a, b in sorted(set(zip(inv.gpu_a, inv.gpu_b))):
        if cc(a) == cc(b) and cc(a)[0] is not None:
            return a, b
    return None


def fig5_inversion_scatter(d: dict, out: str) -> str:
    inv = d["inversions"]
    if not len(inv):
        return print("skipped fig5: inversions.parquet is empty")
    inv = inv.assign(klass=inv.apply(_inv_class, axis=1))
    exam = {(a, b): examined_pairs(d["cells"], a, b)
            for a, b in sorted(set(zip(inv.gpu_a, inv.gpu_b)))}
    ctrl = _same_arch_pair(d)
    is_ctrl = ((inv.gpu_a == ctrl[0]) & (inv.gpu_b == ctrl[1])) if ctrl else None
    cols = [("cross-architecture pairs (%d GPU pairs)"
             % (len(exam) - (1 if ctrl else 0)),
             inv if ctrl is None else inv[~is_ctrl],
             {k: v for k, v in exam.items() if k != ctrl})]
    if ctrl:
        cols.append(("same-architecture control: %s vs %s"
                     % (short_gpu(ctrl[0]), short_gpu(ctrl[1])),
                     inv[is_ctrl], {ctrl: exam[ctrl]}))

    fig, axes = plt.subplots(len(REGIMES), len(cols), squeeze=False,
                             figsize=(3.4 * len(cols) + 2.2, 3.3 * len(REGIMES)))
    for i, reg in enumerate(REGIMES):
        for j, (title, sub, ex) in enumerate(cols):
            ax = axes[i][j]
            s = sub[sub.cell.str.startswith(reg + "|")]
            n_ex = sum(e.get(reg, 0) for e in ex.values())
            lim = (0.2, 5.0)
            if len(s):
                lo = min(s.r1.min(), s.r2.min()) * 0.8
                hi = max(s.r1.max(), s.r2.max()) * 1.25
                lim = (min(lo, 1 / hi), max(hi, 1 / lo))
            # The two inversion quadrants. Everything plotted is already a sign
            # flip -- analysis.inversions only records pairs whose ordering
            # changed -- so the shading names the quadrants rather than
            # selecting anything.
            for (x0, x1, y0, y1) in ((lim[0], 1, 1, lim[1]), (1, lim[1], lim[0], 1)):
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, zorder=0,
                                       facecolor="#0072b214", edgecolor="none"))
            for name, colour in INV_CLASS:
                k = s[s.klass == name]
                if len(k):
                    ax.scatter(k.r1, k.r2, s=7, alpha=0.55, linewidths=0,
                               color=colour, label=name)
            ax.axhline(1, color="black", linewidth=0.7)
            ax.axvline(1, color="black", linewidth=0.7)
            ax.plot(lim, lim, color="#888888", linewidth=0.6, linestyle=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.set_aspect("equal")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("T(A)/T(B) on the pair's first GPU (log)")
            ax.set_ylabel("T(A)/T(B) on the pair's second GPU (log)")
            n_pr = int(s.practical.sum()) if len(s) else 0
            ax.set_title("%s -- %s\n%d sign flips, %d past %.0f%% both ways\n"
                         "(%s of %d comparable pairs)"
                         % (reg, title, len(s), n_pr, 100 * (PRACTICAL - 1),
                            "%.1f%%" % (100 * n_pr / n_ex) if n_ex else "n/a",
                            n_ex), fontsize=7)
            if not len(s):
                ax.text(0.5, 0.62, "no matched pairs\n(this regime was not run "
                        "on both devices)", transform=ax.transAxes, ha="center",
                        va="center", fontsize=7, color="#666666")

    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=c, label=n)
                        for n, c in INV_CLASS],
               loc="outside center right", frameon=False,
               title="separation of the flip")
    fig.suptitle("Matched-cell ordering flips between GPU pairs\n"
                 "each point is one implementation pair (A, B) in one "
                 "configuration; the off-diagonal quadrants (shaded) are the\n"
                 "inversions -- A beats B on one device and loses on the other",
                 fontsize=9)
    return save(fig, out, "fig5_inversion_scatter.pdf")


# --------------------------------------------------------------------------- #
# numbers.json
# --------------------------------------------------------------------------- #

def _r(x, n=4):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    return None if not np.isfinite(f) else round(f, n)


def _flips(cells: pd.DataFrame, regime: str, gpus: list) -> dict:
    """Winner-flip rate over the cells every GPU in `gpus` ran.

    The per-GPU winner counts are what makes a flip rate readable: 82% of
    configurations changing hands says nothing about which kernel lost them.
    """
    d = cells[(cells.regime == regime) & cells.gpu_name.isin(gpus)]
    n = d.groupby("cell").gpu_name.nunique()
    m = d[d.cell.isin(n[n == len(gpus)].index)]
    if not len(m):
        return {}
    w = m.loc[m.groupby(["gpu_name", "cell"]).median_us.idxmin()]
    per = w.groupby("cell").implementation.nunique()
    return {"gpus": [short_gpu(g) for g in gpus],
            "n_gpus": len(gpus),
            "n_matched_cells": int(len(per)),
            "winner_flip_rate": _r((per > 1).mean()),
            "winners_per_gpu": {short_gpu(g): s.implementation.value_counts()
                                                .to_dict()
                                for g, s in w.groupby("gpu_name")}}


def numbers(d: dict, wins: pd.DataFrame, prov: pd.DataFrame) -> dict:
    rows, cells, summary = d["rows"], d["cells"], d["summary"] or {}
    env = summary.get("environment") or {}
    ok = usable(rows)
    out = {}

    out["dataset"] = {
        "n_rows": int(len(rows)),
        "n_cell_medians": int(len(cells)),
        "gpus": [short_gpu(g) for g in gpu_order(d)],
        "status_counts": {k: int(v) for k, v in
                          rows.status.value_counts().sort_index().items()},
        "n_implementations": {r: int(cells[cells.regime == r].implementation.nunique())
                              for r in REGIMES if (cells.regime == r).any()},
        "repeats_per_cell_median": _r(cells.reps.median(), 1) if len(cells) else None,
    }
    torch_v, cuda_v, fa_v, fi_v = _versions(d)
    out["software_snapshot"] = {"torch": torch_v, "cuda": cuda_v,
                                "flash_attn": fa_v, "flashinfer": fi_v,
                                "triton": next((m.get("versions", {}).get("triton")
                                                for m in env.values()
                                                if m.get("versions")), None)}
    # The decode cross-GPU comparison confounds architecture with a commit and
    # driver change. This is the table that says so, and it is computed.
    if "git_sha" in rows:
        out["provenance_of_rows"] = {
            "%s|%s" % (short_gpu(g), reg): {"commit": sha[:8],
                                            "driver": env.get(g, {}).get("driver"),
                                            "rows": int(n)}
            for (g, reg, sha), n in
            rows.groupby(["gpu_name", "regime", rows.git_sha.str[:8]]).size().items()}
        out["confounded_regimes"] = sorted(
            reg for reg in set(rows.regime)
            if rows[rows.regime == reg].git_sha.str[:8].nunique() > 1)

    # Finding 1, over every subset of GPUs. The matched-cell count changes
    # with the subset -- a device that skipped a configuration removes it for
    # everyone -- so the denominator travels with every rate.
    out["winner_flips_all_gpus"] = {
        reg: _flips(cells, reg, sorted(set(sh.gpu_name)))
        for reg in REGIMES for sh in [shared_cells(cells, reg)] if len(sh)}
    out["winner_flips_by_gpu_subset"] = {}
    for reg in REGIMES:
        gs = sorted(set(cells[cells.regime == reg].gpu_name))
        for r in range(2, len(gs) + 1):
            for sub in itertools.combinations(gs, r):
                f = _flips(cells, reg, list(sub))
                if f:
                    out["winner_flips_by_gpu_subset"][
                        "%s|%s" % (reg, " vs ".join(short_gpu(g) for g in sub))] = f

    # Winner counts, and the absolute latency behind them: never a ranking
    # without the microseconds it is a ranking of.
    out["winners_by_gpu"] = {
        "%s|%s" % (short_gpu(g), reg): {
            "counts": s.winner.value_counts().to_dict(),
            "median_winning_latency_us": _r(s.winner_us.median(), 2),
            "min_winning_latency_us": _r(s.winner_us.min(), 2),
            "max_winning_latency_us": _r(s.winner_us.max(), 2)}
        for (g, reg), s in wins.groupby(["gpu_name", "regime"])}

    # The same rate, restricted to cells whose winner is decisively ahead of
    # second place on EVERY device in the subset. A flip between two
    # implementations that were never apart is not a portability failure, and
    # 60.7% of cells have no separated winner at all, so the unrestricted rate
    # counts a lot of coin flips. Reporting both is the honest form: prefill
    # survives the restriction, decode largely does not.
    sepset = {(r.gpu_name, r.cell) for r in wins.itertuples() if r.separated}
    win = {(r.gpu_name, r.cell): r.winner for r in wins.itertuples()}
    out["winner_flips_separated_only"] = {}
    for reg in REGIMES:
        gs = sorted(set(cells[cells.regime == reg].gpu_name))
        for r in range(2, len(gs) + 1):
            for sub in itertools.combinations(gs, r):
                # Same matching rule _flips uses: cells every GPU in the
                # subset actually ran.
                d0 = cells[(cells.regime == reg) & cells.gpu_name.isin(sub)]
                nn = d0.groupby("cell").gpu_name.nunique()
                matched = sorted(nn[nn == len(sub)].index)
                keep = [c for c in matched
                        if all((g, c) in sepset for g in sub)]
                if len(keep) < 5:      # a rate over <5 cells is not a rate
                    continue
                flips = sum(1 for c in keep
                            if len({win.get((g, c)) for g in sub}) > 1)
                out["winner_flips_separated_only"][
                    "%s|%s" % (reg, " vs ".join(short_gpu(g) for g in sub))] = {
                        "n_cells_separated_on_all": len(keep),
                        "flip_rate": _r(flips / len(keep)),
                        "n_cells_unrestricted": len(matched)}

    out["winner_separation"] = {
        "practical_threshold": PRACTICAL,
        "overall_separated_fraction": _r(wins.separated.mean()),
        "by_regime": {reg: {"n": int(len(s)),
                            "separated": _r(s.separated.mean()),
                            "statistical_only": _r((s.sig & ~s.practical).mean()),
                            "median_winner_over_runner_up": _r(s.ratio.median(), 3)}
                      for reg, s in wins.groupby("regime")}}

    # Finding 2.
    sp = d["spread"]
    out["spread_slowest_over_fastest"] = {
        reg: {"median": _r(s.spread.median(), 2), "p90": _r(s.spread.quantile(.9), 2),
              "max": _r(s.spread.max(), 2), "n_cells": int(len(s)),
              "by_gpu": {short_gpu(g): _r(v, 2) for g, v in
                         s.groupby("gpu_name").spread.median().items()}}
        for reg, s in sp.groupby("regime")} if len(sp) else {}

    out["rank_correlation"] = summary.get("rank_correlation", {})

    inv = d["inversions"]
    if len(inv):
        ex = {(a, b): examined_pairs(cells, a, b)
              for a, b in sorted(set(zip(inv.gpu_a, inv.gpu_b)))}
        out["inversions"] = {}
        for (a, b, reg), s in (inv.assign(reg=inv.cell.str.split("|").str[0])
                                  .groupby(["gpu_a", "gpu_b", "reg"])):
            n = ex[(a, b)].get(reg, 0)
            out["inversions"]["%s|%s vs %s" % (reg, short_gpu(a), short_gpu(b))] = {
                "comparable_pairs": int(n),
                "sign_flips": int(len(s)),
                "sign_flip_rate": _r(len(s) / n) if n else None,
                "practical_inversions": int(s.practical.sum()),
                "practical_rate": _r(s.practical.sum() / n) if n else None,
                "statistical_and_practical_rate":
                    _r((s.sig & s.practical).sum() / n) if n else None}

    # Decode is bandwidth-bound; this is what makes the regime contrast mean
    # something. Peak is MEASURED on the host, never a vendor figure.
    dec = ok[(ok.regime == "decode") & ok.eff_bw_gbs.notna()]
    out["decode_bandwidth"] = {"_note": "utilisation above 1 means the KV slice "
                               "fit in cache and was not streamed from HBM; the "
                               "denominator is this host's measured copy rate"}
    out["decode_bandwidth"].update({
        short_gpu(g): {"measured_peak_gbs": _r(peak_bw(d).get(g), 1),
                       "median_effective_gbs": _r(s.eff_bw_gbs.median(), 1),
                       "max_effective_gbs": _r(s.eff_bw_gbs.max(), 1),
                       "median_utilisation": _r(s.bw_util.median(), 3),
                       "max_utilisation": _r(s.bw_util.max(), 3)}
        for g, s in dec.groupby("gpu_name")})

    # Finding 3: torch.compile does not rewrite naive attention into SDPA.
    ind = rows[rows.implementation.str.startswith(("P1-", "D1-"))]
    out["inductor_fusion"] = {
        "n_inductor_rows": int(len(ind)),
        "n_rows_with_counter": int(ind.fuse_attention.notna().sum())
                               if "fuse_attention" in ind else 0,
        "n_fuse_attention_nonzero": int((ind.fuse_attention.fillna(0) > 0).sum())
                                    if "fuse_attention" in ind else 0,
        "by_gpu": {short_gpu(g): int((s.fuse_attention.fillna(0) > 0).sum())
                   for g, s in ind.groupby("gpu_name")}
        if "fuse_attention" in ind else {}}

    # Where an implementation could not run at all, and why. An absent kernel is
    # a portability result, not a missing data point.
    out["coverage"] = {
        "%s|%s" % (short_gpu(g), reg): {
            impl: {k: int(v) for k, v in s.status.value_counts().items()}
            for impl, s in sub.groupby("implementation")
            if (s.status != "OK").any()}
        for (g, reg), sub in rows.groupby(["gpu_name", "regime"])}
    out["never_ran"] = {
        "%s|%s" % (short_gpu(g), reg): sorted(
            set(sub.implementation) - set(sub[sub.status == "OK"].implementation))
        for (g, reg), sub in rows.groupby(["gpu_name", "regime"])}

    # Cost of standardising on one backend, with the latency it costs.
    port = d["portability"]
    if len(port):
        out["portability_cost"] = {
            "%s|%s" % (reg, impl): {
                "median_p_ratio": _r(s.p_ratio.median(), 3),
                "p05_p_ratio": _r(s.p_ratio.quantile(.05), 3),
                "median_latency_us": _r(s.median_us.median(), 2),
                "median_best_in_cell_us": _r(s.best_us.median(), 2),
                "n_cells": int(len(s))}
            for (reg, impl), s in port.groupby(["regime", "implementation"])}

    sel = d["selector"] or {}
    if "accuracy" in sel:
        out["selector"] = {k: _r(sel[k]) for k in
                           ("accuracy", "median_regret", "p95_regret")}
        out["selector"]["n_heldout"] = sel.get("n_heldout")
        out["selector"]["vs_fixed"] = {
            k: {kk: _r(vv) for kk, vv in v.items()}
            for k, v in (sel.get("vs_fixed") or {}).items()}

    out["binary_provenance"] = ({lib: prov.loc[lib].to_dict() for lib in prov.index}
                                if prov is not None and len(prov) else {})
    out["device_manifest"] = {
        short_gpu(g): {"cc": "sm%s%s" % (m.get("cc_major"), m.get("cc_minor")),
                       "sm_count": m.get("sm_count"),
                       "measured_peak_bw_gbs": m.get("measured_peak_bw_gbs"),
                       "driver": m.get("driver"), "cuda": m.get("cuda")}
        for g, m in sorted(env.items())}
    out["not_collected"] = {
        "nsight_compute": "blocked on every platform used; no counter data exists",
        "nsight_systems": "blocked on every platform used; no trace data exists"}
    return out


# --------------------------------------------------------------------------- #

def _selftest():
    """One runnable check on the two pieces of real logic: the provenance
    parser and the minor-version-compat rule."""
    import tempfile
    txt = ("# binary provenance -- test\n# torch 2.9.0+cu128\n"
           "flash-attn-2   951MB  SASS[ sm_80 sm_90 sm_120 ]  PTX[ none ]\n"
           "flash-attn-3  NOT FOUND\n"
           "cublas 110MB SASS[ sm_80 sm_86 ] PTX[ sm_120 ]\n"
           "# flashinfer JITs at run time and ships no cubins\n")
    p = os.path.join(tempfile.mkdtemp(), "provenance.txt")
    open(p, "w", encoding="utf8").write(txt)
    libs, head = parse_provenance(p)
    assert libs["flash-attn-2"]["sass"] == [80, 90, 120], libs
    assert libs["flash-attn-2"]["ptx"] == []
    assert libs["flash-attn-3"] is None
    assert head["torch"] == "2.9.0+cu128"
    libs["flashinfer (JIT)"] = JIT
    m = provenance_matrix(libs, [80, 86, 89, 90, 120], "12.8")
    assert m.loc["flash-attn-2", "sm_80"] == NATIVE
    assert m.loc["flash-attn-2", "sm_86"] == OLDER       # sm_80 cubin, same major
    assert m.loc["flash-attn-2", "sm_89"] == OLDER
    assert m.loc["cublas", "sm_89"] == OLDER
    assert m.loc["cublas", "sm_120"] == PTX
    assert m.loc["flash-attn-3", "sm_90"] == MISSING
    assert m.loc["flashinfer (JIT)", "sm_90"] == JIT
    assert m.loc["flashinfer (JIT)", "sm_120"] == JITFAIL
    assert provenance_matrix(libs, [120], "12.9").loc["flashinfer (JIT)", "sm_120"] == JIT
    print("selftest ok")


def fig6_winner_bands(d: dict, wins: pd.DataFrame, out: str) -> str:
    """The winner map binned to workload bands, sized for one paper column.

    fig1 draws one row per configuration, which is right for the artifact and
    unreadable at 3.03 in: its 5 pt labels land near 2 pt. Binning to a dozen
    bands keeps the pattern the figure exists to show, which is whether the
    colour changes across a row, and makes the labels legible in print.
    """
    w = wins.copy()
    k = cell_key(w.cell)
    w = pd.concat([w, k.drop(columns=k.columns.intersection(w.columns))], axis=1)

    def band(r):
        lo = "N" if r.regime == "prefill" else "KV"
        n = "short" if r.N <= 1024 else ("mid" if r.N <= 4096 else "long")
        b = "B1" if r.B == 1 else ("B4-8" if r.B <= 8 else "B16+")
        return "%s %s %s %s" % (r.regime[:3], b, lo, n)

    w["band"] = w.apply(band, axis=1)
    gpus = [short_gpu(g) for g in gpu_order(d)]
    bands = sorted(w.band.unique(), key=lambda s: (s.split()[0], s))
    impls = sorted(w.winner.unique())
    cmap = {im: COLOURS.get(im, "#8A92A0") for im in impls}

    fig, ax = plt.subplots(figsize=(3.03, 0.26 * len(bands) + 0.9))
    for yi, bd in enumerate(bands):
        for xi, g in enumerate(gpus):
            sub = w[(w.band == bd) & (w.gpu_name.map(short_gpu) == g)]
            if not len(sub):
                continue
            # Modal winner in the band, and whether the band is decisive: a
            # band where most cells have no separated winner is drawn hollow,
            # so a colour change there is not read as a real difference.
            top = sub.winner.value_counts().idxmax()
            share = sub.separated.mean()
            ax.add_patch(Rectangle((xi, yi), 1, 1,
                                       facecolor=cmap[top] if share >= .5 else "none",
                                       edgecolor=cmap[top], linewidth=.8,
                                       hatch=None if share >= .5 else "///"))
    ax.set_xlim(0, len(gpus)); ax.set_ylim(0, len(bands))
    ax.set_xticks([i + .5 for i in range(len(gpus))])
    ax.set_xticklabels(gpus, fontsize=6, rotation=35, ha="right")
    ax.set_yticks([i + .5 for i in range(len(bands))])
    ax.set_yticklabels(bands, fontsize=5.6)
    ax.invert_yaxis()
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    handles = [Patch(facecolor=cmap[i], label=i) for i in impls]
    handles.append(Patch(facecolor="none", edgecolor="#555", hatch="///",
                         label="no separated winner"))
    ax.legend(handles=handles, fontsize=5, loc="upper center",
              bbox_to_anchor=(.5, -.14), ncol=2, frameon=False)
    fig.tight_layout()
    return save(fig, out, "fig6_winner_bands.pdf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("processed", nargs="?", default="results/processed",
                    help="directory analysis.py wrote")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--provenance", default="results_live/provenance.txt")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()

    _style()
    d = load(a.processed)
    if not len(d["cells"]):
        raise SystemExit("no cells.parquet under %r; run akp.analysis first"
                         % a.processed)
    os.makedirs(a.out, exist_ok=True)

    wins = winner_table(d["rows"])
    fig1_winner_map(d, wins, a.out)
    fig6_winner_bands(d, wins, a.out)
    fig2_prefill_latency(d, a.out)
    fig3_decode_bandwidth(d, a.out)
    mat = pd.DataFrame()
    if os.path.exists(a.provenance):
        _, mat = fig4_provenance(d, a.provenance, a.out)
    else:
        print("skipped fig4: no %s" % a.provenance)
    fig5_inversion_scatter(d, a.out)

    n = numbers(d, wins, mat)
    p = os.path.join(a.out, "numbers.json")
    json.dump(n, open(p, "w", encoding="utf8"), indent=2, sort_keys=True,
              default=str)
    print("wrote %s (%.0f kB)" % (p, os.path.getsize(p) / 1e3))
    for reg, v in (n.get("winner_flips_all_gpus") or {}).items():
        print("%s: winner changes on %.1f%% of %d configurations matched across "
              "%d GPUs" % (reg, 100 * v["winner_flip_rate"],
                           v["n_matched_cells"], v["n_gpus"]))
    print("winner separated from #2 on %.1f%% of (GPU, configuration) cells"
          % (100 * wins.separated.mean()))


if __name__ == "__main__":
    main()
