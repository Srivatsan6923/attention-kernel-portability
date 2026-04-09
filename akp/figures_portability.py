"""The two portability figures the publication guide specifies.

Figure A: how often there is a separated winner, per device, per regime, with
the margin sensitivity beside it. This is the study's largest-n result and it
existed only as a number in prose.

Figure B: whether the selected backend transfers, in two panels. Panel A is the
device-pair winner-flip rate, native set against common set. Panel B is the
distribution of what transferring actually costs, which is the quantity a flip
count cannot express: a changed winner whose replacement is 0.2% slower is not
a portability problem.

Both read results/processed/ledger.json so they cannot disagree with the
paper. Nothing here recomputes a rate.

    python -m akp.figures_portability
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from akp.figures import _style, save  # noqa: E402
from akp.analysis import PRACTICAL, per_cell_median, usable  # noqa: E402
from paper.ledger import build_index, mode_of, regime_of, short  # noqa: E402

COL_PRE = "#3b6ea5"
COL_DEC = "#b5651d"
COL_GREY = "#9a9a9a"


def fig_separation(L, out, name="figA_separation.pdf"):
    """Per device, the fraction of configurations with a separated winner."""
    by = L["questions"]["separation_by_device"]
    # One order for every figure and table in the paper and on the site:
    # ascending compute capability. Sorting by a measured rate makes the same
    # device sit in a different column in each figure.
    CC = {"A100": 8.0, "A10": 8.6, "L40": 8.9, "L40S": 8.9,
          "H100": 9.0, "RTX 5090": 12.0}
    devs = sorted(by, key=lambda g: (CC.get(g, 99), g))
    x = np.arange(len(devs))
    w = 0.38

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [3, 1.15]})

    for off, key, col, lab in ((-w / 2, "prefill_fwd", COL_PRE, "forward prefill"),
                               (w / 2, "decode", COL_DEC, "decode")):
        vals = [by[g][key]["pct"] for g in devs]
        lo = [by[g][key]["pct"] - by[g][key]["lo"] for g in devs]
        hi = [by[g][key]["hi"] - by[g][key]["pct"] for g in devs]
        ax.bar(x + off, vals, w, color=col, label=lab, zorder=3)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none",
                    ecolor="#333", elinewidth=0.7, capsize=2, zorder=4)
        for xi, g in zip(x + off, devs):
            d = by[g][key]
            ax.text(xi, 2, "%d/%d" % (d["k"], d["n"]), rotation=90, ha="center",
                    va="bottom", fontsize=5.5, color="white", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(devs, rotation=20, ha="right")
    ax.set_ylabel("configurations with a separated\nfastest backend (%)")
    ax.set_ylim(0, 100)
    ax.axhline(50, color=COL_GREY, lw=0.6, ls=":", zorder=2)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("A  Separated fastest backends by GPU", loc="left")

    ms = L["questions"]["separation_margin_sensitivity"]
    margins = sorted(ms)
    ax2.plot([float(m) for m in margins], [ms[m]["prefill_fwd"]["pct"] for m in margins],
             "o-", color=COL_PRE, ms=3, lw=1.1, label="forward prefill")
    ax2.plot([float(m) for m in margins], [ms[m]["decode"]["pct"] for m in margins],
             "s-", color=COL_DEC, ms=3, lw=1.1, label="decode")
    ax2.set_xticks([float(m) for m in margins])
    ax2.set_xticklabels(["%.2f" % float(m) for m in margins])
    ax2.set_xlabel("practical margin")
    ax2.set_ylabel("separated (%)")
    ax2.set_ylim(0, 60)
    ax2.set_title("B  Margin sensitivity", loc="left")
    return save(fig, out, name)


def transfer_samples(rows):
    """Per-comparison transfer cost, for the distribution panel.

    Recomputed here only because the ledger stores summary statistics; the
    definition is imported, not restated.
    """
    cells = per_cell_median(usable(rows))
    rec = build_index(cells)
    gpus = sorted({g for g, _ in rec})
    out = {"prefill": [], "decode": []}
    unavail = {"prefill": [0, 0], "decode": [0, 0]}
    for a in gpus:
        for b in gpus:
            if a == b:
                continue
            for (g, c), r in rec.items():
                if g != a or (b, c) not in rec:
                    continue
                reg = regime_of(c)
                if reg == "prefill" and mode_of(c) != "fwd":
                    continue
                if not r["sep"]:
                    continue
                tgt = rec[(b, c)]
                unavail[reg][1] += 1
                if r["winner"] not in tgt["lat"]:
                    unavail[reg][0] += 1
                    continue
                out[reg].append(tgt["lat"][r["winner"]] / min(tgt["lat"].values()))
    return out, unavail


def fig_transfer(L, rows, out, name="figB_transfer.pdf"):
    q = L["questions"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                  gridspec_kw={"width_ratios": [1.15, 1.35]})

    # ---- Panel A: flip rate, native vs common backend set
    groups = [("forward\nprefill", "flip_prefill_fwd_nonblackwell", "flip_prefill_fwd_common"),
              ("decode", "flip_decode_nonblackwell", "flip_decode_common")]
    x = np.arange(len(groups))
    w = 0.36
    for off, idx, col, lab in ((-w / 2, 1, "#4c78a8", "native set"),
                               (w / 2, 2, "#a8734c", "common set")):
        vals = [q[g[idx]]["pct"] for g in groups]
        lo = [q[g[idx]]["pct"] - q[g[idx]]["lo"] for g in groups]
        hi = [q[g[idx]]["hi"] - q[g[idx]]["pct"] for g in groups]
        ax.bar(x + off, vals, w, color=col, label=lab, zorder=3)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor="#333",
                    elinewidth=0.7, capsize=2, zorder=4)
        for xi, g in zip(x + off, groups):
            d = q[g[idx]]
            ax.text(xi, 1.5, "%d/%d" % (d["k_flipped"], d["n_separated_both"]),
                    rotation=90, ha="center", va="bottom", fontsize=5.5,
                    color="white", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("matched workload comparisons\nwith a different winner (%)")
    ax.set_ylim(0, 70)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("A  Winner flips, Ampere/Ada/Hopper", loc="left")

    # ---- Panel B: what the flip actually costs
    samples, unavail = transfer_samples(rows)
    for key, col, lab in (("prefill", COL_PRE, "forward prefill"),
                          ("decode", COL_DEC, "decode")):
        v = np.sort(np.asarray(samples[key], float))
        if not len(v):
            continue
        ax2.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                 color=col, lw=1.3, label="%s (n=%d)" % (lab, len(v)))
    ax2.axvline(1.0, color=COL_GREY, lw=0.7, ls="-")
    ax2.axvline(1.10, color="#c0392b", lw=0.9, ls="--")
    ax2.text(1.105, 0.06, "1.10", fontsize=6, color="#c0392b")
    ax2.set_xscale("log")
    ax2.set_xlim(0.98, 4)
    ax2.set_xticks([1, 1.1, 1.5, 2, 3])
    ax2.set_xticklabels(["1.0", "1.1", "1.5", "2", "3"])
    # A log axis relabels its own minor ticks ("4 x 10^0") over the top of the
    # explicit ones; the ratio is small and unitless, so suppress them.
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel("target latency of the source's choice / target best")
    ax2.set_ylabel("cumulative fraction of costed transfers")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False, loc="lower right", fontsize=6)
    ax2.set_title("B  Cost of transferring the choice", loc="left")
    # Coverage belongs in the caption: inside the axes it overlapped the
    # curves and shrank past legibility at column width.
    print("  caption coverage: " + "; ".join(
        "%s %d of %d source choices had no eligible target timing"
        % (k, u[0], u[1]) for k, u in sorted(unavail.items())))
    return save(fig, out, name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("processed", nargs="?", default="results/processed")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args(argv)

    L = json.load(open(os.path.join(a.processed, "ledger.json"), encoding="utf8"))
    rows = pd.read_parquet(os.path.join(a.processed, "rows.parquet"))
    os.makedirs(a.out, exist_ok=True)
    _style()
    fig_separation(L, a.out)
    fig_transfer(L, rows, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
