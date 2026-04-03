"""The main result ledger: one frozen output every surface must quote.

The guide's Table 3. Every question the paper, README, site and dashboard ask
is answered here once, with its exact subset, its numerator and denominator,
its interval, and its coverage. Numbers drifted between surfaces in this
project because each recomputed its own; this exists so none of them do.

    python paper/ledger.py                      # results/processed -> ledger.{json,md}

Conventions this file fixes, all of them from the publication review:

  * Forward prefill and forward-plus-backward are different populations. The
    inference question is forward-only; fwd_bwd is reported separately and
    never pooled into a headline.
  * The separation ratio is runner-up latency / fastest latency, and the
    margin and the interval are applied to that same estimator.
  * "Native set" is every backend a device ran. "Common set" is the
    intersection of backends eligible on BOTH devices of a pair, recomputed
    from scratch: winners and separation are re-derived after restricting, not
    filtered afterwards. This is what controls for FA3 being H100-only and
    FlashInfer being absent on the 5090.
  * Transfer cost is the target-device latency of the SOURCE device's chosen
    backend divided by the target's own best. Source choices unavailable on
    the target are reported as a coverage fraction, never dropped and never
    silently replaced by the target's oracle.
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from akp.analysis import PRACTICAL, _boot_ratio, per_cell_median, usable  # noqa: E402

MARGINS = (1.05, 1.10, 1.15)


def short(g):
    return (g.replace("NVIDIA ", "").replace("-SXM4-80GB", "")
             .replace(" 80GB HBM3", "").replace("GeForce ", ""))


def regime_of(cell):
    return cell.split("|")[0]


def mode_of(cell):
    # prefill|B|Hq|Hkv|D|N|dtype|mode|launch|cache|causal
    p = cell.split("|")
    return p[7] if len(p) > 7 else "fwd"


def build_index(cells, margin=PRACTICAL):
    """(gpu, cell) -> per-cell record over whatever backends are present."""
    rec = {}
    ids = "repeat_ids" in cells.columns
    for (g, c), sub in cells.groupby(["gpu_name", "cell"]):
        s = sub.sort_values(["median_us", "implementation"])
        lat = dict(zip(s.implementation, s.median_us))
        if len(s) < 2:
            rec[(g, c)] = dict(winner=s.iloc[0].implementation, sep=False,
                               ratio=float("nan"), lat=lat, n=len(s))
            continue
        w, ru = s.iloc[0], s.iloc[1]
        ratio = ru.median_us / w.median_us
        kw = dict(ids_a=ru.repeat_ids, ids_b=w.repeat_ids) if ids else {}
        lo, hi = _boot_ratio(ru.samples, w.samples, **kw)
        rec[(g, c)] = dict(winner=w.implementation,
                           sep=bool((lo > 1 or hi < 1) and ratio >= margin),
                           ratio=float(ratio), lat=lat, n=len(s))
    return rec


def restrict(cells, keep):
    """Re-derive per-cell records over a restricted backend set."""
    return build_index(cells[cells.implementation.isin(keep)])


def boot_rate(pairs, n=2000, seed=0):
    """Cluster bootstrap over CELLS: one cell feeds several device pairs."""
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    by = {}
    for c, f in pairs:
        by.setdefault(c, []).append(f)
    keys = list(by)
    rate = 100 * float(np.mean([f for _, f in pairs]))
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        draws.append(100 * np.mean([f for i in pick for f in by[keys[i]]]))
    return rate, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def sep_rate(rec, pred):
    hits = [(c, r["sep"]) for (g, c), r in rec.items() if pred(g, c) and r["n"] >= 2]
    if not hits:
        return dict(n=0, k=0, pct=float("nan"), lo=float("nan"), hi=float("nan"))
    rate, lo, hi = boot_rate(hits)
    return dict(n=len(hits), k=int(sum(f for _, f in hits)),
                pct=rate, lo=lo, hi=hi)


def flips(rec, gpus, pred, both_separated=True):
    """(cell, flipped) per device-pair comparison."""
    out, excluded = [], 0
    for i, a in enumerate(gpus):
        for b in gpus[i + 1:]:
            for (g, c), r in rec.items():
                if g != a or not pred(a, b, c) or (b, c) not in rec:
                    continue
                r2 = rec[(b, c)]
                if both_separated and not (r["sep"] and r2["sep"]):
                    excluded += 1
                    continue
                out.append((c, r["winner"] != r2["winner"]))
    return out, excluded


def transfer_cost(rec, gpus, pred, require_source_sep=True):
    """Target latency of the source's pick / target's own best.

    Unavailable source picks are counted, never dropped and never replaced by
    the target's oracle, because doing either turns a coverage problem into a
    flattering ratio.
    """
    costs, unavailable, total = [], 0, 0
    for a in gpus:
        for b in gpus:
            if a == b:
                continue
            for (g, c), r in rec.items():
                if g != a or not pred(a, b, c) or (b, c) not in rec:
                    continue
                if require_source_sep and not r["sep"]:
                    continue
                tgt = rec[(b, c)]
                total += 1
                if r["winner"] not in tgt["lat"]:
                    unavailable += 1
                    continue
                best = min(tgt["lat"].values())
                costs.append(tgt["lat"][r["winner"]] / best)
    if not costs:
        return dict(n=0, unavailable=unavailable, total=total,
                    median=float("nan"), p95=float("nan"), frac_within_10=float("nan"))
    a = np.array(costs)
    return dict(n=len(a), unavailable=unavailable, total=total,
                unavailable_pct=100.0 * unavailable / max(total, 1),
                median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                frac_within_10=float((a <= 1.10).mean()))


def main():
    proc = os.path.join(ROOT, "results", "processed")
    rows = pd.read_parquet(os.path.join(proc, "rows.parquet"))
    u = usable(rows)
    cells = per_cell_median(u)
    gpus = sorted(set(cells.gpu_name))
    BLK = [g for g in gpus if "5090" in g]
    NONBLK = [g for g in gpus if g not in BLK]

    rec = build_index(cells)

    # Backends eligible per (device, regime): what a common-set restriction
    # has to intersect over.
    elig = {}
    for (g, c), r in rec.items():
        elig.setdefault((g, regime_of(c)), set()).update(r["lat"])

    P_FWD = lambda g, c: regime_of(c) == "prefill" and mode_of(c) == "fwd"
    P_BWD = lambda g, c: regime_of(c) == "prefill" and mode_of(c) == "fwd_bwd"
    DEC = lambda g, c: regime_of(c) == "decode"

    L = {"generated_from": "results/processed/rows.parquet",
         "practical_margin": PRACTICAL, "gpus": [short(g) for g in gpus],
         "questions": {}}
    Q = L["questions"]

    # ---------------------------------------------------------- separation
    Q["separation_prefill_fwd"] = dict(
        question="Fraction of forward-prefill configurations with a separated fastest backend",
        subset="regime=prefill, mode=fwd, >=2 eligible backends, all devices pooled",
        **sep_rate(rec, P_FWD))
    Q["separation_decode"] = dict(
        question="Fraction of decode configurations with a separated fastest backend",
        subset="regime=decode, >=2 eligible backends, all devices pooled",
        **sep_rate(rec, DEC))
    Q["separation_prefill_fwd_bwd"] = dict(
        question="Same, forward-plus-backward prefill (appendix population, not an inference result)",
        subset="regime=prefill, mode=fwd_bwd",
        **sep_rate(rec, P_BWD))

    # per device, for Figure 1
    Q["separation_by_device"] = {}
    for g in gpus:
        Q["separation_by_device"][short(g)] = dict(
            prefill_fwd=sep_rate(rec, lambda gg, c, g=g: gg == g and P_FWD(gg, c)),
            decode=sep_rate(rec, lambda gg, c, g=g: gg == g and DEC(gg, c)))

    # margin sensitivity
    Q["separation_margin_sensitivity"] = {}
    for m in MARGINS:
        r_m = build_index(cells, margin=m)
        Q["separation_margin_sensitivity"]["%.2f" % m] = dict(
            prefill_fwd=sep_rate(r_m, P_FWD), decode=sep_rate(r_m, DEC))

    # excluding L40 (3 designed repeats, realised median 2)
    noL40 = cells[cells.gpu_name != "NVIDIA L40"]
    r_noL40 = build_index(noL40)
    Q["separation_excluding_L40"] = dict(
        prefill_fwd=sep_rate(r_noL40, P_FWD), decode=sep_rate(r_noL40, DEC))

    # ------------------------------------------------------------- flips
    def flip_block(pred_pair, gset, label, sub):
        native, excl = flips(rec, gset, pred_pair)
        rate, lo, hi = boot_rate(native)
        allc, _ = flips(rec, gset, pred_pair, both_separated=False)
        arate, _, _ = boot_rate(allc)
        return dict(question=label, subset=sub,
                    n_separated_both=len(native),
                    k_flipped=int(sum(f for _, f in native)),
                    pct=rate, lo=lo, hi=hi,
                    excluded_not_separated=excl,
                    n_all_shared=len(allc), pct_all_shared=arate)

    Q["flip_prefill_fwd_nonblackwell"] = flip_block(
        lambda a, b, c: P_FWD(a, c), NONBLK,
        "Winner-flip rate, forward prefill, pairs among Ampere/Ada/Hopper",
        "native backend set; both devices separated")
    Q["flip_decode_nonblackwell"] = flip_block(
        lambda a, b, c: DEC(a, c), NONBLK,
        "Winner-flip rate, decode, pairs among Ampere/Ada/Hopper",
        "native backend set; both devices separated")

    # common-set: recompute winners over the intersection, per pair and regime
    def common_flip(pred, gset, label):
        out, excl = [], 0
        for i, a in enumerate(gset):
            for b in gset[i + 1:]:
                reg = "prefill" if "prefill" in label else "decode"
                keep = elig.get((a, reg), set()) & elig.get((b, reg), set())
                if len(keep) < 2:
                    continue
                rr = restrict(cells, keep)
                for (g, c), r in rr.items():
                    if g != a or not pred(a, c) or (b, c) not in rr:
                        continue
                    r2 = rr[(b, c)]
                    if not (r["sep"] and r2["sep"]):
                        excl += 1
                        continue
                    out.append((c, r["winner"] != r2["winner"]))
        rate, lo, hi = boot_rate(out)
        return dict(question=label, subset="common backend set, re-derived",
                    n_separated_both=len(out),
                    k_flipped=int(sum(f for _, f in out)),
                    pct=rate, lo=lo, hi=hi, excluded_not_separated=excl)

    Q["flip_prefill_fwd_common"] = common_flip(
        P_FWD, NONBLK, "Winner-flip rate, forward prefill, common backend set")
    Q["flip_decode_common"] = common_flip(
        DEC, NONBLK, "Winner-flip rate, decode, common backend set")

    if BLK:
        Q["flip_prefill_fwd_blackwell"] = flip_block(
            lambda a, b, c: P_FWD(a, c) and ((a in BLK) != (b in BLK)),
            sorted(gpus),
            "Winner-flip rate, forward prefill, RTX 5090 vs each other device",
            "native backend set; both devices separated")
        Q["flip_decode_blackwell"] = flip_block(
            lambda a, b, c: DEC(a, c) and ((a in BLK) != (b in BLK)),
            sorted(gpus), "Winner-flip rate, decode, RTX 5090 vs each other device",
            "native backend set; both devices separated")

    # --------------------------------------------------------- transfer cost
    Q["transfer_cost_prefill_fwd"] = dict(
        question="Cost of transferring the source device's forward-prefill choice",
        subset="separated source winner; native set; all ordered device pairs",
        **transfer_cost(rec, gpus, lambda a, b, c: P_FWD(a, c)))
    Q["transfer_cost_decode"] = dict(
        question="Cost of transferring the source device's decode choice",
        subset="separated source winner; native set; all ordered device pairs",
        **transfer_cost(rec, gpus, lambda a, b, c: DEC(a, c)))

    # ------------------------------------------------------ availability
    fi = rows[(rows.implementation == "D4-flashinfer")]
    for g in BLK:
        sub = fi[fi.gpu_name == g]
        Q["flashinfer_on_blackwell"] = dict(
            question="FlashInfer decode outcomes on the RTX 5090",
            subset="implementation=D4-flashinfer, gpu=RTX 5090, all attempts",
            attempted=int(len(sub)),
            **{k: int(v) for k, v in sub.status.value_counts().items()})

    out_json = os.path.join(proc, "ledger.json")
    json.dump(L, open(out_json, "w", encoding="utf8"), indent=2, default=float)

    # markdown mirror
    def pct(d, k="pct"):
        v = d.get(k)
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else "%.1f%%" % v

    m = ["# Main result ledger", "",
         "Generated by `paper/ledger.py` from `results/processed/rows.parquet`.",
         "Every number quoted in the paper, README, article and dashboard comes from here.",
         "", "| question | subset | n / N | estimate [95% CI] | coverage |",
         "|---|---|---|---|---|"]
    for key in ("separation_prefill_fwd", "separation_decode",
                "separation_prefill_fwd_bwd"):
        d = Q[key]
        m.append("| %s | %s | %d / %d | %s [%.0f, %.0f] | all devices |"
                 % (d["question"], d["subset"], d["k"], d["n"], pct(d), d["lo"], d["hi"]))
    for key in ("flip_prefill_fwd_nonblackwell", "flip_decode_nonblackwell",
                "flip_prefill_fwd_common", "flip_decode_common",
                "flip_prefill_fwd_blackwell", "flip_decode_blackwell"):
        if key not in Q:
            continue
        d = Q[key]
        m.append("| %s | %s | %d / %d | %s [%.0f, %.0f] | %d comparisons excluded as not separated |"
                 % (d["question"], d["subset"], d["k_flipped"], d["n_separated_both"],
                    pct(d), d["lo"], d["hi"], d.get("excluded_not_separated", 0)))
    for key in ("transfer_cost_prefill_fwd", "transfer_cost_decode"):
        d = Q[key]
        m.append("| %s | %s | %d costed / %d source choices | median %.3fx, p95 %.3fx | %.1f%% unavailable on target |"
                 % (d["question"], d["subset"], d["n"], d["total"],
                    d["median"], d["p95"], d.get("unavailable_pct", 0.0)))
    if "flashinfer_on_blackwell" in Q:
        d = Q["flashinfer_on_blackwell"]
        m.append("| %s | %s | %d attempts | ERROR %d, OOM %d | no usable timing |"
                 % (d["question"], d["subset"], d["attempted"],
                    d.get("ERROR", 0), d.get("OOM", 0)))

    m += ["", "## Margin sensitivity (separation rate)", "",
          "| margin | forward prefill | decode |", "|---|---|---|"]
    for k, v in Q["separation_margin_sensitivity"].items():
        m.append("| %sx | %s (%d/%d) | %s (%d/%d) |"
                 % (k, pct(v["prefill_fwd"]), v["prefill_fwd"]["k"], v["prefill_fwd"]["n"],
                    pct(v["decode"]), v["decode"]["k"], v["decode"]["n"]))
    d = Q["separation_excluding_L40"]
    m += ["", "Excluding L40 (3 designed repeats, realised median 2): forward prefill %s (%d/%d), decode %s (%d/%d)."
          % (pct(d["prefill_fwd"]), d["prefill_fwd"]["k"], d["prefill_fwd"]["n"],
             pct(d["decode"]), d["decode"]["k"], d["decode"]["n"]), ""]

    out_md = os.path.join(ROOT, "paper", "ledger.md")
    io.open(out_md, "w", encoding="utf8", newline="\n").write("\n".join(m) + "\n")
    print("wrote %s and %s" % (out_json, out_md))
    print("\n".join(m[4:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
