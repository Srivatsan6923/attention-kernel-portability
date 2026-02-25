"""Load raw shards, derive metrics, and report portability, dispatch, and
selector results.

Derived quantities are computed here rather than stored in raw rows, so fixing a
formula is a re-analysis rather than a re-run.

    python -m akp.analysis results/raw
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

FIT_GPUS = ("A100", "H100")          # substring match; the rest are held out
PRACTICAL = 1.10                     # ratio that counts as a practical inversion

# nvidia-smi clocks_throttle_reasons bits. HwSlowdown, SyncBoost, the two
# thermal slowdowns and HwPowerBrake are erratic; the low three are not.
ERRATIC_THROTTLE = 0xF8


def throttle_bits(df: pd.DataFrame) -> pd.Series:
    return df.tel_throttle.fillna("0x0").map(lambda s: int(str(s), 16))


# --------------------------------------------------------------------------- #
# Load and derive
# --------------------------------------------------------------------------- #

def load(raw="results/raw", dispatch=None) -> pd.DataFrame:
    dispatch = dispatch or os.path.join(os.path.dirname(raw.rstrip("/\\")),
                                       "dispatch.jsonl")
    rows = [json.loads(l)
            for f in glob.glob(os.path.join(raw, "*", "*.jsonl"))
            for l in open(f, encoding="utf8")]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no rows under {raw!r}; expected {raw}/<gpu>/*.jsonl")
    if os.path.exists(dispatch):
        d = pd.DataFrame([json.loads(l) for l in open(dispatch, encoding="utf8")])
        # Pods append concurrently, so the same dispatch_id can land more than
        # once. Without the dedup the merge is many-to-many and silently
        # duplicates result rows, which would tighten every bootstrap CI.
        d = d.drop_duplicates("dispatch_id")
        df = df.merge(d[["dispatch_id", "kernels"]], on="dispatch_id",
                      how="left", validate="m:1")
    envdir = os.path.join(os.path.dirname(raw.rstrip("/\\")), "environment")
    return derive(df, environments(envdir))


def derive(df: pd.DataFrame, env: dict | None = None) -> pd.DataFrame:
    itemsize = df["dtype"].map({"fp16": 2, "bf16": 2}).fillna(2)
    pre = df["regime"] == "prefill"

    flops = np.where(
        pre,
        4 * df.batch * df.hq * df.seq_len ** 2 * df.head_dim
        * np.where(df.causal, 0.5, 1.0) * np.where(df["mode"] == "fwd_bwd", 3.5, 1.0),
        4 * df.batch * df.hq * df.seq_len * df.head_dim)
    df["tflops"] = flops / (df.median_us * 1e-6) / 1e12

    # GQA-expanding impls really move Hq heads worth of KV; charging them the
    # algorithmic minimum would flatter them.
    heads = np.where(df.get("gqa_mode", "native") == "expanded", df.hq, df.hkv)
    df["kv_bytes"] = 2 * df.batch * heads * df.seq_len * df.head_dim * itemsize
    df["eff_bw_gbs"] = np.where(pre, np.nan, df.kv_bytes / (df.median_us * 1e-6) / 1e9)
    df["us_per_token"] = np.where(pre, np.nan, df.median_us)
    df["tokens_per_s"] = np.where(pre, np.nan, df.batch / (df.median_us * 1e-6))

    # Fraction of the device's achievable bandwidth. Decode is bandwidth-bound,
    # so this is what makes the prefill/decode contrast mean something rather
    # than just being two different latencies.
    def _peak(g):
        m = (env or {}).get(g, {})
        return m.get("measured_peak_bw_gbs") or m.get("theoretical_bw_gbs")

    peak = df.gpu_name.map(_peak)
    df["bw_util"] = df.eff_bw_gbs / peak

    # Kept rather than excluded (see usable), so the rate is reportable in
    # threats to validity instead of silently shaping the dataset.
    if "tel_throttle" in df:
        df["power_capped"] = (throttle_bits(df) & 0x4) != 0

    # Cell identity: the workload shape, minus which impl ran.
    df["cell"] = (df.regime + "|B" + df.batch.astype(str) + "|Hq" + df.hq.astype(str)
                  + "|Hkv" + df.hkv.astype(str) + "|D" + df.head_dim.astype(str)
                  + "|N" + df.seq_len.astype(str) + "|" + df.dtype + "|" + df["mode"]
                  + "|" + df.launch + "|" + df.get("cache", "warm")
                  + "|c" + df.causal.astype(int).astype(str))
    return df


def spread(cells: pd.DataFrame) -> pd.DataFrame:
    """Slowest/fastest ratio per configuration.

    Finding 2 is about how far apart the implementations are, and whether that
    gap closes in decode; the winner alone cannot say that.
    """
    g = cells.groupby(["gpu_name", "cell"]).median_us
    out = (g.max() / g.min()).rename("spread").reset_index()
    out["regime"] = out.cell.str.split("|").str[0]
    return out


def observed_backend(kernels: str) -> str:
    """Name the kernel family that actually ran.

    `matched` only says the launch was consistent with what was requested. For
    the implementations whose whole point is which backend got picked -- SDPA
    at q_len=1 above all -- the useful answer is the name.
    """
    import re
    for label, pat in (("flashinfer", r"flashinfer"),
                       ("pytorch-flash", r"pytorch_flash"),
                       ("flash-attn", r"flash::"),
                       ("cudnn", r"cudnn"),
                       ("cutlass-fmha", r"fmha_cutlass|cutlassF|fmha"),
                       ("triton", r"^_attn|triton_|_attn_fwd|_attn_bwd"),
                       ("unfused-gemm", r"gemm|softmax")):
        if re.search(pat, kernels or "", re.I):
            return label
    return "unknown"


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows allowed into a speed ranking: measured, correct and not throttled."""
    ok = (df.status == "OK") & df.median_us.notna()
    if "tel_throttle" in df:
        # GpuIdle (0x1), ApplicationsClocks (0x2) and SwPowerCap (0x4) are the
        # steady state of a loaded datacenter GPU, not faults. Excluding
        # SwPowerCap costs 21% of A100 decode rows, and since it fires with
        # workload size (2.8% at N=512, 45% at N=16384) and unevenly across
        # implementations, it would delete the large-config end of the grid and
        # bias against whichever kernels drive the GPU hardest. It is worth
        # ~1.6% of SM clock and is common-mode across the interleaved
        # implementations at a config. Hardware and thermal slowdowns are
        # erratic rather than steady, so those still disqualify a row.
        ok &= (throttle_bits(df) & ERRATIC_THROTTLE) == 0
    if "correctness_pass" in df:
        # The gate runs once per equivalence class, so most rows carry NaN.
        # Disqualify the whole class its representative failed on -- otherwise a
        # kernel that fails the gate loses one row and every other cell of that
        # class still counts as correct.
        ok &= df.correctness_pass.fillna(True)
        if "gate_class" in df:
            failed = set(df.loc[df.correctness_pass == False, "gate_class"].dropna())
            ok &= ~df.gate_class.isin(failed)
    return df[ok]


# --------------------------------------------------------------------------- #
# Portability: rankings, inversions, rank correlation
# --------------------------------------------------------------------------- #

def per_cell_median(df: pd.DataFrame) -> pd.DataFrame:
    """Median per (gpu, cell, impl), keeping repeats for the bootstrap."""
    g = df.groupby(["gpu_name", "cell", "implementation"])
    return g.agg(median_us=("median_us", "median"),
                 reps=("median_us", "size"),
                 samples=("median_us", list)).reset_index()


def winners(cells: pd.DataFrame) -> pd.DataFrame:
    i = cells.groupby(["gpu_name", "cell"]).median_us.idxmin()
    return cells.loc[i, ["gpu_name", "cell", "implementation", "median_us"]]


def winner_flips(cells: pd.DataFrame) -> dict:
    """How often the fastest implementation changes across GPUs.

    This is the number finding 1 is about. A pooled count of which
    implementation wins most often does not answer it.
    """
    w = winners(cells)
    per = w.groupby("cell").implementation.nunique()
    shared = w.groupby("cell").gpu_name.nunique()
    common = per[shared == shared.max()]
    if common.empty:
        return {"n_cells": 0}
    return {"n_cells": int(len(common)),
            "n_gpus": int(shared.max()),
            "flip_rate": float((common > 1).mean()),
            "top1_stable": float((common == 1).mean())}


def _boot_ratio(a, b, n=2000, seed=0):
    """Cluster bootstrap over process repeats: reps inside one process
    share clocks, allocator state and thermal point, so they are not
    independent draws."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = rng.choice(a, (n, len(a)), replace=True).mean(1)
    rb = rng.choice(b, (n, len(b)), replace=True).mean(1)
    r = ra / rb
    return np.percentile(r, 2.5), np.percentile(r, 97.5)


def inversions(cells: pd.DataFrame, g1: str, g2: str, fdr=0.05) -> pd.DataFrame:
    """Pairs whose ordering flips between two GPUs.

    Statistical: signs differ and both bootstrap CIs exclude 1. Practical: both
    ratios past 10%. We lead with the practical count, because at this family
    size a nominal alpha manufactures inversions out of noise.
    """
    out, examined = [], 0
    for cell in sorted(set(cells[cells.gpu_name.str.contains(g1)].cell)
                       & set(cells[cells.gpu_name.str.contains(g2)].cell)):
        sub = cells[cells.cell == cell]
        s1 = sub[sub.gpu_name.str.contains(g1)].set_index("implementation")
        s2 = sub[sub.gpu_name.str.contains(g2)].set_index("implementation")
        common = sorted(set(s1.index) & set(s2.index))
        for i, a in enumerate(common):
            for b in common[i + 1:]:
                examined += 1
                r1 = s1.median_us[a] / s1.median_us[b]
                r2 = s2.median_us[a] / s2.median_us[b]
                if np.sign(r1 - 1) == np.sign(r2 - 1):
                    continue
                lo1, hi1 = _boot_ratio(s1.samples[a], s1.samples[b])
                lo2, hi2 = _boot_ratio(s2.samples[a], s2.samples[b])
                out.append(dict(
                    cell=cell, a=a, b=b, r1=r1, r2=r2, pairs_examined=examined,
                    sig=(lo1 > 1 or hi1 < 1) and (lo2 > 1 or hi2 < 1),
                    practical=(max(r1, 1 / r1) >= PRACTICAL
                               and max(r2, 1 / r2) >= PRACTICAL)))
    return pd.DataFrame(out)


def rank_correlation(cells: pd.DataFrame, g1: str, g2: str) -> dict:
    from scipy.stats import kendalltau, spearmanr
    rs, ks = [], []
    for cell in set(cells.cell):
        sub = cells[cells.cell == cell]
        s1 = sub[sub.gpu_name.str.contains(g1)].set_index("implementation").median_us
        s2 = sub[sub.gpu_name.str.contains(g2)].set_index("implementation").median_us
        common = sorted(set(s1.index) & set(s2.index))
        if len(common) < 3:
            continue
        rs.append(spearmanr(s1[common], s2[common]).statistic)
        ks.append(kendalltau(s1[common], s2[common]).statistic)
    return {"spearman_median": float(np.nanmedian(rs)) if rs else float("nan"),
            "kendall_median": float(np.nanmedian(ks)) if ks else float("nan"),
            "n_cells": len(rs)}


# --------------------------------------------------------------------------- #
# Dispatch: did the requested backend actually run
# --------------------------------------------------------------------------- #

def dispatch_audit(df: pd.DataFrame, impls) -> pd.DataFrame:
    """Did the kernel that ran match the impl that was requested?"""
    import re
    rows = []
    for _, r in df[df.status == "OK"].iterrows():
        pats = impls[r.implementation].kernel_patterns
        k = r.get("kernels") or ""
        # An unobserved probe is not a mismatch. Counting "we could not see the
        # kernel" as "the wrong kernel ran" would inflate the headline number in
        # the direction that flatters the hypothesis.
        probed = bool(k) and not k.startswith("<")
        rows.append(dict(
            implementation=r.implementation, gpu_name=r.gpu_name, cell=r.cell,
            probed=probed,
            matched=(any(re.search(p, k, re.I) for p in pats) if probed
                     else float("nan")),
            # The fuse_attention counter misses when an identical graph was
            # already compiled in the process, so take the launched kernels as
            # the evidence and let the counter corroborate.
            fused_into_sdpa=bool(r.get("fuse_attention", 0)) or bool(
                r.implementation.startswith(("P1-", "D1-")) and probed
                and re.search(r"flash|fmha|cutlass|cudnn", k, re.I)),
            observed=observed_backend(k) if probed else "unprobed",
            n_kernels=r.get("n_kernels", 0)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Hardware-aware backend selector
# --------------------------------------------------------------------------- #

FEATURES = ["cc", "bw_gbs", "is_decode", "log_len", "log_batch",
            "head_dim", "gqa", "is_bf16"]


def features(cells: pd.DataFrame, meta: dict) -> pd.DataFrame:
    p = cells.cell.str.split("|", expand=True)
    f = pd.DataFrame(index=cells.index)
    f["cc"] = cells.gpu_name.map(lambda g: meta[g]["cc"])
    f["bw_gbs"] = cells.gpu_name.map(lambda g: meta[g]["bw"])
    f["is_decode"] = (p[0] == "decode").astype(int)
    f["log_batch"] = np.log2(p[1].str[1:].astype(float))
    f["gqa"] = p[2].str[2:].astype(int) // p[3].str[3:].astype(int)
    f["head_dim"] = p[4].str[1:].astype(int)
    f["log_len"] = np.log2(p[5].str[1:].astype(float))
    f["is_bf16"] = (p[6] == "bf16").astype(int)
    return f


def selector(cells: pd.DataFrame, meta: dict, max_depth=4) -> dict:
    """Fit an interpretable rule on the principal GPUs, test on the rest.

    Split by hardware, not by a random split of one grid: the question is
    whether the rule transfers to a GPU it never saw.
    """
    from sklearn.tree import DecisionTreeClassifier, export_text

    best = winners(cells).set_index(["gpu_name", "cell"])
    lat = cells.set_index(["gpu_name", "cell", "implementation"]).median_us
    tbl = best.reset_index()
    tbl = tbl.join(features(tbl, meta))

    fit = tbl[tbl.gpu_name.str.contains("|".join(FIT_GPUS))]
    hold = tbl[~tbl.gpu_name.str.contains("|".join(FIT_GPUS))]
    if fit.empty or hold.empty:
        return {"error": "need at least one fit GPU and one held-out GPU"}

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(fit[FEATURES], fit.implementation)
    pred = clf.predict(hold[FEATURES])

    def latency(gpu, cell, impl):
        try:
            return lat.loc[(gpu, cell, impl)]
        except KeyError:
            return np.nan

    oracle = hold.median_us.values
    chosen = np.array([latency(g, c, i) for g, c, i
                       in zip(hold.gpu_name, hold.cell, pred)])
    res = {"accuracy": float((pred == hold.implementation.values).mean()),
           "n_heldout": int(len(hold)),
           "median_regret": float(np.nanmedian(chosen / oracle)),
           "p95_regret": float(np.nanpercentile(chosen / oracle, 95)),
           "rule": export_text(clf, feature_names=FEATURES, max_depth=max_depth)}

    for policy in sorted(set(cells.implementation)):
        fixed = np.array([latency(g, c, policy) for g, c in
                          zip(hold.gpu_name, hold.cell)])
        if np.isnan(fixed).mean() > 0.5:
            continue
        res.setdefault("vs_fixed", {})[policy] = {
            "median_speedup": float(np.nanmedian(fixed / chosen)),
            "p95_speedup": float(np.nanpercentile(fixed / chosen, 95)),
            "coverage": float(1 - np.isnan(fixed).mean())}
    return res


# --------------------------------------------------------------------------- #

def nsight(path="results/profile") -> pd.DataFrame:
    """Nsight counters per kernel, from whatever scripts/profile.sh collected.

    ncu emits one row per kernel per metric; pivot to one row per kernel and
    label it with the backend family so it joins to the benchmark rows.
    """
    frames = []
    for f in glob.glob(os.path.join(path, "*", "ncu.csv")):
        try:
            d = pd.read_csv(f, skiprows=lambda i: False, low_memory=False)
        except Exception:
            continue
        cols = {c.lower(): c for c in d.columns}
        kn, mn, mv = (cols.get("kernel name"), cols.get("metric name"),
                      cols.get("metric value"))
        if not (kn and mn and mv):
            continue
        d = d[[kn, mn, mv]].rename(columns={kn: "kernel", mn: "metric",
                                            mv: "value"})
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        w = d.pivot_table(index="kernel", columns="metric", values="value",
                          aggfunc="median").reset_index()
        w["gpu_name"] = os.path.basename(os.path.dirname(f)).replace("-", " ")
        w["backend"] = w.kernel.map(observed_backend)
        frames.append(w)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def environments(path=None):
    path = path or "results/environment"
    out = {}
    for f in glob.glob(os.path.join(path, "*.json")):
        m = json.load(open(f, encoding="utf8"))
        out[m["gpu_name"]] = m
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?", default="results/raw")
    ap.add_argument("--out", default="results/processed")
    a = ap.parse_args(argv)

    from akp.impls import IMPLS

    df = load(a.raw)
    cells = per_cell_median(usable(df))
    env = environments()
    os.makedirs(a.out, exist_ok=True)

    def write(name, obj):
        path = os.path.join(a.out, name)
        if name.endswith(".json"):
            json.dump(obj, open(path, "w", encoding="utf8"), indent=2, default=str)
        else:
            obj.to_parquet(path)

    write("rows.parquet", df)
    write("cells.parquet", cells.drop(columns=["samples"]))

    audit = dispatch_audit(df, IMPLS)
    write("dispatch_audit.parquet", audit)
    write("spread.parquet", spread(cells))
    ncu = nsight()
    if len(ncu):
        write("ncu.parquet", ncu)

    gpus = sorted(set(df.gpu_name))
    inv = []
    pairs = {}
    for i, g1 in enumerate(gpus):
        for g2 in gpus[i + 1:]:
            d = inversions(cells, g1, g2)
            if len(d):
                d["gpu_a"], d["gpu_b"] = g1, g2
                inv.append(d)
            pairs[g1 + " vs " + g2] = rank_correlation(cells, g1, g2)
            if len(d):
                pairs[g1 + " vs " + g2]["practical_inversion_rate"] = float(
                    d.practical.sum() / d.pairs_examined.max())
    write("inversions.parquet",
          pd.concat(inv, ignore_index=True) if inv else pd.DataFrame())

    meta = {g: {"cc": float(str(m["cc_major"]) + "." + str(m["cc_minor"])),
                "bw": (m.get("measured_peak_bw_gbs")
                       or m.get("theoretical_bw_gbs") or 0.0)}
            for g, m in env.items()}
    sel = selector(cells, meta) if len(gpus) > 1 and meta else {"error": "need >1 GPU"}
    write("selector.json", sel)

    sp = spread(cells)
    summary = {
        "n_rows": int(len(df)),
        "n_cells": int(len(cells)),
        "gpus": gpus,
        # Finding 1: does the fastest implementation change across GPUs.
        "winner_flips": winner_flips(cells),
        # Finding 2: how far apart the implementations are, per regime.
        "spread_by_regime": {k: round(float(v), 2) for k, v in
                             sp.groupby("regime").spread.median().items()},
        "median_bw_util": {k: round(float(v), 3) for k, v in
                           df[df.regime == "decode"].groupby("gpu_name")
                           .bw_util.median().dropna().items()},
        "status": df.status.value_counts().to_dict(),
        "dispatch_mismatch_rate": (float(1 - audit.matched.mean())
                                   if len(audit) else None),
        "dispatch_probe_failures": (int((~audit.probed).sum())
                                    if len(audit) else 0),
        "winners": winners(cells).implementation.value_counts().to_dict(),
        # Finding 3: what actually ran, not just whether it was consistent.
        "observed_backends": (audit.groupby("implementation")["observed"]
                              .agg(lambda x: x.value_counts().to_dict()).to_dict()
                              if len(audit) else {}),
        "rank_correlation": pairs,
        "environment": env,
    }
    write("summary.json", summary)

    print(len(df), "rows |", len(cells), "cell medians | GPUs:", gpus)
    print(df.status.value_counts().to_string())
    if len(audit):
        print("dispatch mismatch: {:.1%} of {} probed cells "
              "({} probes returned nothing)".format(
                  1 - audit.matched.mean(), int(audit.probed.sum()),
                  int((~audit.probed).sum())))
    wf = summary["winner_flips"]
    if wf.get("n_cells"):
        print("winner changes across {} GPUs in {:.1%} of {} shared cells".format(
            wf["n_gpus"], wf["flip_rate"], wf["n_cells"]))
    print("spread (slowest/fastest) by regime:", summary["spread_by_regime"])
    for k, v in pairs.items():
        print("{}: spearman {:.3f} over {} cells, practical inversions {:.1%}".format(
            k, v["spearman_median"], v["n_cells"],
            v.get("practical_inversion_rate", float("nan"))))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
