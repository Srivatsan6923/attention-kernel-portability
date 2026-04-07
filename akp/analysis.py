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

def _beside(raw, name):
    """Sibling of the raw directory. The cluster mounts results somewhere
    other than ./results, so anything resolved relative to the cwd is found
    only by accident."""
    return os.path.join(os.path.dirname(raw.rstrip("/\\")), name)


def load(raw="results/raw", dispatch=None) -> pd.DataFrame:
    dispatch = dispatch or _beside(raw, "dispatch.jsonl")
    rows = [json.loads(l)
            for f in glob.glob(os.path.join(raw, "*", "*.jsonl"))
            for l in open(f, encoding="utf8")]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no rows under {raw!r}; expected {raw}/<gpu>/*.jsonl")
    # config_hash already keys the in-shard resume, but grids overlap: a cell
    # in both prefill_full and prefill_gqa is written by each. Left alone,
    # per_cell_median counts those as extra process repeats, so the overlapping
    # cells get double weight in every winner count and a falsely tight CI.
    if "config_hash" in df:
        dup = int(df.config_hash.duplicated().sum())
        if dup:
            print("[akp] dropped %d rows sharing a config_hash with an "
                  "earlier row (overlapping grids)" % dup)
            df = df.drop_duplicates("config_hash", keep="first")
    if os.path.exists(dispatch):
        d = pd.DataFrame([json.loads(l) for l in open(dispatch, encoding="utf8")])
        # Pods append concurrently, so the same dispatch_id can land more than
        # once. Without the dedup the merge is many-to-many and silently
        # duplicates result rows, which would tighten every bootstrap CI.
        d = d.drop_duplicates("dispatch_id")
        df = df.merge(d[["dispatch_id", "kernels"]], on="dispatch_id",
                      how="left", validate="m:1")
    return derive(df, environments(_beside(raw, "environment")))


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

    # Device constants live only in the environment manifest, but the
    # normalized views (per-SM, against L2) need them beside every row.
    df["sm_count"] = df.gpu_name.map(lambda g: (env or {}).get(g, {}).get("sm_count"))
    df["l2_bytes"] = df.gpu_name.map(lambda g: (env or {}).get(g, {}).get("l2_bytes"))
    df["peak_bw_gbs"] = peak
    df["gate_status"] = gate_status(df)
    df["gate_evidence"] = gate_evidence(df)

    # Working set: the tensors that are resident, Q and O at q_len, K and V at
    # N. expand_kv calls repeat_interleave, not expand, so an implementation
    # that takes the expanded path materialises a second, Hq-head copy and
    # holds it alongside the compact cache. Charging it only Hkv attributed
    # that copy to allocator overhead and made a fused decode path read as 5x
    # over its own working set.
    from akp.impls import NAIVE_LIKE
    gqa_mode = df.get("gqa_mode", pd.Series("native", index=df.index))
    expanded = gqa_mode.fillna("native") == "expanded"
    qo = 2 * df.batch * df.hq * np.where(pre, df.seq_len, 1) * df.head_dim * itemsize
    kv_heads = df.hkv + np.where(expanded, df.hq, 0)
    kv = 2 * df.batch * kv_heads * df.seq_len * df.head_dim * itemsize
    # Only the score-matrix impls materialise B*Hq*N*N. Charging it to the fused
    # kernels would hide the one thing that lets them fit.
    scores = np.where(df.implementation.isin(NAIVE_LIKE),
                      df.batch * df.hq * df.seq_len ** 2 * itemsize, 0)
    df["working_set_bytes"] = qo + kv + scores
    # 2**20, not 1e6: bench.peak_memory_mb divides by 2**20, so the column is
    # MiB and a decimal-MB conversion would inflate every ratio by 4.9%.
    # fwd_bwd holds saved activations and grads this does not model, so those
    # rows read high by construction. Below 1 is not an error either: the
    # compiled arms are charged the score matrix because they might materialise
    # it, and a ratio under 1 is the evidence that Inductor fused it away.
    df["mem_overhead"] = df.peak_allocated_mb * 2 ** 20 / df.working_set_bytes

    # Nothing here is measured traffic: there is no profiler data in this
    # dataset. Decode divides by the KV stream it must read; prefill divides by
    # the working set, which is a model of the algorithmic minimum and ignores
    # every tile re-read. Treat it as an ordering, not a roofline coordinate.
    # fwd_bwd also carries the assumed 3.5x in the numerator (see flops above).
    df["arith_intensity"] = flops / np.where(pre, df.working_set_bytes, df.kv_bytes)

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


def _series(cells: pd.DataFrame, drop: int) -> pd.Series:
    """The cell key with one field removed, for pairing along that axis."""
    p = cells.cell.str.split("|", expand=True)
    return p.drop(columns=drop).apply("|".join, axis=1)


def scaling(cells: pd.DataFrame) -> pd.DataFrame:
    """Empirical alpha in T(N) ~ N^alpha, per (gpu, impl, N-stripped cell).

    The fit can only span the N values that impl actually completed, and the
    score-matrix impls lose their large-N points to OOM_PREDICTED: 50 of
    P0-naive's 73 prefill series stop at N=1024 where the fused kernels reach
    8192. So alpha is not comparable across impls without also reading the
    range it was fitted over -- P0-naive measures 1.90 on its truncated series
    against 1.55 on the ones that ran to the end, and neither number is wrong,
    they are answers about different N. n_min, n_max and n_points travel with
    alpha for exactly that reason.
    """
    d = cells.assign(n=cells.cell.str.split("|").str[5].str[1:].astype(float),
                     series=_series(cells, 5))
    out = []
    for (g, impl, s), sub in d.groupby(["gpu_name", "implementation", "series"]):
        t = sub.groupby("n").median_us.median()      # one point per length
        # Two points fit a line exactly and r2 is 1 by construction, which reads
        # as a confident exponent off nothing.
        if len(t) < 3:
            continue
        x, y = np.log(t.index.values), np.log(t.values)
        a, b = np.polyfit(x, y, 1)
        ss = ((y - y.mean()) ** 2).sum()
        out.append(dict(gpu_name=g, implementation=impl, series=s,
                        regime=s.split("|")[0], alpha=float(a),
                        r2=float(1 - ((y - (a * x + b)) ** 2).sum() / ss)
                        if ss else np.nan,
                        n_points=len(t), n_min=float(t.index.min()),
                        n_max=float(t.index.max())))
    return pd.DataFrame(out)


def portability(cells: pd.DataFrame) -> pd.DataFrame:
    """What standardising on one backend costs: best-in-cell / this impl.

    P <= 1, and P == 1 means this impl won the cell. Restricted to the cells
    every GPU ran, the same shared set winner_flips counts over -- otherwise an
    impl buys portability by being absent wherever it loses.
    """
    shared = cells.groupby("cell").gpu_name.nunique()
    d = cells[cells.cell.isin(shared[shared == shared.max()].index)]
    best = d.groupby(["gpu_name", "cell"]).median_us.transform("min")
    return pd.DataFrame({"gpu_name": d.gpu_name, "implementation": d.implementation,
                         "cell": d.cell, "regime": d.cell.str.split("|").str[0],
                         "median_us": d.median_us, "best_us": best,
                         "p_ratio": best / d.median_us}).reset_index(drop=True)


def cache_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Cold-L2 penalty, pairing rows that differ only in the cache field.

    inner_k is part of the pairing, not a label on it. block_bench zeroes the
    flush buffer once per BLOCK of k calls, so only k == 1 is genuinely cold --
    and k is chosen per process from a timing probe, so one cell's five repeats
    can land on different k. Collapsing them would report a k=3 block, where
    two of every three calls are warm, as a cold measurement. Cold groups with
    no warm partner at their k are kept with us_warm null rather than dropped.
    """
    d = usable(df)
    keys = ["gpu_name", "implementation", "key", "inner_k"]
    g = (d.assign(key=_series(d, 9))
          .groupby(keys + ["cache"])
          .agg(us=("median_us", "median"), cell=("cell", "first"),
               n=("median_us", "size")).reset_index())
    m = (g[g.cache == "cold"].drop(columns="cache")
         .merge(g[g.cache == "warm"].drop(columns="cache"), on=keys,
                how="left", suffixes=("_cold", "_warm")))
    m["sensitivity"] = (m.us_cold - m.us_warm) / m.us_warm
    # Carried per row the way inversions carries pairs_examined: the panel has
    # to state how few pairs there are, and recomputing it there would drift.
    m["n_pairs"] = int(m.sensitivity.notna().sum())
    return m.drop(columns="key").rename(columns={"n_cold": "reps_cold",
                                                 "n_warm": "reps_warm"})


def oom_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Predicted peak against measured peak, for the score-matrix impls.

    The OOM_PREDICTED cells were refused rather than run, so the parquet holds
    no predicted/actual pair anywhere. Recomputing the same prediction on the OK
    rows is the only way to ask whether the predictor is calibrated; the refused
    cells come along as a right-censored band (actual_gb null).
    """
    from akp.impls import NAIVE_LIKE, Cfg, naive_peak_bytes

    out = []
    for r in df[df.implementation.isin(NAIVE_LIKE)
                & df.status.isin(("OK", "OOM_PREDICTED"))].itertuples():
        cfg = Cfg(regime=r.regime, B=int(r.batch), Hq=int(r.hq), Hkv=int(r.hkv),
                  D=int(r.head_dim), N=int(r.seq_len), dtype=r.dtype,
                  mode=r.mode, launch=r.launch, cache=r.cache,
                  causal=bool(r.causal))
        # /1e9 to match the decimal GB run.py records in predicted_peak_gb;
        # peak_allocated_mb is MiB, so it needs 2**20 before it can be compared.
        pred = naive_peak_bytes(cfg) / 1e9
        act = (r.peak_allocated_mb * 2 ** 20 / 1e9 if r.status == "OK"
               else np.nan)
        out.append(dict(gpu_name=r.gpu_name, implementation=r.implementation,
                        cell=r.cell, predicted_gb=pred, actual_gb=act,
                        ratio=pred / act if act > 0 else np.nan,
                        status=r.status))
    # The predictor models the forward score matrix only, so fwd_bwd rows
    # under-predict by construction; the mode is in the cell key, facet on it.
    return pd.DataFrame(out)


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
        # fillna(True) used to turn "no verdict" into "passed", which is the
        # one thing a correctness gate must never do silently. gate_status
        # names the three states, and only an actual failure disqualifies.
        # Ranking requires a passing verdict for that device and class.
        # "ungated" is an absence of evidence and does not qualify a row for a
        # correctness-gated comparison; it is reported, not counted as a pass.
        ok &= gate_status(df) == "pass"
    return df[ok]


def gate_status(df: pd.DataFrame) -> pd.Series:
    """pass / fail / ungated per row, propagated over the equivalence class.

    The gate runs on one representative per (implementation, D, dtype, causal,
    GQA, mode) class, so a verdict belongs to the class, not to the row that
    happened to carry it. A class whose representative failed is disqualified
    whole; a class that was never gated is "ungated", which is an absence of
    evidence and is reported as such rather than counted as a pass.
    """
    st = pd.Series("ungated", index=df.index, dtype=object)
    if "correctness_pass" not in df:
        return st
    if "gate_class" in df and "gpu_name" in df:
        # Keyed on the DEVICE as well as the class. gate_class carries no GPU,
        # so propagating on it alone let a pass recorded on one device mark an
        # untested class on another as verified.
        key = df.gpu_name.astype(str) + "\x00" + df.gate_class.astype(str)
        ok = df.correctness_pass == True
        bad = df.correctness_pass == False
        passed = set(key[ok].dropna())
        failed = set(key[bad].dropna())
        st[key.isin(passed)] = "pass"
        st[key.isin(failed)] = "fail"      # a failure outranks a sibling pass
        st[df.gate_class.isna()] = "ungated"
    else:
        st[df.correctness_pass == True] = "pass"
        st[df.correctness_pass == False] = "fail"
    return st


def gate_evidence(df: pd.DataFrame) -> pd.Series:
    """direct / inherited / none -- how a row's verdict was obtained.

    The gate runs on one representative per (device, class), so most rows
    legitimately inherit within their own device. "none" is the category that
    must never be silently counted as a pass.
    """
    ev = pd.Series("none", index=df.index, dtype=object)
    if "correctness_pass" not in df or "gate_class" not in df or "gpu_name" not in df:
        return ev
    key = df.gpu_name.astype(str) + "\x00" + df.gate_class.astype(str)
    verdicted = set(key[df.correctness_pass.notna()].dropna())
    ev[key.isin(verdicted)] = "inherited"
    ev[df.correctness_pass.notna()] = "direct"
    return ev


# --------------------------------------------------------------------------- #
# Portability: rankings, inversions, rank correlation
# --------------------------------------------------------------------------- #

def per_cell_median(df: pd.DataFrame) -> pd.DataFrame:
    """Median per (gpu, cell, impl), keeping repeats for the bootstrap."""
    g = df.groupby(["gpu_name", "cell", "implementation"])
    # repeat_ids travel with the samples so the bootstrap can resample whole
    # process launches jointly across implementations rather than resampling
    # each implementation's array on its own.
    return g.agg(median_us=("median_us", "median"),
                 reps=("median_us", "size"),
                 samples=("median_us", list),
                 repeat_ids=("repeat", list)).reset_index()


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


def paired_ratio_ci(a, b, n=2000, seed=0, ids_a=None, ids_b=None):
    """Cluster bootstrap over process launches, on the reported estimator.

    Two things this has to get right, and previously did not.

    The point estimate is a ratio of MEDIANS of per-process medians, because
    per_cell_median aggregates with median. Resampling the mean gave an
    interval around a different statistic than the one being reported.

    Both implementations were measured inside the same process launches, so a
    launch is the cluster: resample launch ids once and index both arrays with
    them. Drawing the two arrays independently throws the pairing away and
    understates the drift they share, which is the whole reason the repeats
    are process-level.

    Falls back to independent resampling only when the two sides genuinely do
    not share launches (unequal or disjoint ids), which happens when one
    implementation was unsupported for part of a sweep.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    nan3 = (float("nan"), float("nan"), float("nan"), 0)
    if len(a) == 0 or len(b) == 0:
        return nan3

    # One population for the estimate and the interval. Ranking on every
    # repeat while resampling only the shared ones described two different
    # things, and produced a 50x point ratio beside a [0.5, 0.5] interval.
    if ids_a is not None and ids_b is not None:
        if len(set(ids_a)) != len(ids_a) or len(set(ids_b)) != len(ids_b):
            raise ValueError("duplicate launch ids: %r / %r" % (ids_a, ids_b))
        ia = {v: i for i, v in enumerate(ids_a)}
        ib = {v: i for i, v in enumerate(ids_b)}
        keys = sorted(set(ia) & set(ib))
        a = a[[ia[k] for k in keys]]
        b = b[[ib[k] for k in keys]]

    # A single launch cannot support an interval: resampling one value returns
    # it every draw, giving zero width that then "excludes 1" for any ratio.
    if len(a) < 2 or len(b) < 2:
        return float(np.median(a) / np.median(b)) if len(a) and len(b) else float("nan"), \
               float("nan"), float("nan"), min(len(a), len(b))

    ratio = float(np.median(a) / np.median(b))
    if len(a) == len(b):
        pick = rng.integers(0, len(a), (n, len(a)))
        r = np.median(a[pick], axis=1) / np.median(b[pick], axis=1)
    else:
        ra = np.median(rng.choice(a, (n, len(a)), replace=True), axis=1)
        rb = np.median(rng.choice(b, (n, len(b)), replace=True), axis=1)
        r = ra / rb
    return ratio, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), len(a)


def separated(ratio, lo, margin=None):
    """The runner-up/fastest ratio clears the margin and its interval excludes 1.

    Only a lower bound above 1 is evidence. The ratio is >= 1 by construction,
    so an upper bound below 1 meant the interval contradicted its own point
    estimate; accepting that case let a pathological pairing register as a
    separated winner.
    """
    m = PRACTICAL if margin is None else margin
    if not np.isfinite(ratio) or not np.isfinite(lo):
        return False
    return bool(lo > 1.0 and ratio >= m)


def _boot_ratio(a, b, n=2000, seed=0, ids_a=None, ids_b=None):
    """Back-compatible two-tuple wrapper."""
    _, lo, hi, _ = paired_ratio_ci(a, b, n=n, seed=seed, ids_a=ids_a, ids_b=ids_b)
    return lo, hi


def _boot_mean(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def stability(cells: pd.DataFrame) -> pd.DataFrame:
    """Between-process dispersion -- the term the 5-process design exists for.

    Reps inside one process share clocks and allocator state, so the only
    honest unit of variation is the per-process median. Runs on
    per_cell_median's output before main() drops the samples column.
    """
    out = []
    for r in cells.itertuples():
        s = np.asarray(r.samples, float)
        lo, hi = _boot_mean(s)
        out.append(dict(
            gpu_name=r.gpu_name, implementation=r.implementation, cell=r.cell,
            regime=r.cell.split("|")[0], n_repeats=len(s), mean_us=s.mean(),
            # ddof=1: five processes are a sample of the machine's
            # process-to-process behaviour, not the population of them.
            std_us=s.std(ddof=1) if len(s) > 1 else np.nan,
            ci_lo=lo, ci_hi=hi, rel_ci_width=(hi - lo) / (2 * r.median_us)))
    d = pd.DataFrame(out)
    d["cv"] = d.std_us / d.mean_us
    return d


def inversions(cells: pd.DataFrame, g1: str, g2: str, fdr=0.05,
               regime=None):
    """Pairs whose ordering flips between two GPUs.

    Statistical: signs differ and both bootstrap CIs exclude 1. Practical: both
    ratios past 10%. We lead with the practical count, because at this family
    size a nominal alpha manufactures inversions out of noise.

    g1 and g2 are matched exactly, never as substrings: "NVIDIA A10" is a
    substring of "NVIDIA A100-SXM4-80GB", so a contains() match would fold the
    two devices into one and compare a device against itself.
    """
    out, examined = [], 0
    ids = "repeat_ids" in cells.columns
    for cell in sorted(set(cells[cells.gpu_name == g1].cell)
                       & set(cells[cells.gpu_name == g2].cell)):
        if regime and not cell.startswith(regime):
            continue
        sub = cells[cells.cell == cell]
        s1 = sub[sub.gpu_name == g1].set_index("implementation")
        s2 = sub[sub.gpu_name == g2].set_index("implementation")
        common = sorted(set(s1.index) & set(s2.index))
        for i, a in enumerate(common):
            for b in common[i + 1:]:
                examined += 1
                r1 = s1.median_us[a] / s1.median_us[b]
                r2 = s2.median_us[a] / s2.median_us[b]
                if np.sign(r1 - 1) == np.sign(r2 - 1):
                    continue
                ka = dict(ids_a=s1.repeat_ids[a], ids_b=s1.repeat_ids[b]) if ids else {}
                kb = dict(ids_a=s2.repeat_ids[a], ids_b=s2.repeat_ids[b]) if ids else {}
                lo1, hi1 = _boot_ratio(s1.samples[a], s1.samples[b], **ka)
                lo2, hi2 = _boot_ratio(s2.samples[a], s2.samples[b], **kb)
                sig = bool((lo1 > 1 or hi1 < 1) and (lo2 > 1 or hi2 < 1))
                practical = bool(max(r1, 1 / r1) >= PRACTICAL
                                 and max(r2, 1 / r2) >= PRACTICAL)
                out.append(dict(
                    cell=cell, a=a, b=b, r1=r1, r2=r2,
                    sig=sig, practical=practical,
                    # An inversion required to be both must intersect the two
                    # flags explicitly; they are independent conditions.
                    sig_and_practical=sig and practical))
    df = pd.DataFrame(out)
    # The denominator is the count of pairs COMPARED, not the counter's value
    # at the last inversion. Writing it after the loop makes every row carry
    # the true total, and attrs carries it even when nothing inverted.
    if len(df):
        df["pairs_examined"] = examined
    df.attrs["pairs_examined"] = examined
    return df, examined


def rank_correlation(cells: pd.DataFrame, g1: str, g2: str) -> dict:
    from scipy.stats import kendalltau, spearmanr
    rs, ks = [], []
    for cell in set(cells.cell):
        sub = cells[cells.cell == cell]
        s1 = sub[sub.gpu_name == g1].set_index("implementation").median_us
        s2 = sub[sub.gpu_name == g2].set_index("implementation").median_us
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
            # What a dispatch answer is actually about. The launched kernel is
            # a function of these, not of batch or length, so this is the unit
            # coverage should be judged in.
            # r["dtype"], not r.dtype: r is a Series, so attribute access
            # returns the Series' own dtype ("object") and silently collapses
            # fp16 and bf16 into one class.
            dispatch_class="|".join([r.implementation, r.gpu_name,
                                     "D%s" % r.head_dim, str(r["dtype"]),
                                     "gqa%d" % (r.hq // r.hkv),
                                     str(r["mode"]), str(r.launch)]),
            probed=probed,
            matched=(any(re.search(p, k, re.I) for p in pats) if probed
                     else float("nan")),
            # Inductor's own counter, and nothing else. This is the direct
            # instrument for "was the graph rewritten into SDPA", and it is
            # what invalidator I2 turns on, so it does not get ORed with a
            # guess. notna first: the column is NaN wherever no counter was
            # captured and bool(nan) is True, which had marked FA2, Triton and
            # naive as compiler-rewritten on 96% of rows.
            fused_into_sdpa=bool(pd.notna(r.get("fuse_attention"))
                                 and r.get("fuse_attention")),
            # Corroboration, reported beside the counter rather than merged
            # into it: an attention-library kernel in a compiled path's trace.
            # Kept narrow -- "cutlass" and "fmha" appear in ordinary GEMM names
            # and inside Flash template arguments, so they match the un-fused
            # path too and cannot support this claim.
            attn_kernel_in_trace=bool(
                r.implementation.startswith(("P1-", "D1-")) and probed
                and re.search(r"flash_fwd|flash_bwd|cudnn.*attn", k, re.I)),
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
    # Required, not defaulted: results/raw holds a 68-row laptop smoke shard,
    # and a bare run silently rebuilt results/processed from it, discarding a
    # multi-GPU dataset. Naming the input is cheap; losing the output is not.
    ap.add_argument("raw")
    ap.add_argument("--out", default="results/processed")
    a = ap.parse_args(argv)

    from akp.impls import IMPLS

    df = load(a.raw)
    cells = per_cell_median(usable(df))
    # Same directory load() used. Called bare it defaulted to a path
    # relative to the cwd, came back empty on the cluster, and the
    # selector below silently reported 'need >1 GPU' with five of them.
    env = environments(_beside(a.raw, "environment"))
    os.makedirs(a.out, exist_ok=True)

    def write(name, obj):
        path = os.path.join(a.out, name)
        if name.endswith(".json"):
            json.dump(obj, open(path, "w", encoding="utf8"), indent=2, default=str)
        else:
            obj.to_parquet(path)

    write("rows.parquet", df)
    write("scaling.parquet", scaling(cells))
    write("portability.parquet", portability(cells))
    # Before the drop below: the per-repeat medians are the whole input.
    write("stability.parquet", stability(cells))
    write("cells.parquet", cells.drop(columns=["samples"]))
    write("cache_sensitivity.parquet", cache_sensitivity(df))
    write("oom_calibration.parquet", oom_calibration(df))

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
            d, n_examined = inversions(cells, g1, g2)
            if len(d):
                d["gpu_a"], d["gpu_b"] = g1, g2
                inv.append(d)
            key = g1 + " vs " + g2
            pairs[key] = rank_correlation(cells, g1, g2)
            # Recorded whether or not anything inverted: a pair with zero
            # inversions has a rate of 0, not a missing rate.
            pairs[key]["pairs_examined"] = int(n_examined)
            pairs[key]["practical_inversions"] = int(d.practical.sum()) if len(d) else 0
            pairs[key]["practical_inversion_rate"] = (
                float(d.practical.sum() / n_examined) if n_examined else float("nan"))
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
        "dispatch_class_coverage": (
            float(audit.groupby("dispatch_class").probed.any().mean())
            if len(audit) else None),
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
        # Per class, because that is what the claim is about. CUPTI drops a
        # short fused kernel roughly half the time, so the per-row rate reads
        # far worse than the evidence actually is: every class is probed many
        # times over and one capture answers it.
        cov = audit.groupby("dispatch_class").probed.any()
        print("dispatch mismatch: {:.1%} of {} probed rows | classes covered "
              "{}/{} ({:.0%}), median {} captures each".format(
                  1 - audit.matched.mean(), int(audit.probed.sum()),
                  int(cov.sum()), len(cov), cov.mean(),
                  int(audit.groupby("dispatch_class").probed.sum().median())))
    wf = summary["winner_flips"]
    if wf.get("n_cells"):
        print("winner changes across {} GPUs in {:.1%} of {} shared cells".format(
            wf["n_gpus"], wf["flip_rate"], wf["n_cells"]))
    print("spread (slowest/fastest) by regime:", summary["spread_by_regime"])
    for k, v in pairs.items():
        print("{}: spearman {:.3f} over {} cells, practical inversions {:.1%}".format(
            k, v["spearman_median"], v["n_cells"],
            v.get("practical_inversion_rate", float("nan"))))
    # Printed because it is the last artifact and the easiest to lose: it wrote
    # {"error": "need >1 GPU"} on a two-GPU dataset for as long as nothing
    # showed it, and a selector that never ran looks exactly like one nobody
    # has got to yet.
    if "accuracy" in sel:
        print("selector: {:.1%} top-1 on {} held-out cells, median regret "
              "{:.3f}, p95 {:.3f}".format(sel["accuracy"], sel["n_heldout"],
                                          sel["median_regret"], sel["p95_regret"]))
    else:
        print("selector: not fitted --", sel["error"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
