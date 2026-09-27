"""Between-process dispersion with and without CPU pinning, same host.

    python scripts/numa_ablation.py results_h100_unpinned/processed results/processed

Compares two runs of the same decode grid on one H100 that differ only in
whether each process was pinned to the GPU's NUMA node.

Only eager, warm-cache cells are used. The pinned run also has CUDA-graph cells,
which use a different timer and would inflate the improvement.
"""
import argparse
import json
import sys

import pandas as pd

# Everything that identifies a cell except the repeat, so the spread is across
# process launches of the same work.
KEY = ["implementation", "batch", "hq", "hkv", "head_dim", "seq_len",
       "dtype", "mode", "launch", "cache"]


def dispersion(path, gpu=None, min_repeats=3):
    r = pd.read_parquet(path)
    r = r[(r.status == "OK") & r.median_us.notna() & (r.regime == "decode")
          & (r.launch == "eager") & (r.cache == "warm")]
    if gpu:
        r = r[r.gpu_name.str.contains(gpu)]
    g = r.groupby(KEY)["median_us"]
    cv = (g.std() / g.mean()).dropna()
    # Need at least three launches for a meaningful CV.
    cv = cv[g.count().reindex(cv.index) >= min_repeats]
    return cv, sorted(r.timer.dropna().unique()), r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("unpinned")
    ap.add_argument("pinned")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    rows = []
    for label, path, gpu in (("unpinned", a.unpinned, None),
                             ("pinned", a.pinned, a.gpu)):
        # len(cv) counts (cell, implementation) pairs. n_cells counts cells.
        cv, timers, _ = dispersion(f"{path}/rows.parquet", gpu)
        rows.append(dict(condition=label, n_triples=int(len(cv)),
                         n_cells=int(_.cell.nunique()),
                         median_cv=round(float(cv.median()), 4),
                         p95_cv=round(float(cv.quantile(0.95)), 4),
                         frac_over_25pct=round(float((cv > 0.25).mean()), 4),
                         timers=timers))

    if rows[0]["n_triples"] != rows[1]["n_triples"]:
        print("warning: %d unpinned triples vs %d pinned, the sets are not matched"
              % (rows[0]["n_triples"], rows[1]["n_triples"]), file=sys.stderr)

    out = {"cells": rows,
           "cv_reduction": round(rows[0]["median_cv"] / rows[1]["median_cv"], 2)}
    print(json.dumps(out, indent=2))
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf8"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
