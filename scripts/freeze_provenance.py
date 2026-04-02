"""Recover per-run provenance from the rows themselves.

`results/environment/<gpu>.json` holds ONE manifest per device, rewritten by
whichever run touched that device last. That is lossy in exactly the way that
matters: A10 and A100 decode were recorded at a different harness commit from
their own prefill, and the surviving manifest names only the later one. Reading
the manifest and concluding "every device ran at one commit" is a mistake this
project actually made.

Every row carries its own git_sha, timestamp, timer and telemetry, so the run
structure can be rebuilt from the data rather than trusted from a file that got
overwritten. This writes that table.

    python scripts/freeze_provenance.py results_live/raw

Emits results/processed/run_provenance.csv, one row per
(gpu, regime, git_sha, repeat) actually observed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sha256(path, blocks=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(blocks), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?", default="results_live/raw")
    ap.add_argument("--processed", default="results/processed")
    a = ap.parse_args(argv)

    rows = pd.read_parquet(os.path.join(a.processed, "rows.parquet"))

    # Driver lives only in the manifests. Keep it, but keep it clearly labelled
    # as last-writer-wins rather than as a property of each run.
    envdir = os.path.join(os.path.dirname(a.raw.rstrip("/\\")), "environment")
    drivers = {}
    for f in glob.glob(os.path.join(envdir, "*.json")):
        m = json.load(open(f, encoding="utf8"))
        drivers[m["gpu_name"]] = m.get("driver")

    g = (rows.groupby(["gpu_name", "regime", "git_sha", "repeat"])
             .agg(rows=("status", "size"),
                  ok=("status", lambda s: int((s == "OK").sum())),
                  t_start=("timestamp", "min"),
                  t_end=("timestamp", "max"),
                  timers=("timer", lambda s: "|".join(sorted(set(s.dropna())))))
             .reset_index())
    g["t_start"] = pd.to_datetime(g.t_start, unit="s").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    g["t_end"] = pd.to_datetime(g.t_end, unit="s").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    g["driver_manifest_last_writer"] = g.gpu_name.map(drivers)

    out = os.path.join(a.processed, "run_provenance.csv")
    g.to_csv(out, index=False)
    print("wrote %s (%d runs)" % (out, len(g)))

    # The headline the manifest hides: which (device, regime) pairs disagree on
    # commit, and which devices are internally split.
    per = g.groupby(["gpu_name", "regime"]).git_sha.nunique()
    split_within = per[per > 1]
    by_regime = g.groupby("regime").git_sha.unique()
    print("\ncommits per regime:")
    for reg, shas in by_regime.items():
        print("  %-8s %s" % (reg, ", ".join(sorted(s[:8] for s in shas))))
    if len(split_within):
        print("\n(device, regime) pairs spanning >1 commit:")
        print(split_within.to_string())
    else:
        print("\nno (device, regime) pair spans more than one commit")

    dev_split = g.groupby("gpu_name").git_sha.nunique()
    print("\ndevices whose two regimes were recorded at different commits:")
    for dev, n in dev_split[dev_split > 1].items():
        sub = g[g.gpu_name == dev].groupby("regime").git_sha.first()
        print("  %-24s %s" % (dev, dict((k, v[:8]) for k, v in sub.items())))

    # Checksums over the frozen inputs, so a later release can prove identity.
    man = []
    for f in sorted(glob.glob(os.path.join(a.raw, "*", "*.jsonl"))):
        man.append({"path": os.path.relpath(f, ROOT).replace("\\", "/"),
                    "bytes": os.path.getsize(f), "sha256": sha256(f)})
    mp = os.path.join(a.processed, "raw_manifest.json")
    json.dump({"n_files": len(man), "total_bytes": sum(m["bytes"] for m in man),
               "files": man}, open(mp, "w", encoding="utf8"), indent=2)
    print("\nwrote %s (%d shards, %.1f MB)"
          % (mp, len(man), sum(m["bytes"] for m in man) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
