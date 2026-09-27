"""Sweep runner. Takes a grid, measures each cell and writes one JSON line per row.

Pods can be evicted and a CUDA error can kill the process, so the runner is
built to resume.

  * Rows are keyed by config_hash and skipped if already on disk, so a restart
    continues where it stopped.
  * Each row is appended and fsync'd, so a hard kill loses at most one cell.
  * Implementation order is shuffled per config.
  * Cells that cannot fit are recorded as OOM_PREDICTED and never attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from akp import bench, check
from akp.impls import NAIVE_LIKE, Cfg, impls_for, naive_peak_bytes

# Grids are plain Python because the skip rules are conditional.

def _prod(**axes):
    keys = list(axes)
    for vals in itertools.product(*(axes[k] for k in keys)):
        yield dict(zip(keys, vals))


def grid(name):
    if name == "smoke":
        cfgs = [Cfg("prefill", B=2, Hq=8, Hkv=8, D=64, N=n, dtype=d)
                for n in (128, 256) for d in ("bf16", "fp16")]
        cfgs += [Cfg("prefill", B=2, Hq=8, Hkv=2, D=64, N=256, mode="fwd_bwd")]
        cfgs += [Cfg("decode", B=2, Hq=8, Hkv=2, D=64, N=n, causal=False)
                 for n in (256, 512)]
        cfgs += [Cfg("decode", B=2, Hq=8, Hkv=2, D=64, N=512, causal=False,
                     launch="cudagraph")]
        return cfgs

    if name == "prefill_full":
        return [Cfg("prefill", B=a["B"], Hq=32, Hkv=32, D=a["D"], N=a["N"],
                    dtype=a["dtype"], mode=a["mode"])
                for a in _prod(B=[1, 4, 16], D=[64, 128],
                               N=[256, 512, 1024, 2048, 4096, 8192],
                               dtype=["bf16", "fp16"], mode=["fwd", "fwd_bwd"])]

    if name == "prefill_gqa":            # most deployed models use GQA
        return [Cfg("prefill", B=4, Hq=32, Hkv=8, D=128, N=a["N"],
                    dtype=a["dtype"])
                for a in _prod(N=[512, 2048, 4096], dtype=["bf16", "fp16"])]

    if name == "prefill_noncausal":
        return [Cfg("prefill", B=4, Hq=32, Hkv=32, D=128, N=a["N"],
                    dtype=a["dtype"], causal=False)
                for a in _prod(N=[512, 2048, 4096], dtype=["bf16", "fp16"])]

    if name == "prefill_cold":           # cold L2, reported separately
        return [Cfg("prefill", B=4, Hq=32, Hkv=32, D=128, N=n, cache="cold")
                for n in (512, 2048, 4096)]

    if name == "decode_full":
        return [Cfg("decode", B=a["B"], Hq=32, Hkv=a["Hkv"], D=a["D"],
                    N=a["N"], dtype=a["dtype"], causal=False)
                for a in _prod(B=[1, 8, 32, 64], Hkv=[32, 8], D=[64, 128],
                               N=[512, 1024, 2048, 4096, 8192, 16384],
                               dtype=["bf16", "fp16"])]

    if name == "decode_cudagraph":       # small shapes, where launch overhead matters
        return [Cfg("decode", B=a["B"], Hq=32, Hkv=a["Hkv"], D=a["D"],
                    N=a["N"], dtype=a["dtype"], causal=False,
                    launch="cudagraph")
                for a in _prod(B=[1, 8], Hkv=[32, 8], D=[64, 128],
                               N=[512, 1024, 2048], dtype=["bf16", "fp16"])]

    if name == "decode_cold":            # cold L2, reported separately
        return [Cfg("decode", B=b, Hq=32, Hkv=8, D=128, N=n, causal=False,
                    cache="cold")
                for b in (1, 32) for n in (512, 8192)]

    if name == "profile":
        # Cells picked for profiling: where rankings are likely to change and
        # where the limiting resource changes.
        return [
            Cfg("prefill", B=4, Hq=32, Hkv=32, D=128, N=512),    # small, launch-sensitive
            Cfg("prefill", B=4, Hq=32, Hkv=32, D=128, N=2048),   # the reference prefill
            Cfg("prefill", B=4, Hq=32, Hkv=32, D=64, N=2048),    # head-dim tile change
            Cfg("prefill", B=1, Hq=32, Hkv=32, D=128, N=8192),   # long context
            Cfg("prefill", B=4, Hq=32, Hkv=8, D=128, N=2048),    # GQA prefill
            Cfg("decode", B=1, Hq=32, Hkv=8, D=128, N=512, causal=False),    # launch bound
            Cfg("decode", B=1, Hq=32, Hkv=8, D=128, N=8192, causal=False),   # bandwidth bound
            Cfg("decode", B=32, Hq=32, Hkv=8, D=128, N=8192, causal=False),  # after the crossover
            Cfg("decode", B=64, Hq=32, Hkv=8, D=128, N=16384, causal=False), # largest decode
            Cfg("decode", B=32, Hq=32, Hkv=32, D=128, N=8192, causal=False), # MHA vs GQA
        ]

    raise SystemExit("unknown grid " + repr(name) + ". known: smoke "
                     "prefill_full prefill_gqa prefill_noncausal prefill_cold "
                     "decode_full decode_cudagraph decode_cold profile")


# Compiled implementations run on a subset of the grid because Inductor
# autotunes per shape and that dominates wall time.
#
#   D1-inductor         decode is bandwidth-bound, so a few shapes are enough
#   P1-inductor-nofuse  these two check whether Inductor rewrites the naive
#   P1-inductor-where   form into SDPA, which does not depend on B or N
D1_STRIP = {(1, 128, 8), (32, 128, 8)}


def impl_applies(impl_name, cfg):
    if impl_name == "D1-inductor":
        return ((cfg.B, cfg.D, cfg.Hkv) in D1_STRIP
                and cfg.N in (512, 2048, 8192) and cfg.dtype == "bf16")
    if impl_name in ("P1-inductor-nofuse", "P1-inductor-where"):
        return (cfg.B == 4 and cfg.D == 128 and cfg.dtype == "bf16"
                and cfg.mode == "fwd" and cfg.N in (512, 2048, 4096))
    return True


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

def git_sha():
    # Set at image build because /workspace is not a git repo. Without it all
    # cluster rows would share one sha and a rebuilt image would resume onto
    # stale rows.
    env = os.environ.get("AKP_GIT_SHA")
    if env:
        return env
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def versions():
    import importlib
    out = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for m in ("triton", "flash_attn", "flashinfer", "flash_attn_interface"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "present")
        except Exception:
            out[m] = None
    return out


def _driver():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                               "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def manifest(device):
    d = bench.device_info(device)
    d.update({"versions": versions(), "git_sha": git_sha(),
              "event_overhead_us": bench.event_overhead_us(device),
              "measured_peak_bw_gbs": bench.measured_peak_bw_gbs(device),
              "driver": _driver(), "timestamp": time.time()})
    return d


class DispatchTable:
    """Store each kernel trace once and give rows a 16-char id for it.

    A trace is about 2 KB of mangled C++ names that repeats across shapes and
    repeats. Traces go to results/dispatch.jsonl and analysis.py joins them back.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seen = {}
        if self.path.exists():
            for line in self.path.open(encoding="utf8"):
                try:
                    rec = json.loads(line)
                    self.seen[rec["dispatch_id"]] = rec["kernels"]
                except Exception:
                    pass

    def intern(self, trace, impl, gpu):
        did = hashlib.sha1(trace.encode()).hexdigest()[:16]
        if did not in self.seen:
            self.seen[did] = trace
            with self.path.open("a", encoding="utf8") as fh:
                fh.write(json.dumps({"dispatch_id": did, "implementation": impl,
                                     "gpu_name": gpu, "n_kernels": trace.count("|") + 1,
                                     "kernels": trace}) + chr(10))
                fh.flush()
                os.fsync(fh.fileno())
        return did


def config_hash(cfg, impl, dev, sha, repeat):
    key = "|".join([cfg.key(), impl, dev["gpu_name"], sha, str(repeat)])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #

def oom_predicted_bytes(impl_name, cfg, dev_info):
    """Bytes this cell would need, or 0 if it fits in 85% of device memory.

    Used by both the timed path and prewarm. Attempting the allocation (450 GB
    at B=16 N=8192) leaves the allocator in a bad state even after the OOM is
    caught, and the next Inductor autotune then fails with an illegal memory
    access.
    """
    if impl_name not in NAIVE_LIKE:
        return 0
    need = naive_peak_bytes(cfg)
    return need if need > 0.85 * dev_info["total_memory_gb"] * 1e9 else 0

GATED = set()   # classes that already have a correctness verdict

# A config can gate its class only if the unchunked naive baseline fits well in
# memory. 8 GB covers B=4 N=4096 and smaller, which every class in every grid
# contains. Decode has no score matrix, so naive_peak_bytes is 0 there and
# every decode cell qualifies.
GATE_BUDGET_BYTES = 8e9

SEEN_CLASSES = set()   # used to warn about classes that were never gated

# After one of these errors the CUDA context is unusable and later results are
# garbage. Rows are fsync'd per cell, so aborting loses nothing. The Job
# restarts the index and resume skips rows already on disk.
STICKY_CUDA = ("illegal memory access", "unspecified launch failure",
               "device-side assert", "misaligned address",
               "CUDA error: an illegal instruction")


def _is_sticky(msg: str) -> bool:
    return any(s in msg for s in STICKY_CUDA)


def run_cell(impl, cfg, device, dev_info, reps):
    row = {"implementation": impl.name, "status": "OK"}

    if not impl.supports(cfg, dev_info) or not impl_applies(impl.name, cfg):
        return {**row, "status": "UNSUPPORTED"}

    need = oom_predicted_bytes(impl.name, cfg, dev_info)
    if need:
        return {**row, "status": "OOM_PREDICTED",
                "predicted_peak_gb": round(need / 1e9, 2)}

    built = None
    try:
        built = impl.build(cfg, device)
        out = built.fn()
        torch.cuda.synchronize()

        # Correctness depends on (impl, D, dtype, causal, GQA, mode) and not
        # on batch or length, so one cell per class is checked. The check waits
        # for a small config of the class because the unchunked baseline
        # cannot run at B=16 N=8192.
        cls = (impl.name, cfg.D, cfg.dtype, cfg.causal, cfg.gqa, cfg.mode)
        if cls not in GATED and naive_peak_bytes(cfg) < GATE_BUDGET_BYTES:
            GATED.add(cls)
            row.update(check.gate(cfg, device, out,
                                  built.meta.get("out_layout", "bhsd"),
                                  fn=built.fn))
        SEEN_CLASSES.add(cls)
        row["gate_class"] = "|".join(str(x) for x in cls)
        row["dispatch_trace"] = check.dispatch_probe(built.fn)
        row.update(bench.peak_memory_mb(built.fn, device))
        row.update(bench.measure(built.fn, cfg, device, reps=reps))

        if not row.get("correctness_pass", True):
            row["status"] = "NUMERICAL_FAIL"

        row.update({k: v for k, v in built.meta.items()
                    if not k.startswith("_")})

    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    except NotImplementedError:
        row["status"] = "UNSUPPORTED"
    except RuntimeError as exc:
        # "No available kernel" means the backend does not cover this shape.
        msg = str(exc)
        row["status"] = ("UNSUPPORTED" if "No available kernel" in msg
                         or "not supported" in msg.lower() else "ERROR")
        row["error"] = ("RuntimeError: " + msg)[:400]
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = (type(exc).__name__ + ": " + str(exc))[:400]
    finally:
        del built
        # empty_cache raises once the context is broken. Keep the status
        # recorded above.
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            row.setdefault("error", ("RuntimeError: " + str(exc))[:400])
    if _is_sticky(row.get("error", "")):
        row["fatal"] = True
    return row


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--impl", default=None, help="restrict to one implementation")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--repeat", type=int, default=0,
                    help="process-level repeat index, part of the config hash")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--require-gpu", default=None,
                    help="refuse to run unless the device name contains this; "
                         "the cluster also has 40GB PCIe and MIG A100s and "
                         "silently benchmarking one would corrupt the anchor")
    ap.add_argument("--index", type=int, default=None,
                    help="single task index, repeat and shard are derived as "
                         "index // nshards and index %% nshards, so a k8s "
                         "Indexed Job needs no shell arithmetic")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the grid across pods, each shard keeps every "
                         "implementation for its configs, so the interleaving "
                         "that makes drift common-mode is preserved")
    ap.add_argument("--prewarm", action="store_true",
                    help="build and run each cell once to fill the compile "
                         "caches, then exit without writing rows")
    a = ap.parse_args(argv)

    if a.index is not None:
        a.repeat, a.shard = divmod(a.index, a.nshards)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    device = torch.device("cuda", 0)
    dev_info = bench.device_info(device)
    if a.require_gpu and a.require_gpu.lower() not in dev_info["gpu_name"].lower():
        raise SystemExit("expected a GPU matching %r, got %r"
                         % (a.require_gpu, dev_info["gpu_name"]))
    sha = git_sha()

    slug = dev_info["gpu_name"].replace(" ", "-").replace("/", "-")
    outdir = Path(a.out) / slug
    outdir.mkdir(parents=True, exist_ok=True)
    envdir = Path(a.out).parent / "environment"
    envdir.mkdir(parents=True, exist_ok=True)
    (envdir / (slug + ".json")).write_text(json.dumps(manifest(device), indent=2))

    # Next to the raw shards, so on a cluster the traces also land on the
    # mounted volume.
    dispatch = DispatchTable(Path(a.out).parent / "dispatch.jsonl")
    shard = outdir / (a.grid + "_" + (a.impl or "all") + "_r" + str(a.repeat)
                      + "_s" + str(a.shard) + ".jsonl")
    done = set()
    if shard.exists() and not a.force:
        for line in shard.open(encoding="utf8"):
            try:
                done.add(json.loads(line)["config_hash"])
            except Exception:
                pass
    print("[akp] {} sm{}{} | grid={} repeat={} | {} rows already present".format(
        dev_info["gpu_name"], dev_info["cc_major"], dev_info["cc_minor"],
        a.grid, a.repeat, len(done)), flush=True)

    # Shuffle before sharding. mode and dtype are the innermost product axes,
    # so a plain i % nshards split would put all fwd configs on one node and
    # all fwd_bwd on another. The fixed seed keeps the split the same when a
    # pod resumes.
    cfgs = grid(a.grid)
    random.Random(20260218).shuffle(cfgs)
    cfgs = [c for i, c in enumerate(cfgs) if i % a.nshards == a.shard]

    if a.prewarm:
        # Inductor autotune and FlashInfer JIT are paid per shape. Doing them
        # here keeps them out of the timed repeats.
        for ci, cfg in enumerate(cfgs):
            for impl in impls_for(cfg.regime):
                if not impl.supports(cfg, dev_info) or not impl_applies(impl.name, cfg):
                    continue
                if oom_predicted_bytes(impl.name, cfg, dev_info):
                    continue
                try:
                    impl.build(cfg, device).fn()
                    torch.cuda.synchronize()
                except Exception as exc:
                    print("[akp] prewarm skip {} {}: {}".format(
                        impl.name, cfg.key(), type(exc).__name__), flush=True)
                finally:
                    torch.cuda.empty_cache()
            print("[akp] prewarm {}/{} {}".format(ci + 1, len(cfgs), cfg.key()),
                  flush=True)
        print("[akp] prewarm done, caches populated")
        return

    rng = random.Random(1234 + a.repeat)
    n_new = 0
    t0 = time.time()

    with shard.open("a", encoding="utf8") as fh:
        for ci, cfg in enumerate(cfgs):
            impls = [i for i in impls_for(cfg.regime)
                     if a.impl is None or i.name == a.impl]
            pending = [i for i in impls
                       if config_hash(cfg, i.name, dev_info, sha, a.repeat) not in done]
            if not pending:
                continue

            # Random order so no implementation is always measured first.
            order = list(pending)
            rng.shuffle(order)
            for impl in order:
                merged = run_cell(impl, cfg, device, dev_info, a.reps)
                trace = merged.pop("dispatch_trace", None)
                if trace:
                    merged["dispatch_id"] = dispatch.intern(
                        trace, impl.name, dev_info["gpu_name"])
                    merged["n_kernels"] = trace.count("|") + 1
                merged.update({
                    "config_hash": config_hash(cfg, impl.name, dev_info, sha, a.repeat),
                    "repeat": a.repeat,
                    "regime": cfg.regime, "batch": cfg.B, "hq": cfg.Hq,
                    "hkv": cfg.Hkv, "head_dim": cfg.D, "seq_len": cfg.N,
                    "dtype": cfg.dtype, "mode": cfg.mode, "launch": cfg.launch,
                    "cache": cfg.cache, "causal": cfg.causal,
                    "gpu_name": dev_info["gpu_name"],
                    "compute_capability": "{}.{}".format(dev_info["cc_major"],
                                                         dev_info["cc_minor"]),
                    "git_sha": sha, "timestamp": time.time(),
                })
                for k, v in bench.telemetry().items():
                    merged["tel_" + k] = v
                fh.write(json.dumps(merged) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                n_new += 1

                if merged.get("fatal"):
                    raise SystemExit(
                        "[akp] aborting: {} left the CUDA context unusable on "
                        "{} -- {}. {} rows are on disk, the restarted pod "
                        "resumes after them.".format(
                            impl.name, cfg.key(), merged.get("error", ""), n_new))

            print("[akp] {}/{} {} (+{} rows, {:.0f}s)".format(
                ci + 1, len(cfgs), cfg.key(), len(pending), time.time() - t0),
                flush=True)

    ungated = SEEN_CLASSES - GATED
    if ungated:
        # Warn, because analysis treats a class with no verdict as passing.
        print("[akp] WARNING: {} class(es) ran without a correctness "
              "gate, no config of theirs was cheap enough to gate on: "
              "{}".format(len(ungated), sorted("|".join(str(x) for x in c)
                                               for c in ungated)[:5]))
    print("[akp] wrote {} rows to {}".format(n_new, shard))


if __name__ == "__main__":
    sys.exit(main())
