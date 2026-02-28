"""Per-GPU preflight. The sweep refuses to start unless this reports PASS.

    python -m akp.preflight --require-gpu A100-SXM4-80GB

Every check either asks the driver a question or drives the same code the sweep
drives -- impls.build, check.gate, check.dispatch_probe, bench.block_bench.
Nothing here is a second implementation of a sweep component: a reimplemented
check can pass while the real one is broken, which is the failure mode a
preflight exists to prevent.

Exit status is 0 on PASS and 1 on FAIL, so a Job can gate on it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time

import torch

from akp import bench, check, run
from akp.impls import IMPLS, Cfg, naive_peak_bytes

# Erratic throttles, as in analysis.usable: hardware and thermal slowdowns.
# SwPowerCap is the steady state of a loaded datacenter part, not a fault.
ERRATIC_THROTTLE = 0xF8

# Shapes small enough to be quick and large enough to be real: N >= 128 because
# the Triton tutorial autotunes BLOCK_M up to 128 and writes past a shorter row.
PREFILL = Cfg("prefill", B=1, Hq=8, Hkv=8, D=64, N=256)
PREFILL_BWD = Cfg("prefill", B=1, Hq=8, Hkv=8, D=64, N=256, mode="fwd_bwd")
DECODE = Cfg("decode", B=2, Hq=8, Hkv=2, D=64, N=512, causal=False)


def _smi(fields: str) -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        return [x.strip() for x in out.stdout.strip().split("\n")[0].split(",")]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Gate 0: authorization and resource scope
# --------------------------------------------------------------------------- #

def check_identity(ctx):
    d = ctx["dev"]
    want = ctx["require_gpu"]
    got = d["gpu_name"]
    detail = (f"{got} cc{d['cc_major']}.{d['cc_minor']} "
              f"{d['total_memory_gb']}GB {d['sm_count']}SM")
    if want and want not in got:
        return "FAIL", f"requested {want!r}, got {detail}"
    if torch.cuda.device_count() != 1:
        return "WARN", f"{torch.cuda.device_count()} visible GPUs; using 0. {detail}"
    return "PASS", detail


def check_mig(ctx):
    v = _smi("mig.mode.current")
    if not v or v[0] in ("[N/A]", "N/A", ""):
        return "PASS", "not MIG-capable or not reported"
    if v[0].lower() != "disabled":
        return "FAIL", f"MIG {v[0]}: a slice is not the device we claim to measure"
    return "PASS", "disabled"


def check_exclusive(ctx):
    """Someone else's kernels on our SMs would land in our medians.

    In a container nvidia-smi usually cannot see other namespaces' PIDs, so the
    load-bearing signal is device memory in use that our own process did not
    reserve.
    """
    v = _smi("memory.used")
    if not v:
        return "SKIP", "nvidia-smi unavailable"
    used_mb = float(v[0])
    ours_mb = torch.cuda.memory_reserved() / 2**20
    foreign = used_mb - ours_mb
    apps = ""
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=15).stdout.strip()
    except Exception:
        pass
    detail = f"{used_mb:.0f}MB used, {ours_mb:.0f}MB ours" + (f" | {apps}" if apps else "")
    if foreign > 1024:
        return "FAIL", f"{foreign:.0f}MB in use by another process. {detail}"
    return "PASS", detail


def check_idle(ctx):
    t = bench.telemetry()
    if not t:
        return "SKIP", "no telemetry"
    bits = int(str(t.get("throttle", "0x0")), 16)
    detail = (f"{t['sm_clock_mhz']:.0f}/{t['max_sm_clock_mhz']:.0f}MHz "
              f"{t['temp_c']:.0f}C {t['power_w']:.0f}W throttle={t['throttle']}")
    if bits & ERRATIC_THROTTLE:
        return "FAIL", f"throttled before we started: {detail}"
    return "PASS", detail


# --------------------------------------------------------------------------- #
# Gate 1: environment manifest
# --------------------------------------------------------------------------- #

def check_manifest(ctx):
    m = ctx["manifest"]
    missing = [k for k in ("driver", "git_sha", "measured_peak_bw_gbs",
                           "event_overhead_us") if not m.get(k)]
    v = m["versions"]
    unresolved = [k for k, val in v.items() if val is None]
    detail = (f"driver {m['driver']} | sha {m['git_sha'][:8]} | "
              f"bw {m['measured_peak_bw_gbs']}GB/s | "
              f"event {m['event_overhead_us']}us | "
              + " ".join(f"{k}={val}" for k, val in v.items() if val))
    if missing:
        return "FAIL", f"manifest missing {missing}. {detail}"
    if unresolved:
        return "WARN", f"unresolved: {unresolved}. {detail}"
    return "PASS", detail


def check_power_limit(ctx):
    v = _smi("power.limit,power.max_limit,enforced.power.limit")
    if not v:
        return "SKIP", "not reported"
    return "PASS", f"limit {v[0]}W of max {v[1]}W (enforced {v[2]}W)"


def check_ecc(ctx):
    v = _smi("ecc.errors.uncorrected.volatile.total")
    if not v or v[0] in ("[N/A]", "N/A", ""):
        return "SKIP", "ECC not reported"
    try:
        n = int(v[0])
    except ValueError:
        return "SKIP", v[0]
    return ("FAIL" if n else "PASS"), f"{n} uncorrected volatile"


# --------------------------------------------------------------------------- #
# Gate 2/3: capability and per-implementation smoke
# --------------------------------------------------------------------------- #

def check_capability(ctx):
    d = ctx["dev"]
    cc = d["cc_major"] * 10 + d["cc_minor"]
    bf16 = torch.cuda.is_bf16_supported()
    fp8 = cc >= 89
    if not bf16:
        return "FAIL", f"bf16 unsupported on cc{cc}; both dtype axes need it"
    return "PASS", (f"cc{cc} bf16={bf16} fp8={fp8} "
                    f"cudnn={torch.backends.cudnn.version()}")


def _impl_smoke(name, impl, cfg, device, dev):
    """Build, run, gate numerics and probe dispatch -- run_cell's own order.

    The gate goes before the probe because that is what the sweep does, and the
    order turns out to matter: sixteen probes fired back to back finish faster
    than CUPTI flushes its buffers, and most come back empty. In the sweep an
    fp32 reference and a full timing block sit between consecutive probes, and
    coverage there is 1450 of 1450 rows on this same device.
    """
    if not impl.supports(cfg, dev) or not run.impl_applies(name, cfg):
        return "SKIP", "unsupported at the probe shape"
    built = None
    try:
        built = impl.build(cfg, device)
        out = built.fn()
        torch.cuda.synchronize()
        g = check.gate(cfg, device, out, built.meta.get("out_layout", "bhsd"),
                       fn=built.fn)
        if not g["correctness_pass"]:
            return "FAIL", (f"gate failed on {g.get('failed_tensor', 'out')}: "
                            f"err {g['max_abs_err']:.3e} vs "
                            f"2x naive {2 * g['baseline_max_abs_err']:.3e}")
        trace = check.dispatch_probe(built.fn)
        if trace.startswith("<"):
            # Not FAIL: the kernel ran and was numerically right, so this is
            # the profiler being unavailable rather than the backend being
            # wrong. run.py records the same string and analysis counts it.
            return "WARN", f"correct, but no dispatch trace: {trace}"
        return "PASS", (f"err {g['max_abs_err']:.2e} | "
                        f"{trace.count('|') + 1} kernel(s)")
    except Exception as exc:
        return "FAIL", f"{type(exc).__name__}: {exc}"[:160]
    finally:
        del built
        # An illegal memory access poisons the context, and then empty_cache
        # raises too -- from a finally, which discards the return above and
        # propagates instead. That loses the one thing worth knowing: which
        # implementation faulted.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def check_impls(ctx):
    """Every implementation imports, runs, dispatches and is numerically right.

    Reported per implementation, because "one backend is broken here" is a
    result worth seeing before the sweep rather than a hole discovered after.
    """
    device, dev = ctx["device"], ctx["dev"]
    rows, bad, blind = [], [], []
    poisoned = False
    for name, impl in IMPLS.items():
        if poisoned:
            rows.append(f"    {name:22s} SKIP  not probed, CUDA context already lost")
            continue
        cfg = DECODE if impl.regime == "decode" else PREFILL
        st, detail = _impl_smoke(name, impl, cfg, device, dev)
        rows.append(f"    {name:22s} {st:5s} {detail}")
        if st == "FAIL":
            bad.append(name)
            # Everything after an illegal access fails identically, which
            # buries the one implementation that actually broke.
            if run._is_sticky(detail):
                poisoned = True
                rows.append(f"    -> {name} poisoned the context; "
                            "later implementations were not probed")
        elif st == "WARN":
            blind.append(name)
    ctx["impl_report"] = rows
    traced = sum(1 for r in rows if " PASS " in r)
    if bad:
        return "FAIL", f"{len(bad)} implementation(s) failed: {bad}"
    if blind and traced == 0:
        # Dispatch verification is a headline result of the study, so a host
        # where the profiler never returns anything cannot produce it.
        return "FAIL", f"no implementation produced a dispatch trace: {blind}"
    if blind:
        return "WARN", f"{len(blind)} correct but unprofiled: {blind}"
    return "PASS", f"{len(rows)} implementations probed"


def check_backward(ctx):
    """A backward path can be wrong in dK alone while the forward is perfect."""
    device, dev = ctx["device"], ctx["dev"]
    bad = []
    for name, impl in IMPLS.items():
        if impl.regime != "prefill":
            continue
        st, detail = _impl_smoke(name, impl, PREFILL_BWD, device, dev)
        if st == "FAIL":
            bad.append(f"{name} ({detail})")
    if bad:
        return "FAIL", "; ".join(bad)[:200]
    return "PASS", "dQ/dK/dV within gate at the probe shape"


# --------------------------------------------------------------------------- #
# Gate 9/14: predictors and the timer itself
# --------------------------------------------------------------------------- #

def check_oom_predictor(ctx):
    """The predictor must bound the real allocation, or it is not a guard.

    Probed at N=2048, not at the small shapes above: the caching allocator also
    holds a fixed cuBLAS workspace of order 10-20 MB, which swamps a 256-token
    score matrix and is noise against the 68 GB cell the guard exists to refuse.
    """
    device = ctx["device"]
    cfg = Cfg("prefill", B=1, Hq=8, Hkv=8, D=64, N=2048)
    impl = IMPLS["P0-naive"]
    built = impl.build(cfg, device)
    try:
        stats = bench.peak_memory_mb(built.fn, device)
    finally:
        del built
        torch.cuda.empty_cache()
    predicted_mb = naive_peak_bytes(cfg) / 2**20
    actual_mb = stats["peak_allocated_mb"]
    detail = (f"predicted {predicted_mb:.0f}MB, actual {actual_mb:.0f}MB "
              f"at {cfg.key()}")
    if predicted_mb < actual_mb:
        return "FAIL", f"predictor under-estimates, so it cannot gate: {detail}"
    return "PASS", detail


def check_timer(ctx):
    """Doubling the work must double the measurement.

    This is the one check that catches a harness measuring nothing: an
    unsynchronized timer, a no-op region, or events created inside the timed
    block all return a number that looks plausible and does not scale.
    """
    device = ctx["device"]
    a = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    b = torch.randn(2048, 2048, device=device, dtype=torch.float16)

    def work(n):
        def f():
            for _ in range(n):
                a @ b
        return f

    # 4x against 2x, not 2x against 1x: both arms sit well above any fixed
    # per-block cost, so the ratio tests linearity rather than the size of the
    # intercept. Interleaved for the same reason the sweep interleaves
    # implementations (6.2) -- measured back to back, clock drift lands on
    # whichever ran second and reads as non-linearity.
    # Warm up for a fixed duration, not a fixed count. An A100 idles around
    # 1155 of 1410 MHz and ramps under load; 200 iterations is ~40 ms, which
    # is not enough for that to settle. The bias always falls on the shorter
    # arm -- it runs more of itself at a lower clock -- so an unsettled clock
    # reads as sub-linear scaling and fails a perfectly good host. Two A100
    # nodes scored 1.69 this way while an L40, already pinned at its max
    # clock, scored 1.96.
    settle = time.time() + 1.0
    while time.time() < settle:
        work(8)()
    torch.cuda.synchronize()

    two, four = [], []
    for _ in range(3):
        two.append(bench.block_bench(work(2), device, reps=10, warmup=100)["median_us"])
        four.append(bench.block_bench(work(4), device, reps=10, warmup=100)["median_us"])
    t2, t4 = sorted(two)[1], sorted(four)[1]
    ratio = t4 / t2
    detail = f"2x={t2:.1f}us 4x={t4:.1f}us ratio={ratio:.2f}"
    # 1.75 rather than 1.8: a warm A100 measured 1.81, and a gate that a good
    # host clears by 0.01 fails honest hosts on noise. The failure this has to
    # catch -- a timer measuring nothing, an unsynchronized region, events made
    # inside the timed block -- lands nowhere near 1.75, it lands near 1.0.
    if not 1.75 <= ratio <= 2.25:
        return "FAIL", f"work does not scale linearly with time: {detail}"

    overhead = ctx["manifest"]["event_overhead_us"]
    if overhead > 50:
        return "FAIL", f"event overhead {overhead}us is too large to time with"
    return "PASS", f"{detail} | event overhead {overhead}us"


def check_writer(ctx):
    """A row must survive the write path and come back with its keys."""
    required = ("implementation", "status", "median_us", "gpu_name",
                "config_hash", "git_sha")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe.jsonl")
        row = {k: 1 for k in required}
        with open(path, "a", encoding="utf8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        back = json.loads(open(path, encoding="utf8").read().strip())
    missing = [k for k in required if k not in back]
    if missing:
        return "FAIL", f"round-trip lost {missing}"
    return "PASS", f"fsync round-trip kept {len(required)} required keys"


CHECKS = [
    ("GPU identity", check_identity),
    ("MIG status", check_mig),
    ("Exclusive access", check_exclusive),
    ("Idle / throttle", check_idle),
    ("Environment manifest", check_manifest),
    ("Power limit", check_power_limit),
    ("ECC", check_ecc),
    ("Capability probe", check_capability),
    ("Implementation smoke", check_impls),
    ("Backward smoke", check_backward),
    ("OOM predictor", check_oom_predictor),
    ("Timer validation", check_timer),
    ("Result writer", check_writer),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-gpu", default=None)
    ap.add_argument("--out", default=None, help="write the report as JSON here")
    a = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("preflight: no CUDA device")

    device = torch.device("cuda")
    ctx = {"device": device, "require_gpu": a.require_gpu,
           "dev": bench.device_info(device), "impl_report": []}
    ctx["manifest"] = run.manifest(device)

    print(f"preflight: {ctx['dev']['gpu_name']} | image "
          f"{os.environ.get('AKP_IMAGE_DIGEST', 'undeclared')}\n")

    results, failed = [], []
    for label, fn in CHECKS:
        try:
            status, detail = fn(ctx)
        except Exception as exc:
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"[:160]
        results.append({"check": label, "status": status, "detail": detail})
        print(f"  {label:22s} {status:5s} {detail}")
        if label == "Implementation smoke":
            for line in ctx["impl_report"]:
                print(line)
        if status == "FAIL":
            failed.append(label)

    overall = "FAIL" if failed else "PASS"
    print(f"\nOverall: {overall}" + (f" -- {failed}" if failed else ""))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"gpu": ctx["dev"], "manifest": ctx["manifest"],
                   "checks": results, "overall": overall},
                  open(a.out, "w", encoding="utf8"), indent=2, default=str)

    raise SystemExit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
