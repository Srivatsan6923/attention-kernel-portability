"""Timing, device identification, and per-cell telemetry.

p5/p95 are descriptive spread only; the 5th percentile of 30 samples is too
noisy to infer from. Confidence intervals come from the bootstrap in analysis.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess

import torch
import triton.testing as tt


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #

def device_info(index: int = 0) -> dict:
    p = torch.cuda.get_device_properties(index)
    # Theoretical peak: DDR transfers twice per clock.  memory_clock_rate is kHz.
    peak_bw = p.memory_clock_rate * 1e3 * 2 * (p.memory_bus_width / 8) / 1e9
    return {
        "gpu_name": p.name,
        "gpu_uuid": str(getattr(p, "uuid", "")),
        "cc_major": p.major,
        "cc_minor": p.minor,
        "sm_count": p.multi_processor_count,
        "total_memory_gb": round(p.total_memory / 1e9, 2),
        "l2_bytes": int(p.L2_cache_size),
        "theoretical_bw_gbs": round(peak_bw, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


_SMI_FIELDS = ("clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
               "clocks_throttle_reasons.active")


def telemetry() -> dict:
    """SM clock, temperature, power, throttle reasons, so we can tell
    afterwards whether an inversion was really a thermally limited host."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_SMI_FIELDS}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().split("\n")[0]
        sm_clk, max_clk, temp, power, throttle = [x.strip() for x in out.split(",")]
        return {"sm_clock_mhz": float(sm_clk), "max_sm_clock_mhz": float(max_clk),
                "temp_c": float(temp), "power_w": float(power),
                "throttle": throttle}
    except Exception:
        return {}


def measured_peak_bw_gbs(device, mb: int = 512, reps: int = 20) -> float:
    """Achievable bandwidth from a large copy, reported next to the
    published peak so bw_util is measured rather than assumed."""
    n = mb * 1024 * 1024 // 2
    a = torch.empty(n, dtype=torch.float16, device=device)
    b = torch.empty_like(a)
    ms = tt.do_bench(lambda: b.copy_(a), warmup=10, rep=reps, return_mode="median")
    return round(2 * a.numel() * 2 / (ms * 1e-3) / 1e9, 1)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

def _flush_buffer(device):
    """Buffer larger than L2, zeroed to evict it."""
    n = torch.cuda.get_device_properties(device).L2_cache_size
    return torch.empty(int(n * 2), dtype=torch.int8, device=device)


def event_overhead_us(device, reps: int = 50) -> float:
    """Cost of one cudaEventRecord pair around an empty region.

    Recorded per device: it is how much per-iteration event timing would have
    inflated a short decode kernel, and why block_bench times K at a time.
    """
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(reps):
        s.record(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return round(statistics.median(ts), 3)


def block_bench(fn, device, reps: int = 30, warmup: int = 25,
                target_us: float = 200.0, flush_l2: bool = False) -> dict:
    """K iterations per event pair, median over `reps` blocks.

    K is sized so a block exceeds target_us, keeping the event pair under 1% of
    the measurement even for a 5-15 us decode kernel.

    flush_l2 is the cache condition and is reported, not assumed. Flushing
    before every call measures cold cache; back-to-back serving is warm, and
    reporting flushed numbers as serving latency overstates it.
    """
    flush = _flush_buffer(device) if flush_l2 else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10):
        fn()
    e.record()
    torch.cuda.synchronize()
    per_iter_us = max(s.elapsed_time(e) * 1e3 / 10, 1e-3)
    k = max(1, min(1000, int(target_us / per_iter_us)))

    times = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        s.record()
        for _ in range(k):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3 / k)

    times.sort()
    return {"median_us": statistics.median(times),
            "p5_us": times[max(0, int(0.05 * len(times)) - 1)],
            "p95_us": times[min(len(times) - 1, int(0.95 * len(times)))],
            "all_us": times, "timer": "block_bench", "inner_k": k,
            "reps": reps, "l2_flushed": flush is not None}


def measure(fn, cfg, device, reps: int = 30) -> dict:
    """Dispatch to the right timer for this cell."""
    if cfg.launch == "cudagraph":
        med, p5, p95 = (x * 1e3 for x in
                        tt.do_bench_cudagraph(fn, rep=reps,
                                              quantiles=[0.5, 0.05, 0.95]))
        return {"median_us": med, "p5_us": p5, "p95_us": p95,
                "timer": "do_bench_cudagraph", "reps": reps,
                "l2_flushed": False}
    return block_bench(fn, device, reps=reps, flush_l2=(cfg.cache == "cold"))


def peak_memory_mb(fn, device) -> dict:
    """Peak allocation for one warm call, measured outside the timed region."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize()
    return {"peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20}
