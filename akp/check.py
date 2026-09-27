"""Correctness reference, numerical gate and dispatch probe.

The gate and the probe run once per cell before timing. The tests import the
same functions.
"""

from __future__ import annotations

import torch

from akp.impls import (Cfg, causal_mask, expand_kv, make_inputs,
                       naive_attention)


# --------------------------------------------------------------------------- #
# Layout normalization
# --------------------------------------------------------------------------- #

def to_bhsd(out: torch.Tensor, cfg: Cfg, layout: str) -> torch.Tensor:
    """Bring any implementation's output into the logical (B, Hq, S, D)."""
    if layout == "bhsd":
        return out
    if layout == "bshd":                       # flash-attn
        return out.transpose(1, 2)
    if layout == "bhd":                        # flashinfer decode: (B, Hq, D)
        return out.reshape(cfg.B, cfg.Hq, 1, cfg.D)
    raise ValueError(f"unknown out_layout {layout!r}")


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #

def reference(cfg: Cfg, device, chunk: int = 256) -> torch.Tensor:
    """fp32 reference, chunked over query blocks so memory is O(chunk * N).

    Unchunked it needs 34 GB at B=16, Hq=32, N=4096. It uses the same seed as
    make_inputs, so the implementation and the reference see the same tensors.
    """
    t = make_inputs(cfg, device)
    q = t["q"].float()
    k = expand_kv(t["k"], cfg.gqa).float()
    v = expand_kv(t["v"], cfg.gqa).float()

    n_q, n_k = q.shape[2], k.shape[2]
    out = torch.empty_like(q)
    for i in range(0, n_q, chunk):
        qi = q[:, :, i:i + chunk]
        s = (qi @ k.transpose(-2, -1)) * cfg.scale
        if cfg.causal:
            rows = torch.arange(i, i + qi.shape[2], device=device)
            cols = torch.arange(n_k, device=device)
            # Bottom-right alignment, matching flash-attn >= 2.1.
            keep = cols[None, :] <= (rows[:, None] + (n_k - n_q))
            s = s.masked_fill(~keep, float("-inf"))
        out[:, :, i:i + chunk] = torch.softmax(s, dim=-1) @ v
    return out


def _naive_same_dtype(cfg: Cfg, device) -> torch.Tensor:
    t = make_inputs(cfg, device)
    m = causal_mask(cfg.q_len, cfg.N, device) if cfg.causal else None
    return naive_attention(t["q"], expand_kv(t["k"], cfg.gqa),
                           expand_kv(t["v"], cfg.gqa), m, cfg.scale)


# Floor for the acceptance rule. Without it, a cell where the naive baseline is
# unusually accurate would fail an implementation with normal error for the
# dtype.
TAU = {"fp16": 1e-2, "bf16": 4e-2}


def errors(out: torch.Tensor, ref: torch.Tensor) -> dict:
    o, r = out.float(), ref.float()
    d = (o - r).abs()
    denom = r.abs().clamp_min(1e-6)
    return {
        "max_abs_err": d.max().item(),
        "mean_abs_err": d.mean().item(),
        "max_rel_err": (d / denom).max().item(),
        "cosine_sim": torch.nn.functional.cosine_similarity(
            o.flatten(), r.flatten(), dim=0).item(),
        "has_nonfinite": bool((~torch.isfinite(o)).any().item()),
    }


def accept(err: float, baseline: float, dtype: str) -> dict:
    """E_cand <= max(2 * E_naive_same_dtype, tau[dtype]).

    Both conditions are recorded to show which one decided the cell.
    """
    tau = TAU[dtype]
    return {"pass_relative": bool(err <= 2 * baseline),
            "pass_tolerance": bool(err <= tau),
            "tau_dtype": tau,
            "correctness_pass": bool(err <= max(2 * baseline, tau))}


def _grads(cfg: Cfg, device, dtype):
    """Reference dQ/dK/dV under the same upstream gradient, in `dtype`."""
    t = make_inputs(cfg, device, requires_grad=True)
    cast = (lambda x: x.float()) if dtype == "fp32" else (lambda x: x)
    m = causal_mask(cfg.q_len, cfg.N, device) if cfg.causal else None
    o = naive_attention(cast(t["q"]), cast(expand_kv(t["k"], cfg.gqa)),
                        cast(expand_kv(t["v"], cfg.gqa)), m, cfg.scale)
    o.backward(cast(t["do"]))
    return t["q"].grad, t["k"].grad, t["v"].grad


def gate(cfg: Cfg, device, out: torch.Tensor, layout: str, fn=None) -> dict:
    """Compare against the reference and decide pass/fail.

    Gradients are checked separately, because a backward pass can be wrong in
    dK alone while the forward output is correct.
    """
    ref = reference(cfg, device)
    e = errors(to_bhsd(out, cfg, layout), ref)
    baseline = errors(_naive_same_dtype(cfg, device), ref)["max_abs_err"]
    e["baseline_max_abs_err"] = baseline
    e.update(accept(e["max_abs_err"], baseline, cfg.dtype))
    if e["has_nonfinite"]:
        e["correctness_pass"] = False
        e["failed_tensor"] = "out(nonfinite)"

    got = getattr(fn, "grad_tensors", None)
    if cfg.mode == "fwd_bwd" and got is not None:
        ref_g = _grads(cfg, device, "fp32")
        base_g = _grads(cfg, device, cfg.dtype)
        for name, g, rg, bg in zip("qkv", got, ref_g, base_g):
            if g.grad is None:
                continue
            gi = to_bhsd(g.grad, cfg, layout)
            ei = errors(gi, to_bhsd(rg, cfg, "bhsd"))["max_abs_err"]
            bi = errors(to_bhsd(bg, cfg, "bhsd"), to_bhsd(rg, cfg, "bhsd"))["max_abs_err"]
            e["d" + name + "_max_abs_err"] = ei
            if not accept(ei, bi, cfg.dtype)["correctness_pass"]:
                e["correctness_pass"] = False
                e["failed_tensor"] = "d" + name
    return e


# --------------------------------------------------------------------------- #
# Dispatch probe
# --------------------------------------------------------------------------- #

def dispatch_probe(fn) -> str:
    """CUDA kernels one call actually launches.

    Returns raw names and checks nothing. Classification happens in
    analysis.py, so fixing a bad pattern only needs a re-analysis.
    """
    from torch.profiler import ProfilerActivity, profile

    from torch.autograd import DeviceType

    fn()
    torch.cuda.synchronize()

    # Select events by device type. A short fused kernel can report zero
    # self-time, so filtering by time would drop it.
    # acc_events=True stops the profiler clearing events each cycle.
    # CUPTI sometimes returns no events, most often on the first profile in a
    # process. Retry, and return a distinct marker if it never works.
    for _ in range(4):
        try:
            try:
                ctx = profile(activities=[ProfilerActivity.CUDA], acc_events=True)
            except TypeError:
                ctx = profile(activities=[ProfilerActivity.CUDA])
            with ctx as prof:
                fn()
                fn()
                torch.cuda.synchronize()
        except Exception as exc:            # profiler unavailable in some containers
            return "<probe-failed: " + type(exc).__name__ + ">"
        names = {e.name for e in prof.events() if e.device_type == DeviceType.CUDA}
        if names:
            return "|".join(sorted(names))
        torch.cuda.synchronize()
    return "<no-cuda-kernels>"
