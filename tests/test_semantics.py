"""Semantic checks for the bugs that stay silent and still produce
plausible numbers.

A causal-alignment or GQA-mapping error does not crash and does not look wrong
in a plot, it just invalidates every number. Some tests below assert that two
APIs disagree, because a test that passes trivially would not catch the trap.
"""

from __future__ import annotations

import pytest
import torch

from akp import check
from akp.impls import (IMPLS, Cfg, causal_mask, expand_kv, has, make_inputs,
                       naive_attention)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA required")

DEV = torch.device("cuda", 0) if torch.cuda.is_available() else None


def dev_info():
    from akp import bench
    return bench.device_info(DEV)


# Causal alignment.

@pytest.mark.skipif(not has("flash_attn"), reason="flash-attn not installed")
def test_fa_and_sdpa_causal_disagree_when_qlen_ne_kvlen():
    """FA >= 2.1 is bottom-right aligned, SDPA is_causal is top-left.

    At q_len=2, kv_len=5 they compute different math. If they ever agree, a
    library changed convention and every decode call needs re-deriving.
    """
    import torch.nn.functional as F
    from flash_attn import flash_attn_func

    B, H, Dh, q_len, kv_len = 1, 2, 64, 2, 5
    torch.manual_seed(0)
    q = torch.randn(B, q_len, H, Dh, device=DEV, dtype=torch.float16)
    k = torch.randn(B, kv_len, H, Dh, device=DEV, dtype=torch.float16)
    v = torch.randn(B, kv_len, H, Dh, device=DEV, dtype=torch.float16)

    fa = flash_attn_func(q, k, v, causal=True).transpose(1, 2)
    sd = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)

    assert not torch.allclose(fa.float(), sd.float(), atol=1e-2), (
        "flash-attn and SDPA agreed on causal at q_len != kv_len -- a "
        "convention changed upstream; re-derive the decode call for every impl")


def test_decode_query_attends_to_every_cached_key():
    """Decode is one query over all N keys.

    Top-left causal at q_len=1 keeps exactly one key; asserting the reference
    differs from that degenerate case is what pins the convention.
    """
    cfg = Cfg("decode", B=1, Hq=2, Hkv=2, D=64, N=8, causal=False)
    ref = check.reference(cfg, DEV)

    t = make_inputs(cfg, DEV)
    only_first = naive_attention(t["q"], t["k"][:, :, :1], t["v"][:, :, :1],
                                 None, cfg.scale)
    assert not torch.allclose(ref, only_first.float(), atol=1e-3)

    dense = naive_attention(t["q"], t["k"], t["v"], None, cfg.scale)
    assert torch.allclose(ref, dense.float(), atol=1e-2)


def test_prefill_reference_is_lower_triangular():
    """At q_len == kv_len both conventions coincide; verify against a mask."""
    cfg = Cfg("prefill", B=1, Hq=1, Hkv=1, D=32, N=8)
    t = make_inputs(cfg, DEV)
    q, k, v = (x.float() for x in (t["q"], t["k"], t["v"]))
    s = (q @ k.transpose(-2, -1)) * cfg.scale
    mask = torch.ones(8, 8, dtype=torch.bool, device=DEV).tril()
    manual = torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ v
    assert torch.allclose(check.reference(cfg, DEV), manual, atol=1e-4)


# --------------------------------------------------------------------------- #
# GQA head mapping
# --------------------------------------------------------------------------- #

def test_gqa_uses_repeat_interleave_not_repeat():
    """Query head i attends to kv head i // gqa.

    repeat would give i % Hkv, so check the two differ and we took interleave.
    """
    k = torch.arange(8, device=DEV).reshape(1, 4, 2, 1).float()   # Hkv=4
    interleaved = expand_kv(k, 2)
    assert interleaved.shape[1] == 8
    # head 0 and 1 of the expansion must both come from kv head 0
    assert torch.equal(interleaved[:, 0], k[:, 0])
    assert torch.equal(interleaved[:, 1], k[:, 0])
    assert torch.equal(interleaved[:, 2], k[:, 1])
    assert not torch.equal(interleaved, k.repeat(1, 2, 1, 1))


@pytest.mark.skipif(not has("flash_attn"), reason="flash-attn not installed")
def test_flash_attn_gqa_matches_repeat_interleave():
    from flash_attn import flash_attn_func
    B, Hq, Hkv, N, Dh = 1, 8, 2, 16, 64
    torch.manual_seed(0)
    q = torch.randn(B, N, Hq, Dh, device=DEV, dtype=torch.float16)
    k = torch.randn(B, N, Hkv, Dh, device=DEV, dtype=torch.float16)
    v = torch.randn(B, N, Hkv, Dh, device=DEV, dtype=torch.float16)

    fa = flash_attn_func(q, k, v, causal=True).transpose(1, 2).float()
    ke = expand_kv(k.transpose(1, 2), Hq // Hkv)
    ve = expand_kv(v.transpose(1, 2), Hq // Hkv)
    ref = naive_attention(q.transpose(1, 2).float(), ke.float(), ve.float(),
                          causal_mask(N, N, DEV), Dh ** -0.5)
    assert torch.allclose(fa, ref, atol=2e-2), "GQA head mapping mismatch"


# --------------------------------------------------------------------------- #
# Registry-wide invariants
# --------------------------------------------------------------------------- #

SMALL = [
    Cfg("prefill", B=2, Hq=4, Hkv=4, D=64, N=128),
    Cfg("prefill", B=2, Hq=4, Hkv=1, D=64, N=128),       # GQA
    Cfg("decode", B=2, Hq=4, Hkv=1, D=64, N=128, causal=False),
]


def build_or_skip(impl, cfg):
    """Build and run once, returning (built, out) or None.

    SDPA reports missing coverage by raising "No available kernel" from the call
    rather than from the build, and a sweep records that as UNSUPPORTED. Tests
    treat it the same way instead of failing on it.
    """
    try:
        built = impl.build(cfg, DEV)
        return built, built.fn()
    except (RuntimeError, NotImplementedError) as exc:
        msg = str(exc).lower()
        if "no available kernel" in msg or "not supported" in msg:
            return None
        raise


@pytest.mark.parametrize("cfg", SMALL, ids=lambda c: c.key())
def test_every_impl_passes_the_gate(cfg):
    info = dev_info()
    ran = 0
    for impl in IMPLS.values():
        if impl.regime != cfg.regime or not impl.supports(cfg, info):
            continue
        got = build_or_skip(impl, cfg)
        if got is None:
            continue
        built, out = got
        res = check.gate(cfg, DEV, out, built.meta.get("out_layout", "bhsd"),
                         fn=built.fn)
        assert res["correctness_pass"], (
            f"{impl.name} failed the 2x-naive gate: "
            f"{res['max_abs_err']:.3e} vs baseline {res['baseline_max_abs_err']:.3e}")
        ran += 1
    assert ran >= 2, "no implementations were exercised"


@pytest.mark.parametrize("cfg", SMALL, ids=lambda c: c.key())
def test_run_allocates_nothing_beyond_its_output(cfg):
    """A transpose or contiguous() in the hot path measures a copy.

    The slack is a few outputs worth, so a hidden K/V materialization still
    trips it.
    """
    info = dev_info()
    out_bytes = cfg.B * cfg.Hq * cfg.q_len * cfg.D * 2
    for impl in IMPLS.values():
        if impl.regime != cfg.regime or not impl.supports(cfg, info):
            continue
        if impl.name.startswith(("P0", "P1", "D0", "D1")):
            continue          # score-matrix impls allocate by design
        got = build_or_skip(impl, cfg)
        if got is None:
            continue
        built = got[0]
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(DEV)
        built.fn()
        torch.cuda.synchronize()
        grew = torch.cuda.memory_allocated(DEV) - before
        assert grew <= 4 * out_bytes + 4096, (
            f"{impl.name} allocated {grew}B inside run(); expected <= one output")


@pytest.mark.skipif(not has("flash_attn"), reason="flash-attn not installed")
def test_kv_cache_is_not_mutated_by_timing():
    """flash_attn_with_kvcache appends in place when k/v are passed."""
    cfg = Cfg("decode", B=2, Hq=4, Hkv=1, D=64, N=128, causal=False)
    built = IMPLS["D3-fa-kvcache"].build(cfg, DEV)
    kc, vc = built.meta["_cache_tensors"]
    before = (kc.clone(), vc.clone())
    for _ in range(100):
        built.fn()
    torch.cuda.synchronize()
    assert torch.equal(kc, before[0]) and torch.equal(vc, before[1]), (
        "the KV cache changed while being timed -- k/v must be None")


def test_softmax_scale_is_one_over_sqrt_d():
    for d in (64, 128):
        cfg = Cfg("prefill", B=1, Hq=1, Hkv=1, D=d, N=8)
        assert abs(cfg.scale - d ** -0.5) < 1e-12


def test_layout_normalization_round_trips():
    cfg = Cfg("prefill", B=2, Hq=4, Hkv=4, D=64, N=16)
    x = torch.randn(2, 4, 16, 64, device=DEV)
    assert torch.equal(check.to_bhsd(x, cfg, "bhsd"), x)
    assert torch.equal(check.to_bhsd(x.transpose(1, 2), cfg, "bshd"), x)

    dcfg = Cfg("decode", B=2, Hq=4, Hkv=1, D=64, N=16, causal=False)
    y = torch.randn(2, 4, 64, device=DEV)
    assert check.to_bhsd(y, dcfg, "bhd").shape == (2, 4, 1, 64)


def test_oom_prediction_is_analytic_and_large():
    """The B=16, N=4096 cell must be predicted, never attempted."""
    from akp.impls import naive_peak_bytes
    big = Cfg("prefill", B=16, Hq=32, Hkv=32, D=128, N=4096)
    # Over the 0.85 * capacity threshold run_cell skips on, for an 80 GB A100.
    assert naive_peak_bytes(big) > 0.85 * 80e9
    small = Cfg("prefill", B=1, Hq=32, Hkv=32, D=128, N=256)
    assert naive_peak_bytes(small) / 1e9 < 1


def test_reference_is_independently_correct():
    """P0 both computes the reference and competes in the ranking, so
    check it once against NumPy in fp64, which shares no code with it."""
    import numpy as np
    cfg = Cfg("prefill", B=1, Hq=2, Hkv=2, D=16, N=6)
    t = make_inputs(cfg, DEV)
    q, k, v = (x.double().cpu().numpy() for x in (t["q"], t["k"], t["v"]))
    s = np.einsum("bhqd,bhkd->bhqk", q, k) * cfg.scale
    m = np.tril(np.ones((6, 6), dtype=bool))
    s = np.where(m, s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s); p /= p.sum(-1, keepdims=True)
    ref_np = np.einsum("bhqk,bhkd->bhqd", p, v)
    got = check.reference(cfg, DEV).double().cpu().numpy()
    assert np.abs(got - ref_np).max() < 1e-2


# Adversarial input families.
#
# Standard-normal inputs exercise the easy part of softmax. The trap specific
# to attention is logit magnitude: without max-subtraction the exponentials
# overflow and the kernel returns NaN, and a kernel that only ever sees
# N(0, 1) inputs passes every test while being wrong in production.

FAMILIES = {
    # q.k over D=64 dims has std ~8; scaling both by 16 puts pre-softmax
    # logits near 256, where exp() overflows fp32 unless the max is removed.
    "large-logits": {"q": 16.0, "k": 16.0},
    # Every key equally weighted. Degenerate, and a plausible place for a
    # tie-breaking or normalisation bug to surface.
    "zero-logits": {"q": 0.0, "k": 0.0},
    # Denormal territory in fp16.
    "tiny-values": {"q": 1e-3, "k": 1e-3, "v": 1e-3},
}

ADVERSARIAL = [
    Cfg("prefill", B=1, Hq=4, Hkv=4, D=64, N=128),
    Cfg("decode", B=2, Hq=4, Hkv=1, D=64, N=128, causal=False),
]


def _scaled_inputs(scales):
    """make_inputs, with named tensors rescaled.

    Patched into both akp.impls and akp.check so the implementation and the
    reference it is graded against see the same tensors; check.py binds
    make_inputs at import, so patching one module is not enough.
    """
    def wrapper(cfg, device, seed=0, requires_grad=False):
        t = make_inputs(cfg, device, seed=seed, requires_grad=False)
        return {name: (x * scales[name] if name in scales else x)
                for name, x in t.items()}
    return wrapper


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("cfg", ADVERSARIAL, ids=lambda c: c.key())
def test_impls_stay_finite_and_correct_on_adversarial_inputs(family, cfg,
                                                             monkeypatch):
    wrapper = _scaled_inputs(FAMILIES[family])
    monkeypatch.setattr("akp.impls.make_inputs", wrapper)
    monkeypatch.setattr("akp.check.make_inputs", wrapper)

    info = dev_info()
    ran = 0
    for impl in IMPLS.values():
        if impl.regime != cfg.regime or not impl.supports(cfg, info):
            continue
        got = build_or_skip(impl, cfg)
        if got is None:
            continue
        built, out = got
        assert torch.isfinite(out.float()).all(), (
            f"{impl.name} returned non-finite values on {family!r} inputs; "
            "the softmax is likely missing its max subtraction")
        res = check.gate(cfg, DEV, out, built.meta.get("out_layout", "bhsd"),
                         fn=built.fn)
        assert res["correctness_pass"], (
            f"{impl.name} failed the gate on {family!r}: "
            f"{res['max_abs_err']:.3e} vs baseline "
            f"{res['baseline_max_abs_err']:.3e}")
        ran += 1
    assert ran >= 2, f"no implementations were exercised for {family!r}"


# Decode reads the whole cache, asserted against the implementations.

@pytest.mark.parametrize("where", ["first", "last"])
def test_every_decode_impl_reads_the_named_cache_position(monkeypatch, where):
    """Every decode path must depend on the same allowed keys.

    test_decode_query_attends_to_every_cached_key pins the *reference*. This
    pins the implementations, which is where the trap actually bites: D2-sdpa
    forwards cfg.causal into SDPA, and SDPA's top-left is_causal at q_len=1
    keeps only key 0. A path with that bug is fast and silently wrong.

    V is zeroed except at one cache position, so the output is that position's
    attention weight. Reading the whole cache gives a non-zero output at either
    end; keeping only key 0 gives exactly zero when the live position is last.
    """
    from akp import impls as impls_mod

    cfg = Cfg("decode", B=1, Hq=2, Hkv=2, D=64, N=16, causal=False)
    j = 0 if where == "first" else cfg.N - 1
    real = impls_mod.make_inputs

    def one_live_position(c, device, seed=0, requires_grad=False):
        t = real(c, device, seed=seed, requires_grad=requires_grad)
        v = torch.zeros_like(t["v"])
        v[:, :, j, :] = 1.0
        t["v"] = v
        return t

    monkeypatch.setattr(impls_mod, "make_inputs", one_live_position)

    info = dev_info()
    checked = []
    for impl in IMPLS.values():
        if impl.regime != "decode" or not impl.supports(cfg, info):
            continue
        got = build_or_skip(impl, cfg)
        if got is None:
            continue
        out = got[1].float()
        assert torch.isfinite(out).all(), f"{impl.name} produced non-finite output"
        assert out.abs().max() > 1e-6, (
            f"{impl.name} ignored cache position {j} of {cfg.N}: output is all "
            "zero when only that position carries value, so it is not "
            "attending to every cached key")
        checked.append(impl.name)

    assert len(checked) >= 3, f"too few decode paths exercised: {checked}"


def test_decode_refuses_a_causal_flag_it_would_mis_mask():
    """causal=True at q_len=1 is not a decode configuration we can honour.

    Every published decode row carries causal=False, so no result depends on
    this. It is a guard against a future grid: SDPA would silently keep key 0
    while flash_attn_with_kvcache would keep all N, and the two would be
    compared as if they computed the same thing.
    """
    with pytest.raises(ValueError, match="causal"):
        Cfg("decode", B=1, Hq=2, Hkv=2, D=64, N=8, causal=True)
