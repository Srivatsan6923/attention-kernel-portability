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
    Cfg("prefill", B=2, Hq=4, Hkv=4, D=64, N=64),
    Cfg("prefill", B=2, Hq=4, Hkv=1, D=64, N=64),        # GQA
    Cfg("decode", B=2, Hq=4, Hkv=1, D=64, N=64, causal=False),
]


@pytest.mark.parametrize("cfg", SMALL, ids=lambda c: c.key())
def test_every_impl_passes_the_gate(cfg):
    info = dev_info()
    ran = 0
    for impl in IMPLS.values():
        if impl.regime != cfg.regime or not impl.supports(cfg, info):
            continue
        built = impl.build(cfg, DEV)
        res = check.gate(cfg, DEV, built.fn(),
                         built.meta.get("out_layout", "bhsd"))
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
        built = impl.build(cfg, DEV)
        built.fn()
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
    cfg = Cfg("decode", B=2, Hq=4, Hkv=1, D=64, N=64, causal=False)
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
    assert naive_peak_bytes(big) / 1e9 > 80
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
