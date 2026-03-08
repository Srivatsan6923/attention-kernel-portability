"""Attention implementations behind one interface: build(cfg, device) -> Built.

build() puts inputs in the kernel native layout; fn() only calls the kernel, so
nothing in the timed path is a transpose or an allocation.

Cfg.causal is the intended math, not an argument to pass through. FlashAttention
>= 2.1 aligns causal bottom-right, SDPA is_causal is top-left; they agree only
when q_len == kv_len. At q_len=1 FA attends to all N keys while SDPA attends to
key 0, so decode passes each API its own "attend to everything" spelling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Cfg:
    regime: str            # "prefill" | "decode"
    B: int
    Hq: int
    Hkv: int
    D: int
    N: int                 # prefill: q_len == kv_len.  decode: kv_len
    dtype: str = "bf16"
    mode: str = "fwd"      # "fwd" | "fwd_bwd"          (prefill only)
    launch: str = "eager"  # "eager" | "cudagraph"      (decode only)
    cache: str = "warm"    # "warm" | "cold"  -- L2 state at the start of a call
    causal: bool = True    # mathematical intent; see module docstring

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.D)

    @property
    def gqa(self) -> int:
        return self.Hq // self.Hkv

    @property
    def q_len(self) -> int:
        return self.N if self.regime == "prefill" else 1

    def key(self) -> str:
        return (f"{self.regime}/B{self.B}/Hq{self.Hq}/Hkv{self.Hkv}/D{self.D}"
                f"/N{self.N}/{self.dtype}/{self.mode}/{self.launch}"
                f"/{self.cache}/causal{int(self.causal)}")


def itemsize(cfg: Cfg) -> int:
    return torch.finfo(cfg.torch_dtype).bits // 8


# The impls that materialise the B*Hq*N*N score matrix, and so are the ones
# naive_peak_bytes describes. Lives here rather than in run.py because analysis
# needs it too, and run.py cannot be imported without triton.
NAIVE_LIKE = ("P0-naive", "P1-inductor", "P1-inductor-nofuse",
              "P1-inductor-where")


def naive_peak_bytes(cfg: Cfg) -> int:
    """Peak allocation of the score-matrix impls: scores, fp32 upcast, probs.

    Repeatedly attempting a 68 GB allocation fragments the caching allocator, so
    cells above capacity are skipped rather than tried.
    """
    if cfg.regime != "prefill":
        return 0
    n_scores = cfg.B * cfg.Hq * cfg.N * cfg.N
    qkv = 3 * cfg.B * cfg.Hq * cfg.N * cfg.D * itemsize(cfg)
    # Measured 10.8 B per score element on sm89: softmax(dtype=fp32) holds the
    # dtype scores, an fp32 upcast of them and its fp32 output at once, and the
    # bool causal mask outlives all three. 2*itemsize + 9 bounds that. Erring
    # high only skips a cell; erring low attempts the allocation this exists to
    # refuse, and a failed 68 GB attempt fragments the allocator.
    return qkv + n_scores * (2 * itemsize(cfg) + 9)


# --------------------------------------------------------------------------- #
# Inputs -- logical layout is BHSD everywhere; impls convert in build()
# --------------------------------------------------------------------------- #

def make_inputs(cfg: Cfg, device: torch.device, seed: int = 0,
                requires_grad: bool = False) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    dt = cfg.torch_dtype

    def rnd(*shape):
        return torch.randn(shape, generator=g, device=device, dtype=dt)

    q = rnd(cfg.B, cfg.Hq, cfg.q_len, cfg.D)
    k = rnd(cfg.B, cfg.Hkv, cfg.N, cfg.D)
    v = rnd(cfg.B, cfg.Hkv, cfg.N, cfg.D)

    if requires_grad:
        q, k, v = (t.requires_grad_(True) for t in (q, k, v))

    out = {"q": q, "k": k, "v": v}
    if cfg.mode == "fwd_bwd":
        out["do"] = rnd(cfg.B, cfg.Hq, cfg.q_len, cfg.D)
    return out


def expand_kv(t: torch.Tensor, gqa: int) -> torch.Tensor:
    """Query head i attends to kv head i // gqa.

    repeat_interleave, not repeat: repeat gives i % Hkv and is silently wrong.
    """
    return t if gqa == 1 else t.repeat_interleave(gqa, dim=1)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

@dataclass
class Built:
    fn: Callable[[], torch.Tensor]
    meta: dict = field(default_factory=dict)


@dataclass
class Impl:
    name: str
    regime: str
    build: Callable[[Cfg, torch.device], Built]
    kernel_patterns: tuple[str, ...]   # classified in analysis.py, never asserted here
    supports: Callable[[Cfg, dict], bool] = lambda cfg, dev: True
    note: str = ""


IMPLS: dict[str, Impl] = {}


def register(name, regime, kernel_patterns, supports=None, note=""):
    def deco(build):
        IMPLS[name] = Impl(name=name, regime=regime, build=build,
                           kernel_patterns=kernel_patterns,
                           supports=supports or (lambda cfg, dev: True),
                           note=note)
        return build
    return deco


# --------------------------------------------------------------------------- #
# Reference math (also P0/D0)
# --------------------------------------------------------------------------- #

def causal_mask(n_q: int, n_k: int, device) -> torch.Tensor:
    """Bottom-right aligned causal mask, matching flash-attn >= 2.1."""
    return torch.ones(n_q, n_k, dtype=torch.bool, device=device).tril(
        diagonal=n_k - n_q)


def naive_attention(q, k, v, mask, scale: float, mask_mode: str = "masked_fill"):
    """Explicit score matrix, written the way users write it.

    mask is built once in build(); constructing it here would allocate a
    16M-element tensor per call at N=4096.
    """
    s = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        if mask_mode == "where":
            # Same math as masked_fill, but the spelling TorchInductor
            # _sfdp_pattern_18/19 match against.
            s = torch.where(mask, s, torch.full((), float("-inf"),
                                                dtype=s.dtype, device=s.device))
        else:
            s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    return p @ v


def _fwd_or_fwd_bwd(cfg: Cfg, t: dict, forward: Callable[[], torch.Tensor]):
    """Wrap a forward into the callable the timer will loop over."""
    if cfg.mode != "fwd_bwd":
        return forward
    do = t["do"]
    grads = [x for x in (t["q"], t["k"], t["v"]) if x.requires_grad]

    def fwd_bwd():
        for x in grads:
            x.grad = None
        o = forward()
        o.backward(do, retain_graph=False)
        return o

    fwd_bwd.grad_tensors = (t["q"], t["k"], t["v"])
    return fwd_bwd


# Optional deps are imported lazily: a box without flash-attn or flashinfer
# still loads the module and reports those cells UNSUPPORTED.

def _try(modname):
    try:
        import importlib
        return importlib.import_module(modname)
    except Exception:
        return None


def has(modname) -> bool:
    return _try(modname) is not None


def sm(dev: dict) -> int:
    """Compute capability as an integer, e.g. sm90 -> 90."""
    return dev["cc_major"] * 10 + dev["cc_minor"]


# --------------------------------------------------------------------------- #
# PREFILL
# --------------------------------------------------------------------------- #

@register("P0-naive", "prefill", (r"gemm|matmul|softmax|elementwise",),
          note="explicit N x N score matrix; correctness reference and OOM taxonomy")
def _p0(cfg, device):
    t = make_inputs(cfg, device, requires_grad=(cfg.mode == "fwd_bwd"))
    k = expand_kv(t["k"], cfg.gqa)
    v = expand_kv(t["v"], cfg.gqa)
    m = causal_mask(cfg.q_len, cfg.N, device) if cfg.causal else None
    fwd = lambda: naive_attention(t["q"], k, v, m, cfg.scale)
    return Built(_fwd_or_fwd_bwd(cfg, t, fwd),
                 {"gqa_mode": "expanded" if cfg.gqa > 1 else "native",
                  "out_layout": "bhsd"})


def _compiled_naive(cfg, device, fuse: bool, mask_mode: str = "masked_fill"):
    import torch._inductor.config as icfg
    t = make_inputs(cfg, device, requires_grad=(cfg.mode == "fwd_bwd"))
    k = expand_kv(t["k"], cfg.gqa)
    v = expand_kv(t["v"], cfg.gqa)

    m = causal_mask(cfg.q_len, cfg.N, device) if cfg.causal else None
    torch._dynamo.reset()
    prev = icfg.pattern_matcher
    icfg.pattern_matcher = fuse
    try:
        compiled = torch.compile(naive_attention,
                                 mode="max-autotune-no-cudagraphs",
                                 dynamic=False)
        fwd = lambda: compiled(t["q"], k, v, m, cfg.scale, mask_mode)
        # Compile and autotune here, outside the timed region.
        from torch._dynamo.utils import counters
        counters.clear()
        _fwd_or_fwd_bwd(cfg, t, fwd)()
        torch.cuda.synchronize()
        fired = int(counters["inductor"].get("fuse_attention", 0))
    finally:
        icfg.pattern_matcher = prev

    return Built(_fwd_or_fwd_bwd(cfg, t, fwd),
                 {"gqa_mode": "expanded" if cfg.gqa > 1 else "native",
                  "fuse_attention": fired, "pattern_matcher": fuse,
                  "mask_mode": mask_mode, "out_layout": "bhsd"})


@register("P1-inductor", "prefill", (r"triton_|gemm|flash|fmha|cudnn",),
          note="torch.compile as a user gets it; records the fuse_attention counter")
def _p1(cfg, device):
    return _compiled_naive(cfg, device, fuse=True)


@register("P1-inductor-nofuse", "prefill", (r"triton_|gemm",),
          note="pattern_matcher=False: Inductor's own codegen, not SDPA")
def _p1n(cfg, device):
    return _compiled_naive(cfg, device, fuse=False)


def _sdpa_impl(cfg, device, backend, enable_gqa):
    t = make_inputs(cfg, device, requires_grad=(cfg.mode == "fwd_bwd"))
    if enable_gqa and cfg.gqa > 1:
        k, v, gqa_mode = t["k"], t["v"], "native"
    else:
        k, v = expand_kv(t["k"], cfg.gqa), expand_kv(t["v"], cfg.gqa)
        gqa_mode = "expanded" if cfg.gqa > 1 else "native"

    kw = {"scale": cfg.scale, "is_causal": cfg.causal}
    if enable_gqa and cfg.gqa > 1:
        kw["enable_gqa"] = True

    if backend is None:
        def fwd():
            return F.scaled_dot_product_attention(t["q"], k, v, **kw)
    else:
        def fwd():
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(t["q"], k, v, **kw)

    return Built(_fwd_or_fwd_bwd(cfg, t, fwd),
                 {"gqa_mode": gqa_mode,
                  "sdpa_backend": backend.name if backend else "auto",
                  "out_layout": "bhsd"})


@register("P1-inductor-where", "prefill", (r"triton_|gemm|flash|fmha|cudnn",),
          note="torch.where spelling: the form Inductor's SDPA patterns match")
def _p1w(cfg, device):
    return _compiled_naive(cfg, device, fuse=True, mask_mode="where")


@register("P2b-sdpa-mem-eff", "prefill", (r"fmha|cutlass|efficient",),
          note="CUTLASS fused MHA: the path that fires when flash declines")
def _p2b(cfg, device):
    return _sdpa_impl(cfg, device, SDPBackend.EFFICIENT_ATTENTION, enable_gqa=True)


@register("P2c-sdpa-flash", "prefill", (r"flash",),
          note="same algorithm as P4-fa2, different dispatch AND build provenance")
def _p2c(cfg, device):
    return _sdpa_impl(cfg, device, SDPBackend.FLASH_ATTENTION, enable_gqa=True)


@register("P2d-sdpa-cudnn", "prefill", (r"cudnn|sm\d+_.*attn",),
          note="vendor library path; availability and fallback vary by arch")
def _p2d(cfg, device):
    return _sdpa_impl(cfg, device, SDPBackend.CUDNN_ATTENTION, enable_gqa=True)


# N >= 128 because the tutorial autotunes over BLOCK_M in [64, 128] and writes a
# whole tile: at N < BLOCK_M it runs past the end of the output. Which config
# autotune picks varies per process, so shorter sequences fail intermittently.
@register("P3-triton", "prefill", (r"_attn_fwd|_attn_bwd",),
          supports=lambda cfg, dev: cfg.D in (16, 32, 64, 128, 256) and cfg.N >= 128,
          note="vendored Triton tutorial-06; JIT-retargeted per device; MHA only")
def _p3(cfg, device):
    from akp.vendor.triton_tutorial06 import attention as triton_attention
    t = make_inputs(cfg, device, requires_grad=(cfg.mode == "fwd_bwd"))
    # The tutorial kernel has no GQA path: K/V must be materialized per q-head.
    k = expand_kv(t["k"], cfg.gqa).contiguous()
    v = expand_kv(t["v"], cfg.gqa).contiguous()
    q = t["q"].contiguous()
    fwd = lambda: triton_attention(q, k, v, cfg.causal, cfg.scale, False)
    return Built(_fwd_or_fwd_bwd(cfg, t, fwd),
                 {"gqa_mode": "expanded" if cfg.gqa > 1 else "native",
                  "warp_specialize": False, "out_layout": "bhsd"})


def _fa_prefill(cfg, device, v3: bool):
    mod = _try("flash_attn_interface" if v3 else "flash_attn")
    t = make_inputs(cfg, device, requires_grad=(cfg.mode == "fwd_bwd"))
    # flash-attn wants BSHD; transpose outside the timed region.
    q = t["q"].transpose(1, 2).contiguous()
    k = t["k"].transpose(1, 2).contiguous()
    v = t["v"].transpose(1, 2).contiguous()
    if cfg.mode == "fwd_bwd":
        q, k, v = (x.detach().requires_grad_(True) for x in (q, k, v))
        t = {**t, "q": q, "k": k, "v": v,
             "do": t["do"].transpose(1, 2).contiguous()}
    fn = mod.flash_attn_func
    fwd = lambda: fn(q, k, v, softmax_scale=cfg.scale, causal=cfg.causal)
    return Built(_fwd_or_fwd_bwd(cfg, t, fwd),
                 {"gqa_mode": "native", "fa_version": 3 if v3 else 2,
                  "out_layout": "bshd"})


@register("P4-fa2", "prefill", (r"flash_fwd|flash_bwd|flash::",),
          supports=lambda cfg, dev: has("flash_attn") and cfg.D <= 256,
          note="anchor and cross-device normalizing baseline")
def _p4(cfg, device):
    return _fa_prefill(cfg, device, v3=False)


@register("P4h-fa3", "prefill", (r"flash|hopper",),
          supports=lambda cfg, dev: has("flash_attn_interface") and sm(dev) == 90,
          note="sm90 only: a kernel that cannot be evaluated off-arch")
def _p4h(cfg, device):
    return _fa_prefill(cfg, device, v3=True)


# DECODE: q_len == 1 over a cache pre-filled to N, attention only.
# flash_attn_with_kvcache appends in place when k/v are passed, so k=v=None is
# required or the cache changes while it is being timed.

@register("D0-naive-kv", "decode", (r"gemm|matmul|softmax|elementwise",),
          note="explicit scores over the cache; reference, and shows GQA expansion cost")
def _d0(cfg, device):
    t = make_inputs(cfg, device)
    k, v = expand_kv(t["k"], cfg.gqa), expand_kv(t["v"], cfg.gqa)
    return Built(lambda: naive_attention(t["q"], k, v, None, cfg.scale),
                 {"gqa_mode": "expanded" if cfg.gqa > 1 else "native",
                  "out_layout": "bhsd"})


@register("D1-inductor", "decode", (r"triton_|gemm|flash|fmha",),
          note="strip only: decode is bandwidth-bound, compile cost is not")
def _d1(cfg, device):
    import torch._inductor.config as icfg
    t = make_inputs(cfg, device)
    k, v = expand_kv(t["k"], cfg.gqa), expand_kv(t["v"], cfg.gqa)
    torch._dynamo.reset()
    compiled = torch.compile(naive_attention, mode="max-autotune-no-cudagraphs",
                             dynamic=False)
    fn = lambda: compiled(t["q"], k, v, None, cfg.scale)
    from torch._dynamo.utils import counters
    counters.clear()
    fn(); torch.cuda.synchronize()          # compile outside the timed region
    return Built(fn, {"gqa_mode": "expanded" if cfg.gqa > 1 else "native",
                      "fuse_attention": int(counters["inductor"].get("fuse_attention", 0)),
                      "pattern_matcher": icfg.pattern_matcher,
                      "out_layout": "bhsd"})


@register("D2-sdpa", "decode", (r"flash|fmha|cutlass|gemm",),
          note="default dispatcher: which backend SDPA picks at q_len=1 IS the measurement")
def _d2(cfg, device):
    return _sdpa_impl(cfg, device, None, enable_gqa=True)


@register("D3-fa-kvcache", "decode", (r"flash_fwd_splitkv|flash_fwd|flash::",),
          supports=lambda cfg, dev: has("flash_attn") and cfg.D <= 256,
          note="contiguous vendor KV-cache path; k=v=None so the cache is read-only")
def _d3(cfg, device):
    from flash_attn import flash_attn_with_kvcache
    t = make_inputs(cfg, device)
    # BSHD: q (B,1,Hq,D), caches (B,N,Hkv,D).
    q = t["q"].transpose(1, 2).contiguous()
    kc = t["k"].transpose(1, 2).contiguous()
    vc = t["v"].transpose(1, 2).contiguous()
    seqlens = torch.full((cfg.B,), cfg.N, dtype=torch.int32, device=device)
    fn = lambda: flash_attn_with_kvcache(
        q, kc, vc, k=None, v=None, cache_seqlens=seqlens,
        softmax_scale=cfg.scale, causal=False)
    return Built(fn, {"gqa_mode": "native", "cache_mutated": False,
                      "out_layout": "bshd", "_cache_tensors": (kc, vc)})


@register("D4-flashinfer", "decode", (r"BatchDecode|flashinfer|decode",),
          supports=lambda cfg, dev: has("flashinfer") and cfg.D in (64, 128, 256),
          note="only paged-layout entry; page_size=N_kv so the layout matches D3")
def _d4(cfg, device):
    import flashinfer
    t = make_inputs(cfg, device)
    # page_size = N is one page per sequence, so D3 vs D4 compares kernels
    # rather than layouts. The page_size sweep is a separate ablation.
    page = cfg.N
    q = t["q"].transpose(1, 2).contiguous().squeeze(1)        # (B, Hq, D)
    kc = t["k"].transpose(1, 2).contiguous().unsqueeze(0).reshape(
        cfg.B, page, cfg.Hkv, cfg.D)
    vc = t["v"].transpose(1, 2).contiguous().unsqueeze(0).reshape(
        cfg.B, page, cfg.Hkv, cfg.D)
    indptr = torch.arange(cfg.B + 1, dtype=torch.int32, device=device)
    indices = torch.arange(cfg.B, dtype=torch.int32, device=device)
    last = torch.full((cfg.B,), page, dtype=torch.int32, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    # plan() schedules; it must stay outside the timed region and outside
    # CUDA-graph capture.
    wrapper.plan(indptr, indices, last, cfg.Hq, cfg.Hkv, cfg.D, page,
                 q_data_type=cfg.torch_dtype, kv_data_type=cfg.torch_dtype)
    fn = lambda: wrapper.run(q, (kc, vc))
    return Built(fn, {"gqa_mode": "native", "page_size": page,
                      "out_layout": "bhd", "_cache_tensors": (kc, vc)})


@register("D6-fa-prefill-at-1", "decode", (r"flash_fwd|flash::",),
          supports=lambda cfg, dev: has("flash_attn") and cfg.D <= 256,
          note="MISLABELING CONTROL: the prefill interface called at q_len=1")
def _d6(cfg, device):
    from flash_attn import flash_attn_func
    t = make_inputs(cfg, device)
    q = t["q"].transpose(1, 2).contiguous()
    k = t["k"].transpose(1, 2).contiguous()
    v = t["v"].transpose(1, 2).contiguous()
    return Built(lambda: flash_attn_func(q, k, v, softmax_scale=cfg.scale,
                                         causal=False),
                 {"gqa_mode": "native", "out_layout": "bshd"})


def impls_for(regime: str) -> list[Impl]:
    return [i for i in IMPLS.values() if i.regime == regime]
