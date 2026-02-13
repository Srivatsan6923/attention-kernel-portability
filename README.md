# Attention Kernel Portability

Do conclusions about attention performance transfer across GPU architectures and
across prefill vs. single-token KV-cache decode?

Attention kernels are usually benchmarked on one GPU, in one regime, and the
ranking is reported as if it were a property of the kernel. This repo measures
how far such a ranking actually travels — across Ampere, Ada, Hopper and
Blackwell, and between prefill and decode — with every measurement gated on
numerical correctness and on a check that the kernel which ran is the kernel
that was requested.

## Why the dispatch check matters

A benchmark can call `flash_attn_func` and measure something else entirely: a
fallback path, a different SDPA backend, or a compiler substitution. Every cell
here records the CUDA kernels the call actually launched, and the requested vs.
observed backend is reconciled in analysis rather than assumed.

Two things this has already caught on the development GPU (RTX 4060, sm89,
torch 2.10 / triton 3.6):

- **TorchInductor does not rewrite naive attention into SDPA.**
  `counters["inductor"]["fuse_attention"] == 0` for all three compiled variants,
  including the `torch.where` spelling that `_sfdp_pattern_18/19` are written
  against. Inductor fuses the softmax epilogue into its own Triton kernel — 3
  launched kernels against the naive path's 8 — but keeps both GEMMs and still
  materializes the N×N score matrix.
- **The upstream Triton tutorial-06 kernel is fp16-only.** It hardcodes
  `tl.float16` in the forward and in three backward casts, so it fails to
  compile for bf16 at any shape. `akp/vendor/triton_tutorial06.py` carries a
  minimal patch deriving the dtype from the inputs; the header documents exactly
  what changed.

## Layout

    akp/impls.py     implementations behind one build(cfg, device) -> Built interface
    akp/bench.py     CUDA-event timing, device info, clock/throttle telemetry
    akp/check.py     fp32 reference, correctness gate, dispatch probe
    akp/run.py       grids, resume, interleaved measurement, status taxonomy
    akp/analysis.py  metrics, ranking inversions, dispatch audit, backend selector
    tests/           semantic checks for the failures that stay silent

## Running it

    pip install -e ".[dev]"          # add [kernels] for flash-attn + flashinfer
    pytest tests/ -q
    python -m akp.run --grid smoke   # ~90 s, writes results/raw/<gpu>/*.jsonl
    python -m akp.analysis

Grids: `smoke`, `prefill_full`, `prefill_gqa`, `prefill_noncausal`,
`prefill_cold`, `decode_full`, `decode_cudagraph`, `decode_cold`, `profile`.

Process-level repeats are a shell loop, since each needs a fresh CUDA context:

    for i in 0 1 2 3 4; do python -m akp.run --grid prefill_full --repeat $i; done

Runs are resumable. Rows are keyed by a config hash and appended as they are
produced, so a killed sweep restarts where it stopped.

## Method notes

**Causal alignment.** flash-attn ≥ 2.1 aligns its causal mask bottom-right;
PyTorch SDPA's `is_causal` is top-left. They agree only when `q_len == kv_len`.
At `q_len=1, kv_len=N` flash-attn attends to all N keys and SDPA attends to key
0 — fast, and silently wrong. Decode passes each API its own "attend to
everything" spelling; `tests/test_semantics.py` asserts the two conventions
disagree, so the test fails loudly if either library changes.

**Cache condition is an axis.** Flushing L2 before every call measures
cold-cache behaviour, but back-to-back decode in a server is warm. Both are
measured (`cache=warm` default, `cache=cold` grids) and reported separately.

**Timing.** K iterations per CUDA-event pair, K sized so a block exceeds 200 µs.
Measured event-pair overhead on the dev box is 5.1 µs, which would inflate a
10 µs decode kernel by roughly half if timed per iteration.

**Correctness gate.** `err(impl) <= max(2 · err(naive in same dtype), τ[dtype])`,
applied to the output and to dQ/dK/dV separately, with non-finite values failing
outright. Both halves of the rule are recorded so it is visible which one
decided a cell. Cells that fail are excluded from timing results and kept in the
failure table.

**Ordering.** Implementation order is shuffled per configuration and measured in
interleaved rounds, so clock and thermal drift is common-mode across the
implementations being compared. Confidence intervals come from a bootstrap
clustered on process repeats, since repeats inside one process share clock state.

## Status

Harness complete and verified on the development GPU. Benchmark runs on A100,
H100, and the held-out devices are pending.
