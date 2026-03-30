# Attention Kernel Portability

Do conclusions about attention performance transfer across GPU architectures and
across prefill vs. single-token KV-cache decode?

Attention kernels are usually benchmarked on one GPU, in one regime, and the
ranking is reported as if it were a property of the kernel. This repo measures
how far such a ranking actually travels, across Ampere, Ada, Hopper and
Blackwell and between prefill and decode, with every measurement gated on
numerical correctness and on a check that the kernel which ran is the kernel
that was requested.

## Why the dispatch check matters

A benchmark can call `flash_attn_func` and measure something else entirely: a
fallback path, a different SDPA backend, or a compiler substitution. Every cell
here records the CUDA kernels the call actually launched, and the requested vs.
observed backend is reconciled in analysis rather than assumed.

Two things this caught, first on the development GPU and then across the full
six-device sweep:

- **TorchInductor does not rewrite naive attention into SDPA.**
  `counters["inductor"]["fuse_attention"] == 0` for all three compiled variants,
  including the `torch.where` spelling that `_sfdp_pattern_18/19` are written
  against. Across the finished dataset, 3,645 of 17,537 Inductor rows captured
  the counter and it is 0 on every one of them, on all six devices. Inductor fuses the softmax epilogue into its own Triton kernel, 3
  launched kernels against the naive path's 8, but keeps both GEMMs and still
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
    akp/preflight.py per-GPU gate a sweep refuses to start without
    akp/figures.py   the paper figures and the winner map
    akp/webdata.py   aggregates results/processed into the article's web.json
    tests/           semantic checks for the failures that stay silent
    dashboard/app.py results dashboard
    paper/           the preprint, plus numbers.md tracing every quoted figure
    site/            the long-form article (Astro + MDX)
    env/Dockerfile   pinned image (torch 2.9, flash-attn, flashinfer)
    scripts/         NRP job specs, the single-host sweep loop, profiling

## Running it

    pip install -e ".[dev]"          # add [kernels] for flash-attn + flashinfer
    pytest tests/ -q
    python -m akp.preflight          # must print Overall: PASS
    python -m akp.run --grid smoke   # ~90 s, writes results/raw/<gpu>/*.jsonl
    python -m akp.analysis results/raw

`akp.preflight` is the gate on collecting anything from a GPU. It checks the
device is the one requested and not a MIG slice, that nothing else is resident
on it, that it is not already throttled, that every implementation imports,
runs, dispatches to a real kernel and passes the numerical gate in both
directions, that the OOM predictor bounds the real allocation, and that
doubling the work doubles the measured time. It exits non-zero on failure, so
a sweep can be gated on it, and it drives the same code the sweep drives
rather than reimplementing the checks.

It has caught, on real hardware: an OOM predictor short by 35%, a throttle
filter discarding a fifth of the rows, and an image whose baked git sha named
a commit that could not have produced the code inside it.

Grids: `smoke`, `prefill_full`, `prefill_gqa`, `prefill_noncausal`,
`prefill_cold`, `decode_full`, `decode_cudagraph`, `decode_cold`, `profile`.

Process-level repeats are a shell loop, since each needs a fresh CUDA context:

    for i in 0 1 2 3 4; do python -m akp.run --grid prefill_full --repeat $i; done

Runs are resumable. Rows are keyed by a config hash and appended as they are
produced, so a killed sweep restarts where it stopped.

`--prewarm` builds every cell once without timing or writing anything. Inductor
autotune and FlashInfer JIT are paid per shape and dominate wall time, so doing
them once up front keeps them out of the measured repeats.

## Running on a cluster

    docker build -f env/Dockerfile \
      --build-arg GIT_SHA=$(git rev-parse HEAD) -t ghcr.io/<user>/akp:<tag> .
    docker push ghcr.io/<user>/akp:<tag>

The sha is not optional: every row carries it as its provenance, and the build
refuses anything that is not a full 40-character sha. An image that names one
commit while containing another produces a dataset nobody can reproduce.

On NRP/Nautilus, once per namespace:

    kubectl apply -f scripts/nrp_storage.yaml

Then per grid:

    IMAGE=ghcr.io/<user>/akp:v1 scripts/nrp_launch.sh prefill_full 2 2

That is a prewarm Job followed by an Indexed Job of five process repeats across
NSHARDS config shards, one A100 per pod. Jobs rather than interactive pods,
since interactive pods are destroyed after six hours and a full grid takes
longer than that.

Every pod runs `akp.preflight` before `akp.run` and exits without writing rows
if it fails. The gate is per pod, not per campaign, because pods land on
different nodes. Reports are kept under `/data/preflight/`.

On a single host without Kubernetes (a rented H100, an Ada box):

    scripts/run_gpu.sh prefill_full decode_full

## Dashboard

    pip install -e ".[dash]"
    python -m akp.analysis results/raw   # writes results/processed/
    streamlit run dashboard/app.py

Seven pages: hardware and environment, prefill, decode, dispatch and fallbacks,
ranking inversions, Nsight attribution, and backend selection. It reads
`results/processed/` and recomputes nothing, so a number here and the same
number in the report cannot disagree.

## Method notes

**Causal alignment.** flash-attn ≥ 2.1 aligns its causal mask bottom-right;
PyTorch SDPA's `is_causal` is top-left. They agree only when `q_len == kv_len`.
At `q_len=1, kv_len=N` flash-attn attends to all N keys and SDPA attends to key
0, which is fast and silently wrong. Decode passes each API its own "attend to
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

**Ordering.** Implementation order is shuffled per configuration, so clock and
thermal drift is common-mode across the implementations being compared. Rows
measured while the GPU reported a throttle reason are dropped. Confidence
intervals come from a bootstrap clustered on process repeats, since repeats
inside one process share clock state.

**Profiling.** `scripts/profile.sh` collects Nsight counters and a launch
timeline for ten representative cells. `ncu` needs GPU performance counters,
which shared clusters usually withhold; `nsys` CUDA tracing does not, so the
launch-overhead half of the attribution survives without them.

## Results

Six GPUs across four architecture families (sm80, sm86, sm89, sm90, sm120),
73,230 measured rows, 11,858 usable (GPU, configuration, implementation)
medians, one software snapshot: torch 2.9.0+cu128, CUDA 12.8, triton 3.5.0,
flash-attn 2.8.3, flashinfer 0.6.18.

- **Most configurations have no decisive winner.** On 68.1% of them the fastest
  implementation is not separated from the runner-up by both a 10% margin and a
  bootstrap interval excluding 1.
- **Decode rankings mostly transfer across Ampere, Ada and Hopper** (26 of 274
  separated comparisons flip); **prefill rankings do not** (82 of 144).
- **Decode agreement collapses against Blackwell** (64 of 84), and the cause is
  availability rather than architecture: FlashInfer, the decode winner on most
  other devices, raised a compute-capability error on 950 of 960 attempted rows
  on the RTX 5090.
- **The wheels show the same asymmetry statically.** flash-attn 2.8.3 ships no
  sm_86 or sm_89 cubins and no PTX, so on those parts it runs sm_80 code
  (`scripts/provenance.sh`).

`paper/` holds the preprint and `paper/numbers.md` traces every number in it
back to the code that produced it. `site/` is the long-form write-up.

## Status

Data collection is complete. L40 carries 3 process repeats against 5 elsewhere,
with full cell coverage; subsampling a complete device to 3 repeats moves its
separated-winner count by 3 to 5 cells in 192, so the criterion is dominated by
the 1.10 margin rather than by sampling depth.

The raw shards (456 MB) and `results/processed/` are not in git. A dataset
release is the remaining step before the preprint is submitted.
