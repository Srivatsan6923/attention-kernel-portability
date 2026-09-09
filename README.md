# Attention Kernel Portability

Does the fastest attention backend on one GPU stay the fastest on another?
I benchmarked 10 prefill and 6 decode backends on six NVIDIA GPUs (A10, A100,
L40, L40S, H100, RTX 5090), measuring forward prefill and single-token KV-cache
decode separately.

[Paper](paper/main.pdf) · [Result ledger](paper/ledger.md)

## Results

A backend counts as the winner only if the runner-up is at least 10% slower and
the lower bound of a 95% bootstrap interval on that ratio is above 1.

| | Forward prefill | Decode |
|---|---:|---:|
| Configurations with a winner meeting that rule | 128/447 (28.6%) | 409/1,203 (34.0%) |
| Of those, winner changes across Ampere/Ada/Hopper | 27/64 (42.2%) | 29/289 (10.0%) |
| Median cost of carrying the source GPU's choice | 1.002× | 1.000× |
| 95th percentile of that cost | 1.610× | 1.325× |

Most configurations have no winner by that rule. Where there is one, the label
changes often in prefill and rarely in decode, but carrying the wrong choice
usually costs little at the median. The tail is where it hurts, and 85/590
prefill and 287/1,974 decode source choices had no eligible target timing at
all.

## Smoke test

```bash
pip install -e ".[dev]"
pytest tests/ -q
python -m akp.preflight
python -m akp.run --grid smoke
python -m akp.analysis results/raw
```

This checks the pipeline, not the study. Use the pinned image in
`env/Dockerfile`; full-run scripts are in `scripts/`.

Measurements for this release are available under
[v1.2](https://github.com/Srivatsan6923/attention-kernel-portability/releases/tag/v1.2),
with SHA-256 checksums and instructions for regenerating the results. That
reproduces the analysis from recorded measurements:

```bash
sha256sum -c SHA256SUMS.txt
unzip akp-measurements-v1.zip
python -m akp.analysis ../akp-measurements-v1/raw --out results/processed
python paper/ledger.py
```

Repeating the six-GPU collection is a separate job and needs the hardware.

## Scope

Latency is a single attention call with inputs already prepared: not request
latency, tokens/s, or end-to-end TTFT. Each GPU was measured on the host that
had it, so driver and host differences are not separated from GPU differences.
Dispatch traces were captured for 66.5% of successful attempts; the rest are
unknown, not verified.

The Triton path is the fused-attention tutorial kernel with a BF16 modification.
FlashAttention and FlashInfer are the library implementations. The harness,
correctness checks and analysis are mine.
