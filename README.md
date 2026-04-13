# Attention Kernel Portability

A benchmark of attention backends across six NVIDIA GPUs:
A10, A100, L40, L40S, H100 and RTX 5090.

We measure whether a backend selected on one GPU remains a good
choice on another, separately for forward prefill and single-token
KV-cache decode.

[Paper](paper/main.pdf) · [Result ledger](paper/ledger.md)

## Results

A backend is considered separated from the runner-up when the
runner-up is at least 10% slower and the lower bound of the 95%
bootstrap interval on that latency ratio exceeds 1.

| Measurement | Forward prefill | Decode |
|---|---:|---:|
| Configurations with a separated winner | 128/447 (28.6%) | 409/1,203 (34.0%) |
| Winner changes among eligible Ampere/Ada/Hopper comparisons | 27/64 (42.2%) | 29/289 (10.0%) |
| Median source-to-target latency ratio | 1.002× | 1.000× |
| 95th-percentile transfer ratio | 1.610× | 1.325× |

Winner-change comparisons require separation on both GPUs.
Transfer ratios use all six GPUs and require separation on the
source. No eligible target timing was available for 85/590 prefill
choices and 287/1,974 decode choices.

## Run a smoke test

Use the pinned environment in `env/Dockerfile`.

```bash
pip install -e ".[dev]"
pytest tests/ -q
python -m akp.preflight
python -m akp.run --grid smoke
python -m akp.analysis results/raw
```

A smoke test checks the pipeline; it does not reproduce the full
six-GPU study. Full-run scripts are in `scripts/`.

The measurement records behind the published results are not yet
released, so the six-GPU study cannot currently be reproduced from
this repository alone. A dataset release is pending.

## Scope and implementation

Results measure attention-call latency with prepared inputs.
Host and driver differences are not isolated from GPU differences.
Dispatch traces were captured for 66.5% of successful attempts.

The repository contains the benchmark harness, reference paths,
correctness checks and analysis. The Triton implementation is
adapted from the fused-attention tutorial with a BF16 modification;
FlashAttention and FlashInfer use their library implementations.
