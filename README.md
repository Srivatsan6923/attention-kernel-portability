# Attention Kernel Portability

Does the fastest attention backend on one GPU stay the fastest on another?
I benchmarked 10 prefill and 6 decode implementations on six NVIDIA GPUs (A10,
A100, L40, L40S, H100, RTX 5090), measuring forward prefill and single-token
KV-cache decode separately.

[Paper](paper/main.pdf) · [Article](https://srivatsan6923.github.io/projects/attention-kernel-portability/) · [Result ledger](paper/ledger.md)

## Results

A backend counts as the winner only if the runner-up is at least 10% slower and
the lower bound of a 95% bootstrap interval on that ratio is above 1.

| | Forward prefill | Decode |
|---|---:|---:|
| Configurations with a winner by that rule | 128/447 (28.6%) | 409/1,203 (34.0%) |
| Of those, winner changes across Ampere/Ada/Hopper | 27/64 (42.2%) | 29/289 (10.0%) |
| Median cost of reusing the source GPU's choice | 1.002× | 1.000× |
| 95th percentile of that cost | 1.610× | 1.325× |

Most configurations have no clear winner. Where there is one, it changes often
between GPUs in prefill and rarely in decode. Reusing another GPU's choice
usually costs little at the median, but the tail is large, and in 85/590
prefill and 287/1,974 decode cases the chosen backend could not run on the
target GPU at all.

## Smoke test

```bash
pip install -e ".[dev]"
pytest tests/ -q
python -m akp.preflight
python -m akp.run --grid smoke
python -m akp.analysis results/raw
```

This checks the pipeline, not the study. Use the pinned image in
`env/Dockerfile`. Full-run scripts are in `scripts/`.

## Reproduce

The measurements are in release
[v1.2](https://github.com/Srivatsan6923/attention-kernel-portability/releases/tag/v1.2)
with SHA-256 checksums. Download the assets there, then rebuild the analysis:

```bash
sha256sum -c SHA256SUMS.txt
unzip akp-measurements-v1.zip
python -m akp.analysis ../akp-measurements-v1/raw --out results/processed
python paper/ledger.py
python -m akp.figures_portability results/processed --out site/public/figures
python -m akp.webdata results/processed site/public/data/web.json
```

This takes about two minutes on a CPU. `DATA_README.md` in the release lists
the counts the ledger should print. Collecting new measurements needs the six
GPUs.

## Scope

Latency is a single attention call with inputs already prepared, not request
latency, tokens/s or end-to-end TTFT. Each GPU was measured on its own host, so
host differences are not separated from GPU differences. Kernel traces were
captured for 66.5% of successful runs, and the rest are unverified.

The Triton path is the fused-attention tutorial kernel with a BF16 change.
FlashAttention and FlashInfer are the unmodified libraries. The harness,
correctness checks and analysis are mine.
