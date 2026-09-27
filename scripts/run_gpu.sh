#!/usr/bin/env bash
# Full sweep on one GPU, for hosts without Kubernetes (rented H100, Ada boxes).
#
#   scripts/run_gpu.sh                       # every grid, 5 repeats
#   scripts/run_gpu.sh prefill_full decode_full
#
# Each repeat runs in its own process. The bootstrap resamples these launches.
set -euo pipefail

GRIDS=${*:-"prefill_full prefill_gqa prefill_noncausal prefill_cold decode_full decode_cudagraph decode_cold"}
REPEATS=${REPEATS:-5}

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$PWD/.cache/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$PWD/.cache/triton}
export FLASHINFER_JIT_CACHE_DIR=${FLASHINFER_JIT_CACHE_DIR:-$PWD/.cache/flashinfer}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$FLASHINFER_JIT_CACHE_DIR"

for grid in $GRIDS; do
  # Fill the compile caches once before the timed repeats.
  python -m akp.run --grid "$grid" --prewarm
  for r in $(seq 0 $((REPEATS - 1))); do
    python -m akp.run --grid "$grid" --repeat "$r"
  done
done

python -m akp.analysis results/raw
