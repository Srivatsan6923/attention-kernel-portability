#!/usr/bin/env bash
# Nsight counters and a launch timeline for the ten representative cells.
#
#   scripts/profile.sh [outdir]        # default: results/profile
#
# --prewarm builds each cell and calls it once, which is all a profiler needs.
# Kernels are matched to cells by name afterwards.
#
# ncu needs performance counters, which shared hosts often block
# (ERR_NVGPUCTRPERM). nsys tracing works without them, so it always runs.
set -euo pipefail

OUT=${1:-results/profile}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' /' '--')
mkdir -p "$OUT/$GPU"

# A short metric list. --set full replays each kernel many times and is much
# slower.
METRICS=gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
gpu__time_duration.sum

if ncu --query-metrics >/dev/null 2>&1; then
  echo "ncu: collecting counters"
  ncu --csv --target-processes all --metrics "$METRICS" \
      --kernel-name-base demangled \
      python -m akp.run --grid profile --prewarm \
      > "$OUT/$GPU/ncu.csv" 2> "$OUT/$GPU/ncu.log" || true
  echo "  -> $OUT/$GPU/ncu.csv"
else
  echo "ncu: performance counters unavailable on this host (ERR_NVGPUCTRPERM);"
  echo "     skipping counters, the CUPTI timeline below still runs"
fi

# --sample=none and --cpuctxsw=none avoid perf_event_open, the only part of
# nsys that needs elevated privileges.
echo "nsys: collecting launch timeline"
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --force-overwrite=true \
     -o "$OUT/$GPU/timeline" \
     python -m akp.run --grid profile --prewarm >/dev/null 2>&1 || true

nsys stats --report cuda_gpu_kern_sum --format csv \
     --output "$OUT/$GPU/kern" "$OUT/$GPU/timeline.nsys-rep" >/dev/null 2>&1 || true
echo "  -> $OUT/$GPU/timeline.nsys-rep"

echo
echo "fold into the analysis with:  python -m akp.analysis results/raw"
