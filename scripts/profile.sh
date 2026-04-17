#!/usr/bin/env bash
# Nsight counters and a launch timeline for the ten representative cells.
#
#   scripts/profile.sh [outdir]        # default: results/profile
#
# --prewarm builds each cell and calls it once, which is exactly what a profiler
# wants, so this needs no separate entry point. One ncu pass covers every cell;
# kernels are attributed afterwards by name.
#
# ncu needs GPU performance counters, which shared clusters usually withhold
# (ERR_NVGPUCTRPERM). nsys CUDA tracing does not, so it runs either way and the
# launch-overhead half of the attribution survives without counters.
set -euo pipefail

OUT=${1:-results/profile}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' /' '--')
mkdir -p "$OUT/$GPU"

# Explicit metrics rather than --set full: full replays every kernel many times
# and is roughly twenty times slower for counters we do not read.
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

# --sample=none/--cpuctxsw=none keeps nsys off perf_event_open, which is the
# only part of it that needs elevated privileges.
echo "nsys: collecting launch timeline"
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --force-overwrite=true \
     -o "$OUT/$GPU/timeline" \
     python -m akp.run --grid profile --prewarm >/dev/null 2>&1 || true

nsys stats --report cuda_gpu_kern_sum --format csv \
     --output "$OUT/$GPU/kern" "$OUT/$GPU/timeline.nsys-rep" >/dev/null 2>&1 || true
echo "  -> $OUT/$GPU/timeline.nsys-rep"

echo
echo "fold into the analysis with:  python -m akp.analysis results/raw"
