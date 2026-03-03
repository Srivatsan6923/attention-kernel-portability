#!/usr/bin/env bash
# Launch a grid on NRP.
#
#   IMAGE=ghcr.io/<user>/akp:v3 scripts/nrp_launch.sh decode_full [nshards] [par]
#   GPU=l40 IMAGE=... scripts/nrp_launch.sh decode_full 2 2
#
# GPU picks the device. A100 is the anchor and is a named, quota-limited
# resource; everything else is requested as plain nvidia.com/gpu and is pinned
# by node label instead. The job name carries the device, so grids for two
# devices can run at once without colliding.
set -euo pipefail

GRID=${1:?usage: [GPU=a100|a40|a10|l40|l40s|4090] nrp_launch.sh <grid> [nshards] [parallelism]}
NSHARDS=${2:-2}
PAR=${3:-$NSHARDS}
: "${IMAGE:?set IMAGE to the pushed image reference}"
GPU=${GPU:-a100}

# require is a substring of torch's device name, checked in the pod. The node
# label is the real guard; this is the second one, and it is what stops a
# mislabelled node writing rows under the wrong device.
#
# CPU is the node's per-GPU share, not what the compile would like. Inductor
# autotune and nvcc scale with cores, but a pod that never schedules compiles
# nothing at all: the L40 nodes carry 4 GPUs on 20 cores, so an 8-core request
# is above a fair share and sat Pending for four hours behind every job that
# asked for less. Divide the node's cores by its GPUs and round down.
case "$GPU" in
  a100) PRODUCT=NVIDIA-A100-SXM4-80GB;     RESOURCE=nvidia.com/a100; REQUIRE=A100-SXM4-80GB; CPU=16; MEM=64Gi ;;  # 252c/8g
  a40)  PRODUCT=NVIDIA-A40;                RESOURCE=nvidia.com/a40;  REQUIRE=A40;            CPU=8;  MEM=32Gi ;;
  a10)  PRODUCT=NVIDIA-A10;                RESOURCE=nvidia.com/gpu;  REQUIRE=A10;            CPU=8;  MEM=32Gi ;;  # 124c/8g
  l40)  PRODUCT=NVIDIA-L40;                RESOURCE=nvidia.com/gpu;  REQUIRE=L40;            CPU=4;  MEM=24Gi ;;  # 20c/4g
  l40s) PRODUCT=NVIDIA-L40S;               RESOURCE=nvidia.com/gpu;  REQUIRE=L40S;           CPU=6;  MEM=24Gi ;;  # 28c/4g
  4090) PRODUCT=NVIDIA-GeForce-RTX-4090;   RESOURCE=nvidia.com/gpu;  REQUIRE=4090;           CPU=6;  MEM=24Gi ;;  # 28c/4g
  *) echo "unknown GPU '$GPU'; known: a100 a40 a10 l40 l40s 4090" >&2; exit 1 ;;
esac

NS=$(kubectl config view --minify -o jsonpath='{..namespace}')

# Only the named resources carry a quota. Pods over it sit Pending indefinitely,
# which looks exactly like a hang, so say so before launching rather than after.
case "$RESOURCE" in
  nvidia.com/gpu) echo "namespace $NS: $PRODUCT via nvidia.com/gpu (no named quota); requesting $PAR" ;;
  *)
    # Plain expansion rather than `read < <(...)`: jsonpath prints no trailing
    # newline, so read returns non-zero at EOF and set -e kills the script
    # here, before the first echo, with no output to say why.
    KEY=${RESOURCE#nvidia.com/}
    QUOTA=$(kubectl get resourcequota "$KEY-limit" -n "$NS" \
      -o jsonpath="{.status.hard.requests\\.nvidia\\.com/$KEY} {.status.used.requests\\.nvidia\\.com/$KEY}")
    HARD=${QUOTA%% *}
    USED=${QUOTA##* }
    FREE=$((HARD - USED))
    echo "namespace $NS: $KEY quota $USED/$HARD used, $FREE free; requesting $PAR"
    if [ "$PAR" -gt "$FREE" ]; then
      echo "  (over quota - the extra pods will wait until it frees up)"
    fi ;;
esac

# Five process-level repeats, each split into NSHARDS config shards.
export IMAGE GRID NSHARDS PAR
export COMPLETIONS=$((5 * NSHARDS))
export GPU_PRODUCT=$PRODUCT GPU_RESOURCE=$RESOURCE REQUIRE_GPU=$REQUIRE CPU MEM
# Kubernetes names are RFC 1123: no underscores. The device is in the name so
# two devices can sweep the same grid concurrently.
export JOBNAME="${GRID//_/-}-$GPU"

# Restricted variable list: envsubst blanks every $VAR it knows about, and
# JOB_COMPLETION_INDEX must survive into the container to be expanded there.
VARS='$IMAGE $GRID $JOBNAME $NSHARDS $PAR $COMPLETIONS $GPU_PRODUCT $GPU_RESOURCE $REQUIRE_GPU $CPU $MEM'

kubectl apply -f scripts/nrp_storage.yaml

# Both Jobs are in one file, so applying it starts them together and they would
# compete for the same GPUs. Split the documents and wait for prewarm, whose
# whole purpose is to fill the compile cache before the timed repeats run.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
envsubst "$VARS" < scripts/nrp_job.yaml | awk -v d="$TMP" '/^---$/{n=1; next} {print > (d "/job-" n ".yaml")}'
kubectl apply -f "$TMP/job-.yaml"
echo "waiting for prewarm to finish before starting the sweep..."
kubectl wait --for=condition=complete "job/akp-prewarm-$JOBNAME" --timeout=43200s
kubectl apply -f "$TMP/job-1.yaml"

cat <<EOF

prewarm:  kubectl logs -f job/akp-prewarm-$JOBNAME
sweep:    kubectl get job akp-sweep-$JOBNAME -w
results:  scripts/nrp_fetch.sh ./results
EOF
