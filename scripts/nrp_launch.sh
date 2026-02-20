#!/usr/bin/env bash
# Launch a grid on NRP.
#
#   IMAGE=ghcr.io/<user>/akp:v1 scripts/nrp_launch.sh prefill_full [nshards] [par]
#
# Checks the free A100 quota first, because the namespace is shared and pods
# over quota sit Pending indefinitely, which looks like a hang.
set -euo pipefail

GRID=${1:?usage: nrp_launch.sh <grid> [nshards] [parallelism]}
NSHARDS=${2:-2}
PAR=${3:-$NSHARDS}
: "${IMAGE:?set IMAGE to the pushed image reference}"

NS=$(kubectl config view --minify -o jsonpath='{..namespace}')
read -r HARD USED < <(kubectl get resourcequota a100-limit -n "$NS" \
  -o jsonpath='{.status.hard.requests\.nvidia\.com/a100} {.status.used.requests\.nvidia\.com/a100}')
FREE=$((HARD - USED))
echo "namespace $NS: a100 quota $USED/$HARD used, $FREE free; requesting $PAR"
if [ "$PAR" -gt "$FREE" ]; then
  echo "  (over quota - the extra pods will wait until it frees up)"
fi

# Five process-level repeats, each split into NSHARDS config shards.
export IMAGE GRID NSHARDS PAR
export COMPLETIONS=$((5 * NSHARDS))
# Kubernetes names are RFC 1123: no underscores.
export JOBNAME=${GRID//_/-}

# Restricted variable list: envsubst blanks every $VAR it knows about, and
# JOB_COMPLETION_INDEX must survive into the container to be expanded there.
VARS='$IMAGE $GRID $JOBNAME $NSHARDS $PAR $COMPLETIONS'

kubectl apply -f scripts/nrp_storage.yaml

# Both Jobs are in one file, so applying it starts them together and they would
# compete for the same quota. Split the documents and wait for prewarm, whose
# whole purpose is to fill the compile cache before the timed repeats run.
envsubst "$VARS" < scripts/nrp_job.yaml | awk '/^---$/{d++; next} {print > ("/tmp/akp-job-" d ".yaml")}'
kubectl apply -f /tmp/akp-job-.yaml
echo "waiting for prewarm to finish before starting the sweep..."
kubectl wait --for=condition=complete "job/akp-prewarm-$JOBNAME" --timeout=43200s
kubectl apply -f /tmp/akp-job-1.yaml

cat <<EOF

prewarm:  kubectl logs -f job/akp-prewarm-$JOBNAME
sweep:    kubectl get job akp-sweep-$JOBNAME -w
results:  scripts/nrp_fetch.sh ./results
EOF
