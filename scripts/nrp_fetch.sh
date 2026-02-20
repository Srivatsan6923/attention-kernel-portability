#!/usr/bin/env bash
# Copy results off the PVC.
#
#   scripts/nrp_fetch.sh [dest]      # default dest: ./results
#
# kubectl cp execs tar inside a running container, and once the sweep Jobs are
# Completed there is nothing left to exec into. So stand up a short-lived
# CPU-only pod that mounts the volume, copy, then delete it. No GPU, so it does
# not touch the A100 quota or trip the idle-GPU policy.
set -euo pipefail

DEST=${1:-./results}
POD=akp-fetch

cleanup() { kubectl delete pod "$POD" --ignore-not-found --wait=false >/dev/null 2>&1 || true; }
trap cleanup EXIT

kubectl delete pod "$POD" --ignore-not-found --wait=true >/dev/null 2>&1 || true
kubectl apply -f - >/dev/null <<'YAML'
apiVersion: v1
kind: Pod
metadata:
  name: akp-fetch
spec:
  restartPolicy: Never
  containers:
    - name: fetch
      image: busybox:1.36
      command: ["sh", "-c", "sleep 3600"]
      resources:
        limits:   {cpu: "1", memory: 2Gi}
        requests: {cpu: "1", memory: 2Gi}
      volumeMounts:
        - {name: data, mountPath: /data}
  volumes:
    - {name: data, persistentVolumeClaim: {claimName: akp-data}}
YAML

kubectl wait --for=condition=Ready "pod/$POD" --timeout=300s
mkdir -p "$DEST"
# raw shards, the interned dispatch traces and the per-device manifests all
# live under /data, not under /data/raw.
kubectl cp "$POD:/data/raw" "$DEST/raw"
kubectl cp "$POD:/data/environment" "$DEST/environment"
kubectl cp "$POD:/data/dispatch.jsonl" "$DEST/dispatch.jsonl"

echo "copied to $DEST"
find "$DEST" -name '*.jsonl' | wc -l | xargs echo "  jsonl files:"
