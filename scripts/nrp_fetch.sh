#!/usr/bin/env bash
# Copy results off the PVC.
#
#   scripts/nrp_fetch.sh [dest]      # default dest: ./results
#
# kubectl cp needs a running container, and the sweep pods have exited. This
# starts a small CPU-only pod that mounts the volume, copies, and deletes it.
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
kubectl cp "$POD:/data/raw" "$DEST/raw"
kubectl cp "$POD:/data/environment" "$DEST/environment"
kubectl cp "$POD:/data/dispatch.jsonl" "$DEST/dispatch.jsonl"
# Preflight reports, one per pod.
kubectl cp "$POD:/data/preflight" "$DEST/preflight" 2>/dev/null || true

echo "copied to $DEST"
find "$DEST/raw" -name '*.jsonl' | wc -l | xargs echo "  shards:"
for d in "$DEST"/raw/*/; do
  [ -d "$d" ] && echo "  $(basename "$d"): $(cat "$d"/*.jsonl 2>/dev/null | wc -l) rows"
done
