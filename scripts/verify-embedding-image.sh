#!/usr/bin/env bash
set -euo pipefail
image=${1:?image required}
revision=${2:?source revision required}
release=${3:?release tag required}
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/embedding-evidence"
name="rag-embedding-check-$$"
cleanup() {
  docker logs "$name" > "$root/embedding-evidence/runtime.log" 2>&1 || true
  docker inspect "$name" > "$root/embedding-evidence/container.json" 2>/dev/null || true
  docker rm -f "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
# Named Docker cache persists across checks and is not removed by cleanup.
cache_mount=rag-embedding-validation-cache
if [[ -n ${RAG_EMBEDDING_VERIFY_CACHE_PATH:-} ]]; then
  cache_mount=$RAG_EMBEDDING_VERIFY_CACHE_PATH
  mkdir -p "$cache_mount"
fi
docker run -d --platform linux/amd64 --name "$name" --tmpfs /tmp:rw,size=256m \
  -v "$cache_mount:/data/models" \
  -e "RAG_SOURCE_REVISION=$revision" -e "RAG_RELEASE_REVISION=$release" \
  "$image" > "$root/embedding-evidence/container-id.txt"
for ((attempt=1; attempt<=130; attempt++)); do
  status=$(docker inspect --format '{{.State.Health.Status}}' "$name")
  if [[ $status == healthy ]]; then
    break
  fi
  if [[ $status == unhealthy ]] || ((attempt == 130)); then
    echo 'Embedding candidate failed readiness' >&2
    exit 1
  fi
  sleep 5
done
docker exec "$name" python -m rag_embedding.verification \
  --url http://127.0.0.1:8080 --source-revision "$revision" --release-revision "$release" \
  > "$root/embedding-evidence/verification.json"
cat "$root/embedding-evidence/verification.json"
