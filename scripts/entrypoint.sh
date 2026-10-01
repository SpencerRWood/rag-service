#!/bin/sh
set -eu

# The shared candidate gate selects the code-server role with this variable.
if [ -n "${DAGSTER_GRPC_PORT:-}" ]; then
    exec dagster api grpc -m rag_service.dagster.definitions \
        -h 0.0.0.0 -p "$DAGSTER_GRPC_PORT"
fi

exec "$@"
