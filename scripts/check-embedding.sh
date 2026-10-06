#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --project services/embedding --frozen --group dev
uv run --project services/embedding ruff check services/embedding
uv run --project services/embedding ruff format --check services/embedding
uv run --project services/embedding mypy services/embedding/src services/embedding/tests
uv run --project services/embedding pytest services/embedding/tests
