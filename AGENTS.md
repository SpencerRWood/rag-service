# Repository Guidance

## Purpose and boundaries
- Build the source-backed document ingestion and retrieval API in `src/rag_service/`.
- Keep the FastAPI foundation distinct from planned storage, ingestion, and indexing work described in `README.md`.

## Development
- Preserve source provenance and document-version semantics when adding retrieval behavior.
- Put public HTTP behavior in the service package and focused checks in `tests/`.

## Validation
- Run `uv sync --frozen --group dev`, then `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, and `uv run pytest`.
- Run configured pre-commit hooks for a full repository check.

## Architecture constraints
- Keep storage and embedding providers configurable; do not couple API contracts to one local backend.
- Use Alembic for persistent schema changes when the database layer is present.

## Safety
- Keep source documents, credentials, and provider tokens out of Git and test output.

## Branching strategy

- This repository uses **release-branch integration**. Create Story branches from the verified active `release/vX.Y.Z` and target Story PRs there; `main` changes only after the release readiness gate.
- Follow the [canonical branching and delivery policy](https://github.com/SpencerRWood/codex-config/blob/main/docs/branching-policy.md), available through the installed codex-config source path. Verify the strategy, release association, fetched base and PR target before implementation and record them in Story evidence. Story completion does not authorize publication or deployment; retain existing review, CI, security and rollback requirements.
