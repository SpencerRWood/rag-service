# RAG Service

Source-backed retrieval API for knowledge-base documents. The service will store
original source files, process durable document versions asynchronously, index
provenance-rich chunks, and expose retrieval over HTTP.

## Initial architecture

- FastAPI API with a liveness endpoint at `/health`
- PostgreSQL/pgvector and Alembic for persistent state and vector indexing
- Dagster for durable ingestion and reprocessing jobs
- Configurable filesystem or S3-compatible source storage
- Local Qwen3 embeddings by default, with optional OpenRouter overrides

The foundation implements knowledge-base configuration, versioned tenant and
knowledge-base persistence, portable original-file storage, and an importable
Dagster code location. The OpenProject R1 backlog adds document lifecycle,
processing, retrieval, MCP, and operational diagnostics in subsequent stories.

The repository follows `SpencerRWood/template-fastapi-service`. Its Dagster
`Definitions`, asset, asset job, typed resource, and secret-free runtime smoke
job follow `SpencerRWood/template-python-dagster`, including the same pinned
Dagster/PostgreSQL/SQLAlchemy/psycopg2 dependency set. Schedule and sensor
modules are extension points; no recurring workload is enabled yet.

## Local development

```sh
cp .env.example .env
uv sync --frozen --group dev
uv run --env-file .env alembic upgrade head
uv run --env-file .env uvicorn rag_service.main:app --reload
```

Run the baseline checks:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## Configuration

Copy `.env.example` to an uncommitted `.env` for local development and supply
a PostgreSQL connection URL. Deployment injects environment variables at
runtime; the application does not read a secret file or run migrations on
startup. Apply `alembic upgrade head` as an infrastructure-owned deployment
step. Application image rollback retains the database; destructive schema
downgrade is refused.

Defaults use filesystem/NAS storage, local Qwen3-Embedding-0.6B embeddings
with 1024 dimensions, and chunks of 512 with overlap 64. S3-compatible storage
uses `RAG_S3_BUCKET`, optional endpoint/region, and the SDK runtime credential
chain. OpenRouter is selectable by setting the provider, model, dimensions,
endpoint, and runtime API key. Secrets are redacted in settings and omitted
from public configuration responses.

`POST /knowledge-bases` creates a knowledge base with effective service
defaults or explicit overrides. `GET /knowledge-bases` accepts bounded
`limit`/`offset` pagination; `GET /knowledge-bases/{id}` retrieves its persisted
configuration. Requests use the migrated default tenant. `/health` is liveness
only; `/version` exposes package version and non-sensitive
`RAG_SOURCE_REVISION`/`RAG_RELEASE_REVISION` supplied by infrastructure. Missing
revision values are reported as `unknown`.

## Dagster code location

The same application image supports both roles. Its default command starts
the API. Setting `DAGSTER_GRPC_PORT` starts the code server using
`rag_service.dagster.definitions`; infrastructure owns the shared Dagster
daemon, webserver, database, registration, and runtime placement. Locally:

```sh
uv run dagster api grpc -m rag_service.dagster.definitions -h 0.0.0.0 -p 4000
```

`configuration_job` proves asset/resource configuration without processing
documents. `runtime_smoke_job` uses the template's in-process executor and
memory IO manager, with no application secrets or external API dependencies.

## Delivery and tracking

`pyproject.toml` maps Wood Tools to Wood Platform project **3** and RAG Service
initiative **418**. Core Retrieval is epic **419**, in planning version R1
**20**. OpenProject credentials are tooling inputs supplied through Infisical,
not application runtime settings. From this repository:

```sh
infisical run --env=dev --path=/openproject -- wood story next --json
wood repo validate --json
```

If this checkout has no Infisical context, use the existing authorized tooling
context with `--project-config-dir`; do not copy its secrets into the repo.

PR validation calls `validate.yml@v3`; release calls the centralized
`release-container.yml@v3` contract. A candidate image must pass the shared
PostgreSQL-backed Dagster runtime gate before semantic tags/GitHub Release
publication. Dev promotion consumes that validated digest and targets the
infrastructure-owned `rag_service_image_ref` pin. Infrastructure service setup,
runtime secrets, and the `INFRASTRUCTURE_PR_TOKEN` must be provisioned there
before promotion can succeed. Production deployment remains outside R1.

The storage contract suite uses the same tests for filesystem and an isolated
S3 emulator. Migration/API tests use SQLite by default; set
`RAG_TEST_DATABASE_URL` to a disposable PostgreSQL database to exercise the
same suite with isolated per-test schemas. Never point this at production.
