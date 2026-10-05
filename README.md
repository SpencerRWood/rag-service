# RAG Service

Source-backed retrieval API for knowledge-base documents. The service will store
original source files, process durable document versions asynchronously, index
provenance-rich chunks, and expose retrieval over HTTP and read-only MCP.

## Initial architecture

- FastAPI API with a liveness endpoint at `/health`
- PostgreSQL/pgvector and Alembic for persistent state and vector indexing
- Dagster for durable ingestion and reprocessing jobs
- Configurable filesystem or S3-compatible source storage
- Local Qwen3 embeddings by default, with optional OpenRouter overrides

The service implements knowledge-base configuration, durable document/source
version history, portable original-file storage, asynchronous parsing/chunking
and embedding, pgvector retrieval over HTTP, and an importable Dagster code
location and a read-only MCP adapter. Operational diagnostics remain later backlog work.

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

## Document lifecycle

`POST /knowledge-bases/{kb}/documents` accepts multipart `file` and an optional
`metadata` JSON form field containing `title`, `tags`, `source`, and
`connector_metadata`. The response contains `document`, `version`, `created`,
`generation_id`, `dagster_run_id`, and `launch_submitted`.
An identical SHA-256 checksum within the knowledge base returns the existing
document and source version with HTTP 200; new content returns HTTP 201. Identical
content in another knowledge base creates an independent identity. Filenames do
not determine identity, and retries preserve existing metadata and lifecycle state.

`POST /knowledge-bases/{kb}/documents/{doc}/versions` uploads changed source bytes
for an explicit document. Each distinct source gets a monotonically increasing
version number; retrying any historical source returns its original version ID.
Content already owned by another document in the knowledge base returns HTTP 409.
`PATCH /knowledge-bases/{kb}/documents/{doc}` updates only supplied metadata fields.

Document list/get endpoints expose `latest_version_id`, `active_version_id`, and
the latest attempted version's `status`. Version list/get endpoints expose
checksum, source filename, byte size, and lifecycle state. Lists support bounded
`limit`/`offset` pagination. Download exact original bytes with
`GET /knowledge-bases/{kb}/documents/{doc}/versions/{version}/original`.

Uploads store originals and enter `pending`; they are not declared successfully
processed by HTTP callers. The internal `transition_version` boundary supports
`pending → processing → ready/failed` and explicit `failed → processing` retries.
The active source is the highest numbered successfully indexed version. Failed,
pending, or late-finishing older versions cannot displace a newer indexed source.
Dagster jobs own parsing, chunking, and embedding. New processing generations
become `ready` only when their chunks and vectors are complete. Legacy #422
parsed-only generations retain their history and status but have a null
`index_generation_id`; they are excluded from active/retrieval selection until
explicitly reprocessed through the existing endpoint. No automatic backfill runs.

### Asynchronous processing

Uploads reserve a processing generation and submit `ingestion_job` through the
shared Dagster GraphQL endpoint. The API makes bounded metadata requests and does
not parse, chunk, or invoke models. If submission is unavailable or its response
is lost, the original and launch reservation remain durable; `launch_submitted`
is false and `dagster_run_id` is null until recovery. The enabled
`recover_launches` sensor repairs interrupted reservations and submissions in
batches of 50. This is recovery of explicit API operations, not connector scheduling.
`RAG_DAGSTER_URL` and `RAG_DAGSTER_LOCATION` select the shared control plane and
registered code location. Dagster assigns run IDs, persists their states/events,
and executes jobs outside the API process. Run config/tags contain opaque IDs.

`POST /knowledge-bases/{kb}/documents/{doc}/versions/{version}/reprocess` accepts
`{"reparse": false}` (or `{}`) and a required `Idempotency-Key` header. A new key
reserves a new derived generation and snapshots effective knowledge-base chunk
settings; it never creates another source version. Repeating the key returns the
same generation/run. Reusing it with different options returns HTTP 409. Stored
parsed content is reused by default; `reparse: true` explicitly selects a new parse.

`POST /knowledge-bases/{kb}/documents/{doc}/versions/{version}/generations/{generation}/retry`
also requires `Idempotency-Key` and retries a failed generation without re-upload.
The same retry key returns the same attempt even after completion. A new retry
key requires failed state. Both endpoints return HTTP 202 with `generation`,
`dagster_run_id`, and `launch_submitted`.
`GET /knowledge-bases/{kb}/documents/{doc}/versions/{version}/generations` returns
bounded generation history, status, parse checkpoint, timestamps, safe error code,
and snapshotted chunk settings. Document reads additionally report
`active_processing_generation_id`, the highest numbered ready generation of the
active source. Dagster's UI/API supplies the durable orchestration run state.

PDF uses pypdf; DOCX uses python-docx; HTML uses Beautiful Soup; CSV uses the
standard csv module; XLSX uses openpyxl; Markdown, plain text, and source code use
strict UTF-8 decoding with section/line extraction. Parsed segments and parser
name/version/time are stored separately from chunks. LlamaIndex consumes those
normalized segments with a deterministic whitespace tokenizer: chunk size and
overlap count whitespace tokens, with splits bounded by source units. Every
CSV/XLSX row chunk repeats its table header and retains sheet, row and column
context; repeated headers are additional context outside the body token budget.
Other chunks retain page, section, paragraph, or source path/line range when
available. Empty/scanned PDFs without extractable text fail safely; OCR is outside
this Story. Originals with unsupported binary/text encodings remain downloadable.

Processing uses the existing knowledge-base mutation lock for each stage. Parsing
commits a reusable checkpoint; chunks, embeddings, and ready state commit atomically.
Interrupted uncommitted work rolls back. Duplicate runs and automatic Dagster step
retries reuse the same generation, and completed work is a no-op. Failed source
versions or reprocessing never clear previous ready source/generation state.
The recovery sensor reconciles terminal failed/canceled Dagster runs into retryable
application failures. Infrastructure enables shared Dagster run monitoring so dead
workers become terminal runs. A queued run remains pending while its daemon is
unavailable. Per-knowledge-base serialization trades ingestion throughput for
transactional safety; processing can delay other mutations in that knowledge base.

`DELETE /knowledge-bases/{kb}/documents/{doc}` is retry-safe soft deletion.
Normal reads, uploads to that document, and processing exclude deleted identities;
explicit `include_deleted=true` on reads exposes retained metadata, versions, and
originals. Checksums remain reserved: re-uploading a deleted document's content
returns HTTP 409 rather than recreating it. Restoration is outside this API.

Mutations serialize on a knowledge-base database row, with uniqueness and scope
constraints guarding source identity. The lock spans storage and commit, trading
per-knowledge-base upload throughput for simple retry safety. Deterministic storage
keys tolerate interruption between storage and commit. Failed storage retains a
failed source reservation, and an identical retry repairs it without creating a
new version. Such failures return a safe HTTP 503; previous ready sources remain
active. An interrupted uncommitted write may leave an unreferenced original;
garbage collection is outside this story. `RAG_MAX_UPLOAD_BYTES` bounds the bytes
read into application memory (default 25 MiB); oversized originals return HTTP 413.
Infrastructure should also cap request sizes before multipart parsing/spooling.

## Embedding provider contract

The default provider calls a separately deployed Qwen3-Embedding-0.6B endpoint
using 1024-dimensional vectors. Neither the API nor the code location loads model
weights or starts an embedding server. Infrastructure chooses the runtime, host,
accelerator, and network placement. Configure `RAG_EMBEDDING_ENDPOINT` as an
OpenAI-compatible API base URL, including `/v1` (default
`http://localhost:8080/v1`). Both document batches and queries use
`POST {base}/embeddings` with `model`, `input`, `dimensions`, and float encoding.
For a vLLM Qwen endpoint, serve `Qwen/Qwen3-Embedding-0.6B` in pooling mode and
set its served model name to `Qwen3-Embedding-0.6B` to match the configured alias.
The local adapter leaves document text unchanged and formats queries with the
[Qwen retrieval instruction](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

Select OpenRouter by configuring `RAG_EMBEDDING_PROVIDER=openrouter`, a supported
embedding model/dimension, `RAG_EMBEDDING_ENDPOINT=https://openrouter.ai/api/v1`,
and the runtime `RAG_EMBEDDING_API_KEY`. Its adapter uses the same provider
interface and sends `search_document` or `search_query` input types through
the [OpenRouter embeddings API](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request).
There is no fallback to another provider or to generative inference. Credentials
never enter generation records or Dagster run config. Provider errors expose safe
application failures, and `/health` stays independent of provider availability.

`RAG_EMBEDDING_TIMEOUT` bounds each provider request (default 30 seconds), and
`RAG_EMBEDDING_BATCH_SIZE` bounds batches (default 32, maximum 256). Responses
must match the configured model, contain exactly one vector for each input, and
use finite, nonzero vectors of the configured dimension. Response indices restore
input order before storage. Provider request failures, malformed responses, and
interrupted batches roll back all derived artifacts for that processing stage.

The first processing reservation pins one index generation for the knowledge
base. Each processing generation references that embedding space; each indexed
chunk records provider, model, dimensions, index generation, processing generation,
and creation time. Database foreign keys prevent mixed identities, and PostgreSQL
checks vector dimensions. Existing generations and queries use the pinned identity
even if service defaults change. Changing a knowledge base's provider/model/dimension
requires a replacement index and is rejected with HTTP 409; automatic replacement
build/promotion is outside #423. Original versions and previously ready generations
remain available. The embedding endpoint is snapshotted with the generation.

## HTTP retrieval

`POST /knowledge-bases/{knowledge_base_id}/retrieve` accepts:

```json
{
  "query": "How is source content versioned?",
  "result_count": 5,
  "filters": [
    {"field": "source", "operator": "eq", "value": "manual"},
    {"field": "tag", "operator": "in", "value": ["reference", "architecture"]}
  ]
}
```

The query is nonempty and at most 8192 characters; `result_count` defaults to 5
and accepts 1–100. Optional `version_id` selects one source version within this
knowledge base. Normal queries select the highest numbered indexed source of each
non-deleted document, then its highest numbered completed indexed processing
generation. Pending or failed work never displaces ready retrieval state, and
late completion of an older source/generation cannot displace a newer one.

Filters support `title`, `source`, and `tag`, with case-sensitive `eq` or `in`.
`tag` tests individual tag membership; `in` accepts any value in its nonempty
list. Multiple filters combine with AND before ranking and result limiting.
At most 20 filters and 100 values per IN filter are accepted. Filtering uses current
document metadata, including for historical versions. Arbitrary connector JSON
predicates and other operators are deferred. Unsupported fields/operators and
invalid request shapes return HTTP 422.

```json
{
  "knowledge_base_id": "00000000-0000-0000-0000-000000000010",
  "results": [
    {
      "chunk_id": "00000000-0000-0000-0000-000000000011",
      "document_id": "00000000-0000-0000-0000-000000000012",
      "version_id": "00000000-0000-0000-0000-000000000013",
      "text": "Source versions preserve the exact original bytes.",
      "score": 0.91,
      "source": {
        "filename": "guide.md",
        "media_type": "text/markdown",
        "checksum": "<source SHA-256>",
        "location": {"section": "Versioning", "line_start": 4, "line_end": 8}
      },
      "metadata": {
        "title": "Guide",
        "tags": ["reference"],
        "source": "manual",
        "connector_metadata": {}
      }
    }
  ]
}
```

All shown response fields are stable and required; metadata `source` may be null,
and `location` contains the available parser provenance (page, section, sheet,
row/column context, or source path/line range). PostgreSQL uses exact pgvector
cosine search: scores range from -1 to 1, higher is better, with chunk ID breaking
ties. An empty index or no matching indexed chunks returns HTTP 200 with an empty
`results` array. Unknown knowledge bases, cross-base versions, and deleted versions
return HTTP 404. Unavailable embedding or persistence dependencies return safe
HTTP 503 errors. SQLite provides only a deterministic development/test adapter;
deployed similarity search runs in PostgreSQL/pgvector. No approximate index,
hybrid search, reranking, answer generation, or connectors are added here.
LlamaIndex orchestrates embedding and retrieval through the reusable service layer;
HTTP clients depend on this contract and `/openapi.json`, not the database schema.

Apply Alembic revision `0004` before running this release. PostgreSQL must have
pgvector installed and the migration role must be able to create its `vector`
extension in `public` (or an administrator can provision it first). Custom database
search paths must include `public`. Schema downgrade remains refused.

## Read-only MCP retrieval

The API hosts a stateless Streamable HTTP MCP endpoint at `/mcp`, using the
[official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
Clients initialize an MCP session, list tools, then call:

```json
{
  "name": "search",
  "arguments": {
    "knowledge_base_id": "00000000-0000-0000-0000-000000000010",
    "request": {"query": "How is source content versioned?", "result_count": 5}
  }
}
```

`request` is the HTTP `RetrievalRequest`, including the same metadata filters,
limits, and optional historical `version_id`. Structured output is exactly the
HTTP `RetrievalResponse`; the SDK also emits JSON text for clients reading text
content. Call `fetch` with `knowledge_base_id` and a search result's `chunk_id`
to read that exact indexed chunk, source locator, checksum, and current metadata.
HTTP exposes the same fetch contract at
`GET /knowledge-bases/{knowledge_base_id}/chunks/{chunk_id}`. Fetch omits the
query-dependent similarity score and never calls an embedding provider. Retained
completed chunks remain readable after newer versions or reprocessing; unindexed,
pending, failed, deleted, or cross-base evidence cannot be fetched.

Only `search` and `fetch` are registered, both annotated read-only, non-destructive,
and idempotent. No resources, prompts, write tools, or generative inference are
exposed. Search uses the existing embedding runtime; no generative-model
credential or particular consumer account entitlement is required. Invalid calls
produce MCP tool errors, and dependency errors expose safe messages.

Set `RAG_MCP_PATH` to change the absolute endpoint path. Set
`RAG_MCP_ALLOWED_HOSTS` and `RAG_MCP_ALLOWED_ORIGINS` as JSON arrays for the
infrastructure-owned hostname and permitted browser origins. Defaults allow
loopback hosts/origins with ports; DNS rebinding protection remains enabled.
Infrastructure owns network exposure, TLS, and access controls; authentication
and authorization are outside #424. The endpoint shares the API's configured
database and lifecycle, and startup does not migrate or contact model providers.

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
memory IO manager. Inside the centralized candidate-image gate it exercises
parsing, provenance, vector persistence, transactions, and completed-operation replay in a disposable
schema using the gate's PostgreSQL credentials. Local smoke tests use temporary
SQLite. It never reads the real RAG database/originals or calls model providers.

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
