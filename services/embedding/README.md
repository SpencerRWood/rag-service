# RAG embedding runtime

This independent Python 3.12 module serves `Qwen/Qwen3-Embedding-0.6B` with
Sentence Transformers on CPU. It builds `rag-embedding` separately from the
Python 3.14 RAG application. The primary application and Dagster workers keep
using `EndpointEmbedding`; neither imports this module nor loads model weights.

The Qwen model revision defaults to
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Sentence Transformers, Transformers,
CPU PyTorch and their dependencies are constrained in `pyproject.toml` and
resolved in `uv.lock`. The model loads once during lifespan startup, off the
event loop. Downloads reuse `/data/models`; infrastructure owns its persistent
mount. A first cold start needs access to Hugging Face. Cached restarts reuse
the same model revision; Hugging Face metadata checks have bounded timeouts.

## API

`GET /v1/models` returns the loaded model catalog used by the existing RAG
dependency probe. It returns 503 until readiness succeeds, so the probe cannot
report availability before the model has loaded.

`GET /health` returns `{"status":"ok"}` while the process serves requests,
including during model loading or after loading fails. `GET /ready` returns
503 with `loading` or `failed`, and 200 with `ready` only after the model loads
and its native output dimension is checked. Readiness includes model/revision,
dimensions and deployment source/release identity, without credentials.

`POST /v1/embeddings` accepts:

```json
{
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "input": ["A document passage", "Another passage"],
  "dimensions": 1024,
  "encoding_format": "float"
}
```

`input` may also be a single string. Responses use OpenAI's `object`, ordered
`data` entries (`object`, `index`, `embedding`), `model`, and token `usage` shape.
Each vector has 1024 finite float values and unit norm. Inputs must be non-empty
text: token-ID arrays and base64 output are unsupported. Limits are 256 texts,
32768 characters per text, 262144 characters total and 1 MiB request bodies.
Tokenization truncates each text at `RAG_EMBEDDING_MAX_TOKENS` (8192 by default).
The existing RAG client supplies the Qwen instruction for queries; this runtime
does not add a second prompt to documents or queries.

Errors return `{"error":{"message":"...","type":"embedding_error","code":"..."}}`.
Statuses include 422 for invalid requests, 413 for oversized bodies, 408 for
uploads exceeding 10 seconds, 503 before model readiness, 429 while inference
is busy, 504 for inference timeout, and 500 for invalid output/inference failure.
One request runs at a time, with model batching within that request. The default
inference timeout is 25 seconds, below the RAG client's 30 seconds. CPU work
cannot be forcibly canceled safely: a timed-out request retains its inference
slot until computation finishes, preventing accumulating background work.
Startup is bounded to 600 seconds; failure leaves readiness false. Run one
Uvicorn worker to avoid duplicate model loads. JSON log events contain model
identity, batch count and elapsed time; request text and raw exceptions are
excluded, and Uvicorn access logs are disabled.

## Configuration and checks

Use existing `RAG_EMBEDDING_MODEL`, `RAG_EMBEDDING_DIMENSIONS`,
`RAG_EMBEDDING_TIMEOUT` and `RAG_EMBEDDING_BATCH_SIZE` settings. Runtime-specific
settings are `RAG_EMBEDDING_CACHE_PATH`, `RAG_EMBEDDING_MODEL_REVISION`,
`RAG_EMBEDDING_MAX_TOKENS`, `RAG_EMBEDDING_THREADS` (default 4) and
`RAG_EMBEDDING_STARTUP_TIMEOUT`. Set `RAG_SOURCE_REVISION` and
`RAG_RELEASE_REVISION` for deployment identity. Model and dimension overrides
that violate this service's contract fail configuration validation.

From the repository root:

```sh
bash scripts/check-embedding.sh
docker build --platform linux/amd64 -f services/embedding/Dockerfile -t rag-embedding:candidate .
bash scripts/verify-embedding-image.sh rag-embedding:candidate <source-revision> <release-tag>
```

Unit tests inject an encoder and never download weights. Candidate-image
verification loads the pinned real model and tests health, readiness, identity,
and distinct normalized 1024-dimensional vectors for a document/query batch.
It retains logs, container inspection and bounded verification JSON under the
ignored `embedding-evidence/` directory. Its named model cache survives checks.
The repository pre-commit check includes independent lint, typing and tests;
centralized validation runs that hook. Semantic-release remains the release
authority. The release workflow builds and verifies the second image before
publishing the exact tested candidate and promoting its digest-qualified pin.
The centralized release workflow currently publishes a single primary image,
so only the second image build is repository-owned.

Infrastructure owns deployment, cache permissions, resource limits, networks,
environment values, immutable pins and runtime verification. Both RAG roles use
`RAG_EMBEDDING_ENDPOINT=http://embedding-service:8080/v1` on an internal network.
No public ingress or published host port is required. `localhost` is used only
for probes running inside the embedding container itself.

For first deployment, the verified released digest must be inserted into the
infrastructure embedding pin before its enabling configuration merges. The
existing promotion helper requires an already valid pin, so bootstrap cannot
use an empty/fabricated digest. After bootstrap, subsequent image promotions
use the centralized promotion workflow normally; retry the first promotion
after the verified initial pin is merged.
