"""Bounded real-model evidence, executed from a RAG container or build runner."""

import argparse
import json
import math
import sys
from urllib.request import Request, urlopen

from rag_embedding.config import MODEL


def verify(base: str, source_revision: str, release_revision: str) -> dict[str, object]:
    def get(path: str) -> dict[str, object]:
        with urlopen(base.rstrip("/") + path, timeout=5) as response:
            result: dict[str, object] = json.load(response)
            return result

    if get("/health")["status"] != "ok":
        raise ValueError("Health failed")
    ready = get("/ready")
    expected = {
        "status": "ready",
        "model": MODEL,
        "dimensions": 1024,
        "source_revision": source_revision,
        "release_revision": release_revision,
    }
    if any(ready.get(key) != value for key, value in expected.items()):
        raise ValueError("Readiness identity mismatch")
    request = Request(
        base.rstrip("/") + "/v1/embeddings",
        data=json.dumps(
            {
                "model": MODEL,
                "input": [
                    "Immutable source evidence.",
                    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: source evidence",
                ],
                "dimensions": 1024,
                "encoding_format": "float",
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    if result["model"] != MODEL or len(result["data"]) != 2:
        raise ValueError("Response identity mismatch")
    for index, item in enumerate(result["data"]):
        vector = item["embedding"]
        if (
            item["index"] != index
            or len(vector) != 1024
            or not all(isinstance(v, float) and math.isfinite(v) for v in vector)
            or not math.isclose(sum(v * v for v in vector), 1, rel_tol=0.001)
        ):
            raise ValueError("Invalid embedding")
    if result["data"][0]["embedding"] == result["data"][1]["embedding"]:
        raise ValueError("Embeddings must depend on input")
    return {
        "status": "passed",
        "model": MODEL,
        "model_revision": ready["model_revision"],
        "dimensions": 1024,
        "batch_size": 2,
        "source_revision": source_revision,
        "release_revision": release_revision,
        "checks": ["health", "ready", "identity", "real_embeddings"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release-revision", required=True)
    args = parser.parse_args()
    try:
        result = verify(args.url, args.source_revision, args.release_revision)
    except Exception:  # noqa: BLE001 -- bounded diagnostic output
        result = {"status": "failed", "reason": "Embedding runtime verification failed"}
        sys.stdout.write(json.dumps(result) + "\n")
        return 1
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
