"""Source lifecycle HTTP, restart, failure, and concurrent transaction contracts."""

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from uuid import UUID, uuid4

import boto3
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from moto import mock_aws
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from rag_service.api.routes.documents import original_storage
from rag_service.config import Settings
from rag_service.main import create_app
from rag_service.models.persistence import Document, DocumentVersion
from rag_service.services.documents import (
    DocumentConflictError,
    DocumentNotFoundError,
    VersionStatus,
    source_versions,
    transition_version,
    upload_original,
)
from rag_service.storage import FileStorage, build_storage


@dataclass
class Harness:
    """One migrated application and its configured original store."""

    settings: Settings
    app: FastAPI
    client: TestClient
    storage: FileStorage
    base: str

    def transition(self, document: str, version: str, target: VersionStatus) -> None:
        with self.app.state.session_factory() as session:
            transition_version(
                session,
                UUID(self.base.split("/")[2]),
                UUID(document),
                UUID(version),
                target,
            )


@pytest.fixture(params=["filesystem", "s3"])
def harness(
    request: pytest.FixtureRequest,
    migrated_database: str,
    tmp_path: Path,
) -> Iterator[Harness]:
    with mock_aws():
        if request.param == "s3":
            boto3.client("s3", region_name="us-east-1").create_bucket(
                Bucket="originals"
            )
        settings = Settings(
            database_url=SecretStr(migrated_database),
            storage_path=tmp_path / "originals",
            storage_backend=request.param,
            s3_bucket="originals",
            max_upload_bytes=1024,
        )
        app = create_app(settings)
        with TestClient(app) as client:
            knowledge_base = client.post(
                "/knowledge-bases", json={"name": "Library"}
            ).json()["id"]
            yield Harness(
                settings,
                app,
                client,
                build_storage(settings),
                f"/knowledge-bases/{knowledge_base}/documents",
            )


def upload(
    harness: Harness, content: bytes = b"source", suffix: str = ""
) -> dict[str, object]:
    response = harness.client.post(
        harness.base + suffix,
        files={"file": ("source.bin", content, "application/octet-stream")},
    )
    assert response.status_code in {200, 201}
    return response.json()  # type: ignore[no-any-return]


def test_exact_original_metadata_duplicates_scope_and_restart(harness: Harness) -> None:
    content = bytes(range(256)) + b"\x00\xff\r\n"
    metadata = {
        "title": "Manual",
        "tags": ["engineering"],
        "source": "connector://manual",
        "connector_metadata": {"external_id": "42", "nested": [True, None, 7]},
    }
    response = harness.client.post(
        harness.base,
        files={"file": ("manual.bin", content)},
        data={"metadata": json.dumps(metadata)},
    )
    assert response.status_code == 201
    first = response.json()
    document, version = first["document"], first["version"]
    for name, value in metadata.items():
        assert document[name] == value
    assert document["status"] == "pending"
    assert document["active_version_id"] is None
    assert version["original_stored"] is True
    assert "storage_key" not in version
    duplicate = upload(harness, content)
    assert duplicate["created"] is False
    assert duplicate["document"] == document
    assert duplicate["version"] == version
    path = f"{harness.base}/{document['id']}/versions/{version['id']}"
    assert harness.client.get(path).json() == version
    assert harness.client.get(path + "/original").content == content
    assert harness.client.get(harness.base).json() == [document]
    other = harness.client.post("/knowledge-bases", json={"name": "Other"}).json()["id"]
    independently_stored = harness.client.post(
        f"/knowledge-bases/{other}/documents", files={"file": ("manual.bin", content)}
    ).json()
    assert independently_stored["document"]["id"] != document["id"]
    restarted = create_app(harness.settings)
    with TestClient(restarted) as client:
        assert client.get(path + "/original").content == content
        retry = client.post(harness.base, files={"file": ("retry.bin", content)}).json()
        assert retry["document"]["id"] == document["id"]
        assert retry["version"]["id"] == version["id"]


def test_versions_active_readiness_failed_replacement_and_late_completion(
    harness: Harness,
) -> None:
    first = harness.client.post(
        harness.base, files={"file": ("first", b"first")}
    ).json()
    document_id, first_id = first["document"]["id"], first["version"]["id"]
    with pytest.raises(DocumentConflictError, match="Invalid"):
        harness.transition(document_id, first_id, "ready")
    harness.transition(document_id, first_id, "processing")
    harness.transition(document_id, first_id, "ready")
    harness.transition(document_id, first_id, "ready")
    second = harness.client.post(
        harness.base + f"/{document_id}/versions",
        files={"file": ("changed", b"changed")},
    ).json()
    second_id = second["version"]["id"]
    assert second["version"]["number"] == 2
    assert second["document"]["active_version_id"] == first_id
    harness.transition(document_id, second_id, "processing")
    harness.transition(document_id, second_id, "failed")
    failed = harness.client.get(harness.base + f"/{document_id}").json()
    assert failed["active_version_id"] == first_id
    assert failed["latest_version_id"] == second_id
    assert failed["status"] == "failed"
    repeated = harness.client.post(
        harness.base + f"/{document_id}/versions",
        files={"file": ("changed", b"changed")},
    ).json()
    assert repeated["created"] is False
    assert repeated["version"]["status"] == "failed"
    third = harness.client.post(
        harness.base + f"/{document_id}/versions", files={"file": ("newest", b"newest")}
    ).json()
    third_id = third["version"]["id"]
    harness.transition(document_id, third_id, "processing")
    harness.transition(document_id, third_id, "ready")
    harness.transition(document_id, second_id, "processing")
    harness.transition(document_id, second_id, "ready")
    assert (
        harness.client.get(harness.base + f"/{document_id}").json()["active_version_id"]
        == third_id
    )
    history = harness.client.get(harness.base + f"/{document_id}/versions").json()
    assert [version["number"] for version in history] == [3, 2, 1]
    for version, content in zip(
        history, [b"newest", b"changed", b"first"], strict=True
    ):
        assert (
            harness.client.get(
                harness.base + f"/{document_id}/versions/{version['id']}/original"
            ).content
            == content
        )
    assert harness.client.get(
        harness.base + f"/{document_id}/versions?limit=1&offset=1"
    ).json() == [history[1]]


def test_metadata_updates_and_soft_deletion_retain_history(harness: Harness) -> None:
    record = harness.client.post(
        harness.base, files={"file": ("source", b"source")}
    ).json()
    document_id, version_id = record["document"]["id"], record["version"]["id"]
    path = harness.base + f"/{document_id}"
    updated = harness.client.patch(
        path,
        json={
            "title": "Updated",
            "tags": ["new"],
            "source": "origin",
            "connector_metadata": {"key": 1},
        },
    )
    assert updated.status_code == 200
    assert updated.json()["title"] == "Updated"
    assert harness.client.patch(path, json={"source": None}).json()["tags"] == ["new"]
    assert harness.client.patch(path, json={"title": None}).status_code == 422
    assert harness.client.delete(path).status_code == 204
    deleted = harness.client.get(path + "?include_deleted=true").json()
    assert deleted["status"] == "deleted"
    assert deleted["deleted_at"] is not None
    assert harness.client.delete(path).status_code == 204
    assert harness.client.get(path + "?include_deleted=true").json() == deleted
    assert harness.client.get(harness.base).json() == []
    assert harness.client.get(harness.base + "?include_deleted=true").json() == [
        deleted
    ]
    assert harness.client.get(path).status_code == 404
    assert harness.client.get(path + "/versions").status_code == 404
    assert (
        harness.client.get(path + "/versions?include_deleted=true").json()[0]["id"]
        == version_id
    )
    assert (
        harness.client.get(
            path + f"/versions/{version_id}/original?include_deleted=true"
        ).content
        == b"source"
    )
    assert (
        harness.client.post(
            harness.base, files={"file": ("source", b"source")}
        ).status_code
        == 409
    )
    assert (
        harness.client.post(
            path + "/versions", files={"file": ("source", b"new")}
        ).status_code
        == 404
    )
    assert harness.client.patch(path, json={"title": "Restore"}).status_code == 404
    with pytest.raises(DocumentNotFoundError):
        harness.transition(document_id, version_id, "processing")


def test_scoping_conflicts_missing_identities_and_input_validation(
    harness: Harness,
) -> None:
    first = harness.client.post(
        harness.base, files={"file": ("source", b"first")}
    ).json()
    second = harness.client.post(
        harness.base, files={"file": ("source", b"second")}
    ).json()
    document_id, version_id = first["document"]["id"], first["version"]["id"]
    assert document_id != second["document"]["id"]
    path = harness.base + f"/{document_id}"
    assert (
        harness.client.post(
            path + "/versions", files={"file": ("source", b"second")}
        ).status_code
        == 409
    )
    assert (
        harness.client.get(path + f"/versions/{second['version']['id']}").status_code
        == 404
    )
    assert harness.client.get(harness.base + f"/{uuid4()}").status_code == 404
    assert harness.client.get(path + f"/versions/{uuid4()}").status_code == 404
    other = harness.client.post("/knowledge-bases", json={"name": "Other"}).json()["id"]
    assert (
        harness.client.get(
            f"/knowledge-bases/{other}/documents/{document_id}/versions/{version_id}"
        ).status_code
        == 404
    )
    missing = f"/knowledge-bases/{uuid4()}/documents"
    assert harness.client.get(missing).status_code == 404
    assert (
        harness.client.post(missing, files={"file": ("source", b"x")}).status_code
        == 404
    )
    for metadata in ["not-json", '{"title":" "}', '{"extra":1}', '{"tags":[""]}']:
        assert (
            harness.client.post(
                harness.base,
                files={"file": ("source", b"x")},
                data={"metadata": metadata},
            ).status_code
            == 422
        )
    assert (
        harness.client.post(
            harness.base, files={"file": ("source", b"x" * 1025)}
        ).status_code
        == 413
    )
    assert (
        harness.client.post(
            harness.base, files={"file": ("source", b"x", "bad\r\nheader")}
        ).status_code
        == 400
    )
    assert (
        harness.client.post(harness.base, files={"file": ("x" * 256, b"x")}).status_code
        == 422
    )
    assert harness.client.get(harness.base + "?limit=101").status_code == 422
    assert (
        harness.client.post(
            harness.base, files={"file": ("source", b"x", "text/☃")}
        ).status_code
        == 422
    )
    assert harness.client.get(path + "/versions?offset=-1").status_code == 422
    empty = harness.client.post(harness.base, files={"file": ("empty", b"")})
    assert empty.status_code == 201
    assert empty.json()["version"]["size"] == 0


class BrokenStorage:
    """Inject provider failures without including provider details in responses."""

    def save(self, _key: str, _content: bytes) -> None:
        raise OSError("private provider failure")

    def read(self, _key: str) -> bytes:
        raise OSError("private provider failure")

    def exists(self, _key: str) -> bool:
        return False

    def delete(self, _key: str) -> None:
        raise OSError("private provider failure")


def test_storage_failures_retain_identity_and_safe_retry(harness: Harness) -> None:
    first = harness.client.post(
        harness.base, files={"file": ("first", b"first")}
    ).json()
    document_id, first_id = first["document"]["id"], first["version"]["id"]
    harness.transition(document_id, first_id, "processing")
    harness.transition(document_id, first_id, "ready")
    path = harness.base + f"/{document_id}"
    harness.app.dependency_overrides[original_storage] = BrokenStorage
    response = harness.client.post(
        path + "/versions", files={"file": ("changed", b"changed")}
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Original storage is unavailable"}
    failed = harness.client.get(path + "/versions").json()[0]
    assert failed["status"] == "failed"
    assert failed["original_stored"] is False
    assert harness.client.get(path).json()["active_version_id"] == first_id
    assert (
        harness.client.get(path + f"/versions/{failed['id']}/original").status_code
        == 503
    )
    assert (
        harness.client.get(path + f"/versions/{first_id}/original").status_code == 503
    )
    with pytest.raises(DocumentConflictError):
        harness.transition(document_id, failed["id"], "processing")
    harness.app.dependency_overrides.clear()
    retry = harness.client.post(
        path + "/versions", files={"file": ("changed", b"changed")}
    )
    assert retry.status_code == 200
    assert retry.json()["version"]["id"] == failed["id"]
    assert retry.json()["version"]["status"] == "pending"
    assert (
        harness.client.get(path + f"/versions/{failed['id']}/original").content
        == b"changed"
    )
    with harness.app.state.session_factory() as session:
        stored = session.get(DocumentVersion, UUID(first_id))
        assert stored is not None
        harness.storage.delete(stored.storage_key)
    assert (
        harness.client.get(path + f"/versions/{first_id}/original").status_code == 503
    )


def test_simultaneous_uploads_and_replacements_are_idempotent(harness: Harness) -> None:
    barrier = Barrier(4)

    def simultaneous(suffix: str = "") -> tuple[str, str]:
        barrier.wait(timeout=10)
        record = harness.client.post(
            harness.base + suffix, files={"file": ("source", b"same")}
        )
        assert record.status_code in {200, 201}
        return record.json()["document"]["id"], record.json()["version"]["id"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(lambda _: simultaneous(), range(4)))
    assert len(set(identities)) == 1
    document_id = identities[0][0]
    barrier.reset()

    # A changed source also has one durable version across concurrent requests.
    def replacement(_: int) -> str:
        barrier.wait(timeout=10)
        result = harness.client.post(
            harness.base + f"/{document_id}/versions",
            files={"file": ("new", b"replacement")},
        )
        assert result.status_code in {200, 201}
        return str(result.json()["version"]["id"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        versions = list(pool.map(replacement, range(4)))
    assert len(set(versions)) == 1
    assert len(harness.client.get(harness.base).json()) == 1
    assert (
        len(harness.client.get(harness.base + f"/{document_id}/versions").json()) == 2
    )


def test_crash_after_storage_before_commit_is_retry_safe(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted_commit(_session: Session) -> None:
        raise RuntimeError("Simulated process interruption")

    knowledge_base_id = UUID(harness.base.split("/")[2])
    with harness.app.state.session_factory() as session:
        monkeypatch.setattr(session, "commit", lambda: interrupted_commit(session))
        with pytest.raises(RuntimeError, match="interruption"):
            upload_original(
                session,
                harness.storage,
                knowledge_base_id,
                b"interrupted",
                "source",
                "application/octet-stream",
            )
        session.rollback()
    assert harness.client.get(harness.base).json() == []
    record = harness.client.post(
        harness.base, files={"file": ("source", b"interrupted")}
    ).json()
    document_id, version_id = record["document"]["id"], record["version"]["id"]
    with harness.app.state.session_factory() as session:
        assert len(source_versions(session, UUID(document_id))) == 1
    assert (
        harness.client.get(
            harness.base + f"/{document_id}/versions/{version_id}/original"
        ).content
        == b"interrupted"
    )


def test_database_uniqueness_survives_bypassing_request_checks(
    harness: Harness,
) -> None:
    record = harness.client.post(
        harness.base, files={"file": ("source", b"source")}
    ).json()
    version_id = UUID(record["version"]["id"])
    with harness.app.state.session_factory() as session:
        original = session.get(DocumentVersion, version_id)
        assert original is not None
        other = harness.client.post(
            "/knowledge-bases", json={"name": "Foreign"}
        ).json()["id"]
        mutations: list[dict[str, object]] = [
            {"number": 2},
            {"checksum": "f" * 64},
            {"number": 2, "checksum": "e" * 64, "knowledge_base_id": UUID(other)},
            {"number": 2, "checksum": "d" * 64, "status": "unknown"},
            {
                "number": 2,
                "checksum": "c" * 64,
                "status": "ready",
                "original_stored": False,
            },
        ]
        for changes in mutations:
            values = {
                column.name: getattr(original, column.name)
                for column in DocumentVersion.__table__.columns
                if column.name != "id"
            }
            values.update(changes)
            values["storage_key"] = f"unique/{uuid4()}"
            session.add(DocumentVersion(**values))
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()
        assert len(list(session.scalars(select(Document)))) == 1


def test_first_upload_storage_failure_can_be_retried_after_restart(
    harness: Harness,
) -> None:
    harness.app.dependency_overrides[original_storage] = BrokenStorage
    failed = harness.client.post(harness.base, files={"file": ("source", b"initial")})
    assert failed.status_code == 503
    document = harness.client.get(harness.base).json()[0]
    assert document["status"] == "failed"
    version_id = document["latest_version_id"]
    restarted = create_app(harness.settings)
    with TestClient(restarted) as client:
        repaired = client.post(harness.base, files={"file": ("retry", b"initial")})
        assert repaired.status_code == 200
        assert repaired.json()["document"]["id"] == document["id"]
        assert repaired.json()["version"]["id"] == version_id
        assert repaired.json()["version"]["original_stored"] is True
        assert repaired.json()["version"]["status"] == "pending"


def test_simultaneous_changed_sources_have_distinct_ordered_versions(
    harness: Harness,
) -> None:
    record = harness.client.post(
        harness.base, files={"file": ("source", b"initial")}
    ).json()
    document_id = record["document"]["id"]
    barrier = Barrier(4)

    def replacement(number: int) -> int:
        barrier.wait(timeout=10)
        result = harness.client.post(
            harness.base + f"/{document_id}/versions",
            files={"file": ("source", f"revision-{number}".encode())},
        )
        assert result.status_code == 201
        return int(result.json()["version"]["number"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        numbers = list(pool.map(replacement, range(4)))
    assert sorted(numbers) == [2, 3, 4, 5]
    assert (
        len(harness.client.get(harness.base + f"/{document_id}/versions").json()) == 5
    )
