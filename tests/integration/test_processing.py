"""Durable Dagster processing, concurrent replay, parse checkpoints and HTTP retry."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import pytest
from dagster import DagsterInstance, DagsterRunStatus, build_sensor_context
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select

from rag_service.config import Settings
from rag_service.dagster.definitions import defs
from rag_service.dagster.launch import DagsterLauncher, LaunchUnavailableError
from rag_service.dagster.resources import RAGResource
from rag_service.dagster.sensors import recover_launches
from rag_service.main import create_app
from rag_service.models.persistence import (
    Chunk,
    DocumentVersion,
    KnowledgeBase,
    ParsedContent,
    ProcessingAttempt,
    ProcessingGeneration,
)
from rag_service.services.documents import upload_original
from rag_service.services.processing import ProcessingFailedError, process_generation
from rag_service.storage import build_storage


def test_dagster_processing_and_reprocessing_survive_restart(  # noqa: PLR0915
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_url=SecretStr(migrated_database), storage_path=tmp_path / "originals"
    )
    monkeypatch.setenv("RAG_STORAGE_PATH", str(settings.storage_path))
    app = create_app(settings)
    with TestClient(app) as client:
        kb = client.post(
            "/knowledge-bases",
            json={"name": "Library", "chunk_size": 8, "chunk_overlap": 2},
        ).json()["id"]
        base = f"/knowledge-bases/{kb}/documents"
        uploaded = client.post(
            base, files={"file": ("data.csv", b"Name,Value\nAlpha,1\nBeta,2")}
        ).json()
        assert uploaded["version"]["status"] == "pending"
        generation_id = UUID(uploaded["generation_id"])
        doc = uploaded["document"]["id"]
        version = uploaded["version"]["id"]
        path = f"{base}/{doc}/versions/{version}"
        config = {
            "ops": {
                "process_document": {"config": {"generation_id": str(generation_id)}}
            }
        }
        instance_path = tmp_path / "dagster"
        instance_path.mkdir()
        with DagsterInstance.local_temp(str(instance_path)) as instance:
            result = defs.resolve_job_def("ingestion_job").execute_in_process(
                run_config=config, instance=instance
            )
            assert result.success
            run_id = result.run_id
        with DagsterInstance.local_temp(str(instance_path)) as restarted:
            run = restarted.get_run_by_id(run_id)
            assert run is not None
            assert run.status == DagsterRunStatus.SUCCESS
            assert any(event.is_dagster_event for event in restarted.all_logs(run_id))
            assert (
                defs.resolve_job_def("ingestion_job")
                .execute_in_process(run_config=config, instance=restarted)
                .success
            )
        with app.state.session_factory() as session:
            generation = session.get_one(ProcessingGeneration, generation_id)
            assert (generation.chunk_size, generation.chunk_overlap) == (8, 2)
            assert generation.status == "ready"
            chunks = list(session.scalars(select(Chunk)))
            assert len(chunks) == 2
            assert chunks[0].text.startswith("Name | Value\n")
            assert chunks[0].provenance["row_start"] == 2
            assert (
                session.get_one(ParsedContent, generation.parsed_content_id).parser_name
                == "csv"
            )
        headers = {"Idempotency-Key": "rechunk-1"}
        with app.state.session_factory() as session:
            knowledge_base = session.get_one(KnowledgeBase, UUID(kb))
            knowledge_base.chunk_size = 4
            knowledge_base.chunk_overlap = 1
            session.commit()
        reprocessed = client.post(path + "/reprocess", json={}, headers=headers)
        assert reprocessed.status_code == 202
        record = reprocessed.json()
        assert record["generation"]["chunk_size"] == 4
        assert record["generation"]["chunk_overlap"] == 1
        assert (
            client.post(path + "/reprocess", json={}, headers=headers).json() == record
        )
        assert (
            client.post(
                path + "/reprocess", json={"reparse": True}, headers=headers
            ).status_code
            == 409
        )
        second = UUID(record["generation"]["id"])
        settings.storage_path.joinpath(
            f"originals/{kb}/{uploaded['version']['checksum']}"
        ).unlink()
        with app.state.session_factory() as session:
            process_generation(session, build_storage(settings), second)
            assert len(list(session.scalars(select(ParsedContent)))) == 1
            assert len(list(session.scalars(select(DocumentVersion)))) == 1
        restarted_app = create_app(settings)
        with TestClient(restarted_app) as restarted_client:
            history = restarted_client.get(path + "/generations").json()
            assert [generation["status"] for generation in history] == [
                "ready",
                "ready",
            ]


def test_failure_retry_checkpoint_and_concurrent_replay(
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_url=SecretStr(migrated_database), storage_path=tmp_path
    )
    app = create_app(settings)
    storage = build_storage(settings)
    with TestClient(app) as client:
        kb = client.post(
            "/knowledge-bases",
            json={"name": "Library", "chunk_size": 8, "chunk_overlap": 2},
        ).json()["id"]
        base = f"/knowledge-bases/{kb}/documents"
        upload = client.post(
            base, files={"file": ("text.txt", b"Some useful content")}
        ).json()
        doc, version = upload["document"]["id"], upload["version"]["id"]
        path = f"{base}/{doc}/versions/{version}"
        gen_id = UUID(upload["generation_id"])
        with app.state.session_factory() as session:
            process_generation(session, storage, gen_id)
        next_generation = client.post(
            path + "/reprocess",
            json={"reparse": True},
            headers={"Idempotency-Key": "bad"},
        ).json()["generation"]
        failed_id = UUID(next_generation["id"])

        def failed_split(_self: object, _text: str) -> list[str]:
            raise ValueError("PRIVATE SOURCE CONTENT")

        with monkeypatch.context() as patch:
            patch.setattr(
                "rag_service.services.processing.SentenceSplitter.split_text",
                failed_split,
            )
            with app.state.session_factory() as session:
                with pytest.raises(ProcessingFailedError) as failure:
                    process_generation(session, storage, failed_id)
                assert "PRIVATE" not in str(failure.value)
                failed = session.get_one(ProcessingGeneration, failed_id)
                assert failed.status == "failed"
                assert failed.parsed_content_id is not None
                assert session.get_one(DocumentVersion, UUID(version)).status == "ready"
                assert len(list(session.scalars(select(Chunk)))) == 1
        assert client.get(base + f"/{doc}").json()["active_version_id"] == version
        retry_path = path + f"/generations/{failed_id}/retry"
        retried = client.post(retry_path, headers={"Idempotency-Key": "retry-1"})
        assert retried.status_code == 202
        assert (
            client.post(retry_path, headers={"Idempotency-Key": "retry-1"}).json()
            == retried.json()
        )
        assert (
            client.post(retry_path, headers={"Idempotency-Key": "retry-2"}).status_code
            == 409
        )
        assert (
            client.post(
                path + "/generations/00000000-0000-0000-0000-000000000000/retry",
                headers={"Idempotency-Key": "missing"},
            ).status_code
            == 409
        )

        def replay(_number: int) -> None:
            with app.state.session_factory() as session:
                process_generation(session, storage, failed_id)

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(replay, range(3)))
        with app.state.session_factory() as session:
            assert session.get_one(ProcessingGeneration, failed_id).status == "ready"
            assert len(list(session.scalars(select(Chunk)))) == 2
            assert len(list(session.scalars(select(ParsedContent)))) == 2
        assert (
            client.post(retry_path, headers={"Idempotency-Key": "retry-1"}).status_code
            == 202
        )


def test_launch_outage_recovered_by_dagster_outbox(
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_url=SecretStr(migrated_database), storage_path=tmp_path
    )
    app = create_app(settings)

    def unavailable(*_args: object) -> str:
        raise LaunchUnavailableError

    monkeypatch.setattr(DagsterLauncher, "launch", unavailable)
    with TestClient(app) as client:
        kb = client.post("/knowledge-bases", json={"name": "Library"}).json()["id"]
        upload = client.post(
            f"/knowledge-bases/{kb}/documents", files={"file": ("text.txt", b"content")}
        ).json()
        assert upload["launch_submitted"] is False
        assert upload["dagster_run_id"] is None
    monkeypatch.setattr(DagsterLauncher, "launch", lambda *_args: "durable-run")
    with (
        DagsterInstance.ephemeral() as instance,
        build_sensor_context(
            instance=instance, resources={"rag": RAGResource()}
        ) as context,
    ):
        recover_launches(context)
        recover_launches(context)
    with app.state.session_factory() as session:
        attempt = session.scalars(select(ProcessingAttempt)).one()
        assert attempt.submitted
        assert attempt.dagster_run_id == "durable-run"
        assert (
            session.get_one(ProcessingGeneration, attempt.generation_id).status
            == "pending"
        )


def test_worker_crash_and_terminal_run_reconciliation(
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_url=SecretStr(migrated_database), storage_path=tmp_path
    )
    app = create_app(settings)
    with DagsterInstance.ephemeral() as instance, TestClient(app) as client:
        run = instance.create_run_for_job(
            defs.resolve_job_def("ingestion_job"),
            run_config={
                "ops": {
                    "process_document": {
                        "config": {
                            "generation_id": "00000000-0000-0000-0000-000000000001"
                        }
                    }
                }
            },
        )
        monkeypatch.setattr(DagsterLauncher, "launch", lambda *_args: run.run_id)
        kb = client.post("/knowledge-bases", json={"name": "Library"}).json()["id"]
        base = f"/knowledge-bases/{kb}/documents"
        uploaded = client.post(
            base, files={"file": ("notes.md", b"# Topic\nContent")}
        ).json()
        gen_id = UUID(uploaded["generation_id"])

        def crash(*_args: object) -> list[str]:
            raise SystemExit("worker killed after parse checkpoint")

        with monkeypatch.context() as patch:
            patch.setattr(
                "rag_service.services.processing.SentenceSplitter.split_text", crash
            )
            with app.state.session_factory() as session, pytest.raises(SystemExit):
                process_generation(session, build_storage(settings), gen_id)
        instance.report_run_failed(run)
        with build_sensor_context(
            instance=instance, resources={"rag": RAGResource()}
        ) as context:
            recover_launches(context)
            recover_launches(context)
        with app.state.session_factory() as session:
            gen = session.get_one(ProcessingGeneration, gen_id)
            assert gen.status == "failed"
            assert gen.error_code == "dagster_run_failed"
            assert gen.parsed_content_id is not None
            assert list(session.scalars(select(Chunk))) == []
        doc, version = uploaded["document"]["id"], uploaded["version"]["id"]
        retry_path = f"{base}/{doc}/versions/{version}/generations/{gen_id}/retry"
        assert (
            client.post(
                retry_path, headers={"Idempotency-Key": "after-restart"}
            ).status_code
            == 202
        )
        with app.state.session_factory() as session:
            process_generation(session, build_storage(settings), gen_id)
            assert session.get_one(ProcessingGeneration, gen_id).status == "ready"
            assert len(list(session.scalars(select(ParsedContent)))) == 1
        failed_upload = client.post(
            base + f"/{doc}/versions", files={"file": ("bad.pdf", b"not a valid PDF")}
        ).json()
        with (
            app.state.session_factory() as session,
            pytest.raises(ProcessingFailedError),
        ):
            process_generation(
                session, build_storage(settings), UUID(failed_upload["generation_id"])
            )
        state = client.get(base + f"/{doc}").json()
        assert state["status"] == "failed"
        assert state["active_version_id"] == version
        assert state["active_processing_generation_id"] == str(gen_id)
        assert client.delete(base + f"/{doc}").status_code == 204
        assert (
            client.post(retry_path, headers={"Idempotency-Key": "deleted"}).status_code
            == 404
        )


def test_orphaned_original_and_automatic_dagster_retry(
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        database_url=SecretStr(migrated_database), storage_path=tmp_path
    )
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path))
    app = create_app(settings)
    storage = build_storage(settings)
    with TestClient(app) as client:
        kb = client.post(
            "/knowledge-bases",
            json={"name": "Library", "chunk_size": 8, "chunk_overlap": 2},
        ).json()["id"]
    with app.state.session_factory() as session:
        # Model an API process stopping after original commit, before reservation.
        _, source, _ = upload_original(
            session,
            storage,
            UUID(kb),
            b"One two three four five six seven eight nine ten.",
            "notes.txt",
            "text/plain",
        )
        source_id = source.id
        assert list(session.scalars(select(ProcessingGeneration))) == []
    with (
        DagsterInstance.ephemeral() as instance,
        build_sensor_context(
            instance=instance, resources={"rag": RAGResource()}
        ) as context,
    ):
        recover_launches(context)
        recover_launches(context)
        with app.state.session_factory() as session:
            generation = session.scalars(select(ProcessingGeneration)).one()
            assert generation.version_id == source_id
            generation_id = generation.id
        original_read = type(storage).read
        reads = 0

        def intermittent_read(self: object, key: str) -> bytes:
            nonlocal reads
            reads += 1
            if reads == 1:
                raise OSError("provider temporary failure")
            return original_read(self, key)  # type: ignore[arg-type]

        monkeypatch.setattr(type(storage), "read", intermittent_read)
        result = defs.resolve_job_def("ingestion_job").execute_in_process(
            run_config={
                "ops": {
                    "process_document": {
                        "config": {"generation_id": str(generation_id)}
                    }
                }
            },
            instance=instance,
        )
        assert result.success
        assert reads == 2
        assert any(
            event.event_type_value == "STEP_UP_FOR_RETRY" for event in result.all_events
        )
        with app.state.session_factory() as session:
            assert (
                session.get_one(ProcessingGeneration, generation_id).status == "ready"
            )
            assert len(list(session.scalars(select(ParsedContent)))) == 1
            chunks = list(session.scalars(select(Chunk).order_by(Chunk.ordinal)))
            assert len(chunks) == 2
            assert all(len(chunk.text.split()) <= 8 for chunk in chunks)
            assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
