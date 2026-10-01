"""Run the same original-byte storage contract for filesystem and S3."""

from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from moto import mock_aws

from rag_service.config import Settings
from rag_service.storage import FileStorage, FilesystemStorage, S3Storage, build_storage


@pytest.fixture(params=["filesystem", "s3"])
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[FileStorage]:
    if request.param == "filesystem":
        yield build_storage(Settings(storage_path=tmp_path))
    else:
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="source-documents")
            yield build_storage(
                Settings(storage_backend="s3", s3_bucket="source-documents")
            )


def test_exact_bytes_and_retry_safe_operations(storage: FileStorage) -> None:
    key = "knowledge-base/version/original.bin"
    assert not storage.exists(key)
    with pytest.raises(FileNotFoundError):
        storage.read(key)
    storage.delete(key)
    content = bytes(range(256)) + b"\x00\xff\xfe\r\n"
    storage.save(key, content)
    storage.save(key, content)
    assert storage.exists(key)
    assert storage.read(key) == content
    storage.save(key, b"")
    assert storage.read(key) == b""
    storage.delete(key)
    storage.delete(key)
    assert not storage.exists(key)


@pytest.mark.parametrize(
    "key", ["", "../escape", "/absolute", "x/../escape", "x\\escape", "x//y", "./x"]
)
def test_invalid_keys_are_rejected(storage: FileStorage, key: str) -> None:
    with pytest.raises(ValueError, match="Storage key"):
        storage.save(key, b"content")


def test_symlink_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(tmp_path, target_is_directory=True)
    storage = FilesystemStorage(root)
    with pytest.raises(ValueError, match="escapes"):
        storage.save("link/escape", b"content")


def test_s3_access_errors_are_not_reported_as_missing() -> None:
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",  # noqa: S106 - isolated SDK test credentials
    )
    storage = S3Storage(client, "source-documents")
    with Stubber(client) as stub:
        stub.add_client_error(
            "head_object", service_error_code="AccessDenied", http_status_code=403
        )
        with pytest.raises(ClientError):
            storage.exists("original")
        stub.add_client_error(
            "get_object", service_error_code="AccessDenied", http_status_code=403
        )
        with pytest.raises(ClientError):
            storage.read("original")
