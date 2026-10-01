"""Portable original-file storage with one save/read/exists/delete contract."""

import os
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Protocol

import boto3
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

from rag_service.config import Settings


class FileStorage(Protocol):
    """Application-owned binary storage boundary."""

    def save(self, key: str, content: bytes) -> None: ...
    def read(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...


def validate_key(key: str) -> str:
    """Reject ambiguous or escaping paths consistently across backends."""
    path = PurePosixPath(key)
    if (
        not key
        or path.is_absolute()
        or "\\" in key
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise ValueError("Storage key must be a relative path without traversal")
    return key


class FilesystemStorage:
    """Atomic file replacement under a filesystem or NAS root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _path(self, key: str) -> Path:
        target = (self.root / validate_key(key)).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Storage path escapes the configured root")
        return target

    def save(self, key: str, content: bytes) -> None:
        """Readers see either the previous bytes or the complete replacement."""
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary.close()
                temporary_path.replace(target)
            finally:
                temporary_path.unlink(missing_ok=True)

    def read(self, key: str) -> bytes:
        """Return original bytes."""
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        """Report only stored files."""
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        """Deleting an absent file is retry-safe."""
        self._path(key).unlink(missing_ok=True)


def is_missing(error: ClientError) -> bool:
    """Absence is distinct from authorization and connectivity failures."""
    return str(error.response.get("Error", {}).get("Code")) in {
        "404",
        "NoSuchKey",
        "NotFound",
    }


class S3Storage:
    """S3-compatible adapter using the SDK's runtime credential provider chain."""

    def __init__(self, client: S3Client, bucket: str) -> None:
        self.client = client
        self.bucket = bucket

    def save(self, key: str, content: bytes) -> None:
        """Save exact source bytes without text conversion."""
        self.client.put_object(Bucket=self.bucket, Key=validate_key(key), Body=content)

    def read(self, key: str) -> bytes:
        """Match the filesystem adapter's missing-file behavior."""
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=validate_key(key))[
                "Body"
            ]
        except ClientError as exc:
            if is_missing(exc):
                raise FileNotFoundError(key) from exc
            raise
        try:
            return body.read()
        finally:
            body.close()

    def exists(self, key: str) -> bool:
        """Do not mistake denied access for absent content."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=validate_key(key))
        except ClientError as exc:
            if is_missing(exc):
                return False
            raise
        return True

    def delete(self, key: str) -> None:
        """S3 delete is idempotent."""
        self.client.delete_object(Bucket=self.bucket, Key=validate_key(key))


def build_storage(settings: Settings) -> FileStorage:
    """Resolve application adapters without infrastructure-specific host policy."""
    if settings.storage_backend == "filesystem":
        return FilesystemStorage(settings.storage_path)
    return S3Storage(
        boto3.client(
            "s3", endpoint_url=settings.s3_endpoint, region_name=settings.s3_region
        ),
        settings.s3_bucket,
    )
