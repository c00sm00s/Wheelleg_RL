"""OSS wrapper for package uploads and mount preparation."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import oss2
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from bctrainctl.core.models import AppConfig
from bctrainctl.utils.console import console
from bctrainctl.utils.credentials import resolve_credentials
from bctrainctl.utils.errors import ClientOperationError


class OssClient:
    """Thin wrapper around `oss2.Bucket` with bctrainctl-friendly errors."""

    def __init__(self, config: AppConfig, *, debug: bool = False) -> None:
        self.config = config
        self.debug = debug
        credentials = resolve_credentials(config)
        self.auth = oss2.Auth(
            credentials.access_key_id,
            credentials.access_key_secret,
        )
        self.endpoint = f"https://oss-{config.region}.aliyuncs.com"
        self.bucket = oss2.Bucket(self.auth, self.endpoint, config.oss_bucket)

    def check_bucket_access(self) -> None:
        try:
            self.bucket.get_bucket_info()
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                f"OSS bucket is not accessible: {self.config.oss_bucket}",
                hint="Check AccessKey permissions, region, bucket name, and OSS network reachability.",
            ) from exc

    def build_oss_uri(self, object_key: str) -> str:
        normalized = object_key.lstrip("/")
        return f"oss://{self.config.oss_bucket}/{normalized}"

    def upload_file(self, local_path: Path, object_key: str) -> str:
        normalized = object_key.lstrip("/")
        total_size = local_path.stat().st_size
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
            console=console,
        ) as progress:
            task_id = progress.add_task(f"Uploading {local_path.name}", total=total_size or 1)

            def callback(consumed: int, _: int) -> None:
                progress.update(task_id, completed=consumed)

            try:
                self.bucket.put_object_from_file(
                    normalized,
                    str(local_path),
                    progress_callback=callback,
                )
            except Exception as exc:  # noqa: BLE001
                raise ClientOperationError(
                    f"Failed to upload package to OSS: {self.build_oss_uri(normalized)}",
                    hint="Check bucket access, object key, network reachability, and local file readability.",
                ) from exc
        return self.build_oss_uri(normalized)

    def upload_bytes(self, data: bytes, object_key: str) -> str:
        normalized = object_key.lstrip("/")
        try:
            self.bucket.put_object(normalized, data)
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                f"Failed to upload bytes to OSS: {self.build_oss_uri(normalized)}",
                hint="Check bucket access and object key permissions.",
            ) from exc
        return self.build_oss_uri(normalized)

    def delete_object(self, object_key: str) -> None:
        normalized = object_key.lstrip("/")
        if not normalized:
            raise ClientOperationError(
                "Failed to delete OSS object: empty object key",
                hint="Expected a concrete OSS object path, not a bucket root.",
            )
        try:
            self.bucket.delete_object(normalized)
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                f"Failed to delete OSS object: {self.build_oss_uri(normalized)}",
                hint="Check bucket permissions and confirm the object key is valid.",
            ) from exc

    def delete_oss_uri(self, oss_uri: str) -> None:
        bucket_name, object_key = self.parse_oss_uri(oss_uri)
        if not object_key:
            raise ClientOperationError(
                f"Failed to delete OSS object: {oss_uri}",
                hint="Expected a concrete OSS object path, not a bucket root.",
            )
        bucket = self._bucket_for_name(bucket_name)
        try:
            bucket.delete_object(object_key)
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                f"Failed to delete OSS object: {oss_uri}",
                hint="Check bucket permissions and confirm the object exists in the target bucket.",
            ) from exc

    def ensure_prefix_placeholder(
        self,
        oss_uri: str,
        *,
        object_name: str = ".bctrainctl_keep",
    ) -> str:
        bucket_name, object_prefix = self.parse_prefix_uri(oss_uri)
        object_key = f"{object_prefix}{object_name}" if object_prefix else object_name
        bucket = self._bucket_for_name(bucket_name)
        try:
            bucket.put_object(object_key, b"")
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                f"Failed to prepare OSS mount prefix: {oss_uri}",
                hint=(
                    "Check bucket permissions, prefix write access, and whether the OSS URI "
                    "points to a bucket in the configured region."
                ),
            ) from exc
        return f"oss://{bucket_name}/{object_key}"

    def head_object(self, object_key: str) -> dict[str, str]:
        normalized = object_key.lstrip("/")
        try:
            result = self.bucket.head_object(normalized)
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                f"Failed to inspect OSS object: {self.build_oss_uri(normalized)}",
                hint="Confirm the object exists and your RAM account can read it.",
            ) from exc
        return {
            "etag": getattr(result, "etag", ""),
            "content_length": str(getattr(result, "content_length", "")),
            "last_modified": str(getattr(result, "last_modified", "")),
        }

    def sign_download_url(self, object_key: str, *, expires_seconds: int = 7 * 24 * 3600) -> str:
        normalized = object_key.lstrip("/")
        try:
            return self.bucket.sign_url("GET", normalized, expires_seconds)
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                f"Failed to sign OSS download URL for {self.build_oss_uri(normalized)}",
                hint="Check bucket permissions and confirm the uploaded object key is valid.",
            ) from exc

    @staticmethod
    def object_key_from_oss_uri(oss_uri: str) -> str:
        _, object_key = OssClient.parse_oss_uri(oss_uri)
        return object_key

    @staticmethod
    def basename_from_oss_uri(oss_uri: str) -> str:
        object_key = OssClient.object_key_from_oss_uri(oss_uri)
        return os.path.basename(object_key)

    @staticmethod
    def parent_uri_from_oss_uri(oss_uri: str) -> str:
        bucket, object_key = OssClient.parse_oss_uri(oss_uri)
        if not bucket:
            raise ClientOperationError(
                f"Invalid OSS URI: {oss_uri}",
                hint="Expected a bucket name in the OSS URI.",
            )
        if not object_key or "/" not in object_key:
            return f"oss://{bucket}"
        parent = object_key.rsplit("/", 1)[0]
        return f"oss://{bucket}/{parent}"

    @staticmethod
    def parse_prefix_uri(oss_uri: str) -> tuple[str, str]:
        bucket, object_key = OssClient.parse_oss_uri(oss_uri)
        if object_key and not object_key.endswith("/"):
            object_key = f"{object_key}/"
        return bucket, object_key

    @staticmethod
    def parse_oss_uri(oss_uri: str) -> tuple[str, str]:
        if not oss_uri.startswith("oss://"):
            raise ClientOperationError(
                f"Invalid OSS URI: {oss_uri}",
                hint="Expected a URI in the form `oss://bucket/path/to/object`.",
            )
        parsed = urlparse(oss_uri)
        host = parsed.netloc
        if not host:
            raise ClientOperationError(
                f"Invalid OSS URI: {oss_uri}",
                hint="Expected a bucket name in the OSS URI.",
            )
        bucket = host.split(".", 1)[0]
        object_key = parsed.path.lstrip("/")
        return bucket, object_key

    def _bucket_for_name(self, bucket_name: str) -> oss2.Bucket:
        if bucket_name == self.config.oss_bucket:
            return self.bucket
        return oss2.Bucket(self.auth, self.endpoint, bucket_name)
