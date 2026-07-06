"""Manifest loading and validation helpers."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from bctrainctl.core.enums import MountType
from bctrainctl.core.models import (
    AppConfig,
    ExtraUpload,
    JobManifest,
    JobMetadata,
    ResourceSpec,
    StorageMount,
)
from bctrainctl.utils.errors import ValidationFailedError

OSS_SIMPLE_URI_RE = re.compile(r"^oss://(?P<bucket>[^/.]+)(?P<path>/.*)?$")


def _normalize_oss_uri(uri: str, region: str) -> str:
    value = uri.strip()
    match = OSS_SIMPLE_URI_RE.match(value)
    if not match:
        return value
    bucket = match.group("bucket")
    path = match.group("path") or "/"
    if not path.endswith("/"):
        path = f"{path}/"
    return f"oss://{bucket}.oss-{region}-internal.aliyuncs.com{path}"


def _normalize_storage_mounts(manifest: JobManifest, region: str) -> list[StorageMount]:
    mounts: list[StorageMount] = []
    for mount in manifest.spec.storage.mounts:
        if mount.type == MountType.OSS:
            normalized_source = _normalize_oss_uri(mount.source, region)
            mounts.append(mount.model_copy(update={"source": normalized_source}))
            continue
        mounts.append(mount)
    return mounts


def load_job_manifest(
    manifest_path: Path,
    *,
    config: AppConfig,
    name_override: str | None = None,
) -> JobManifest:
    """Load and validate a job manifest from YAML."""

    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationFailedError(
            f"Job file not found: {manifest_path}",
            hint="Check the `-f/--file` path and try again.",
        ) from exc
    except yaml.YAMLError as exc:
        raise ValidationFailedError(
            f"Invalid YAML in {manifest_path}: {exc}",
            hint="Fix the YAML syntax and retry `bctrainctl submit`.",
        ) from exc

    if not isinstance(raw, dict):
        raise ValidationFailedError(
            "Job file must contain a YAML object at the root.",
            hint="Start the file with keys like `api_version`, `kind`, `metadata`, and `spec`.",
        )

    try:
        manifest = JobManifest.model_validate(raw)
    except ValidationError as exc:
        raise ValidationFailedError(
            f"Job manifest validation failed:\n{exc}",
            hint="Check required fields such as metadata.name, spec.runtime.entrypoint, and resources.",
        ) from exc

    project_path = manifest.spec.code.project_path
    if not project_path.is_absolute():
        project_path = (manifest_path.parent / project_path).resolve()

    image = manifest.spec.image or config.default_image
    resource_id = manifest.spec.resource_id or config.default_resource_id
    resources = manifest.spec.resources
    if resources is None and config.default_resource_profile is not None:
        resources = ResourceSpec.model_validate(config.default_resource_profile.model_dump())
    if not image:
        raise ValidationFailedError(
            "Job image is missing.",
            hint="Set `spec.image` in job.yaml or configure `default_image` in `bctrainctl init`.",
        )
    if not resource_id:
        raise ValidationFailedError(
            "A workspace-bound DLC quota is missing.",
            hint="Set `spec.resource_id` in job.yaml or configure `default_resource_id` in `bctrainctl init`. Use a bound quota such as `quota-4090`.",
        )
    if not resources:
        raise ValidationFailedError(
            "Job resources are missing.",
            hint="Set `spec.resources` in job.yaml or configure `default_resource_profile` in `bctrainctl init`.",
        )

    if not project_path.exists():
        raise ValidationFailedError(
            f"Project path does not exist: {project_path}",
            hint="Check `spec.code.project_path` and run the command from the expected directory.",
        )
    if not project_path.is_dir():
        raise ValidationFailedError(
            f"Project path is not a directory: {project_path}",
            hint="Point `spec.code.project_path` to a directory that contains your training code.",
        )

    extra_uploads = _resolve_extra_uploads(manifest, manifest_path.parent)

    metadata = manifest.metadata
    if name_override:
        metadata = JobMetadata(name=name_override, project=metadata.project)

    return manifest.model_copy(
        update={
            "metadata": metadata,
            "spec": manifest.spec.model_copy(
                update={
                    "image": image,
                    "resource_id": resource_id,
                    "resources": resources,
                    "code": manifest.spec.code.model_copy(update={"project_path": project_path}),
                    "storage": manifest.spec.storage.model_copy(
                        update={"mounts": _normalize_storage_mounts(manifest, config.region)}
                    ),
                    "extra_uploads": extra_uploads,
                }
            ),
        }
    )


def _resolve_extra_uploads(manifest: JobManifest, base_dir: Path) -> list[ExtraUpload]:
    resolved: list[ExtraUpload] = []
    for upload in manifest.spec.extra_uploads:
        local_path = upload.local
        if not local_path.is_absolute():
            local_path = (base_dir / local_path).resolve()
        if not local_path.exists():
            raise ValidationFailedError(
                f"Extra upload path does not exist: {local_path}",
                hint="Check `spec.extra_uploads[].local` and run from the expected directory.",
            )
        resolved.append(upload.model_copy(update={"local": local_path}))
    return resolved
