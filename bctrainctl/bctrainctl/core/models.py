"""Pydantic models for config, manifests, and local job metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from bctrainctl.core.enums import JobStatus, MountType


class BCTrainctlModel(BaseModel):
    """Base model with strict validation."""

    model_config = ConfigDict(extra="forbid")


class DefaultResourceProfile(BCTrainctlModel):
    gpu: int = Field(gt=0)
    cpu: int = Field(gt=0)
    memory_gb: int = Field(gt=0)


class AppConfig(BCTrainctlModel):
    access_key_id: str | None = None
    access_key_secret: SecretStr | None = None
    region: str = Field(min_length=1)
    workspace_id: str | None = None
    oss_bucket: str = Field(min_length=1)
    oss_prefix: str = Field(min_length=1)
    default_resource_id: str | None = None
    default_image: str | None = None
    default_resource_profile: DefaultResourceProfile | None = None
    credential_backend: Literal["file", "aliyun_cli"] = "file"
    aliyun_profile: str | None = None
    # Experiment tracking keys (consumed by training jobs; optional).
    wandb_api_key: SecretStr | None = None
    swanlab_api_key: SecretStr | None = None
    # Alibaba Cloud Container Registry (ACR) credentials for `bctrainctl acr push`.
    acr_registry: str | None = None
    acr_namespace: str | None = None
    acr_username: str | None = None
    acr_password: SecretStr | None = None

    # Config fields holding a SecretStr, shown masked and serialized in plaintext.
    _SECRET_FIELDS: ClassVar[tuple[str, ...]] = (
        "access_key_secret",
        "wandb_api_key",
        "swanlab_api_key",
        "acr_password",
    )

    @model_validator(mode="after")
    def validate_credentials(self) -> "AppConfig":
        if self.credential_backend == "file":
            if not self.access_key_id or self.access_key_secret is None:
                raise ValueError(
                    "access_key_id and access_key_secret are required when credential_backend=file"
                )
        if self.credential_backend == "aliyun_cli" and not self.aliyun_profile:
            raise ValueError("aliyun_profile is required when credential_backend=aliyun_cli")
        return self

    def masked_dict(self) -> dict[str, Any]:
        payload = self.model_dump()
        for field in self._SECRET_FIELDS:
            payload[field] = "********" if getattr(self, field) is not None else None
        return payload

    def plain_dict(self) -> dict[str, Any]:
        payload = self.model_dump()
        for field in self._SECRET_FIELDS:
            secret = getattr(self, field)
            payload[field] = secret.get_secret_value() if secret is not None else None
        return payload


class JobMetadata(BCTrainctlModel):
    name: str = Field(min_length=1)
    project: str = Field(min_length=1)


class ResourceSpec(BCTrainctlModel):
    gpu: int = Field(gt=0)
    cpu: int = Field(gt=0)
    memory_gb: int = Field(gt=0)


class CodeSpec(BCTrainctlModel):
    project_path: Path
    exclude: list[str] = Field(default_factory=list)
    use_gitignore: bool = Field(default=True)
    gitignore_include: list[str] = Field(default_factory=list)


class RuntimeSpec(BCTrainctlModel):
    workdir: str = Field(min_length=1)
    entrypoint: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("entrypoint must not be empty")
        return value.strip()


class StorageMount(BCTrainctlModel):
    name: str = Field(min_length=1)
    type: MountType
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    read_only: bool = False


class StorageSpec(BCTrainctlModel):
    mounts: list[StorageMount] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_unique_targets(self) -> "StorageSpec":
        targets = [mount.target for mount in self.mounts]
        if len(targets) != len(set(targets)):
            raise ValueError("storage.mounts target values must be unique")
        return self


class ExtraUpload(BCTrainctlModel):
    """An extra local file/directory packaged and extracted into the container.

    Handy for small datasets or test fixtures that should travel with the code
    but live outside the project tree.
    """

    local: Path
    target: str = Field(min_length=1)

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("extra_uploads.target must not be empty")
        return stripped


class JobDeclarationSpec(BCTrainctlModel):
    image: str | None = None
    resource_id: str | None = None
    priority: int | None = Field(default=None, ge=1, le=9)
    resources: ResourceSpec | None = None
    code: CodeSpec
    runtime: RuntimeSpec
    storage: StorageSpec = Field(default_factory=StorageSpec)
    extra_uploads: list[ExtraUpload] = Field(default_factory=list)


class JobManifest(BCTrainctlModel):
    api_version: str = Field(alias="api_version")
    kind: Literal["DLCJob"]
    metadata: JobMetadata
    spec: JobDeclarationSpec


class JobSubmission(BCTrainctlModel):
    submission_id: str
    package_path: Path
    package_hash: str
    oss_code_uri: str
    submitted_at: str
    dlc_job_id: str | None = None
    request_snapshot: dict[str, Any]


class JobRecord(BCTrainctlModel):
    local_id: int | None = None
    submission_id: str
    job_id: str
    job_name: str
    project: str
    status: JobStatus
    created_at: str
    updated_at: str
    last_synced_at: str | None = None
    oss_code_uri: str
    package_path: str
    package_hash: str
    entrypoint: str
    image: str
    raw_spec_json: str
    request_snapshot_json: str
    remote_payload_json: str | None = None


class PackageResult(BCTrainctlModel):
    package_path: Path
    sha256: str
    file_count: int
    size_bytes: int
