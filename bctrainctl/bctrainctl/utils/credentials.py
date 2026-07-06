"""Credential resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass

from bctrainctl.core.models import AppConfig
from bctrainctl.storage.aliyun_cli_store import AliyunCliConfigStore
from bctrainctl.utils.errors import ValidationFailedError


@dataclass(frozen=True, slots=True)
class ResolvedCredentials:
    access_key_id: str
    access_key_secret: str
    source: str
    profile: str | None = None


def resolve_credentials(config: AppConfig) -> ResolvedCredentials:
    """Resolve usable AccessKey credentials from config or Aliyun CLI config."""

    if config.credential_backend == "aliyun_cli":
        profile_name = config.aliyun_profile
        if not profile_name:
            raise ValidationFailedError(
                "aliyun_profile is missing from config.",
                hint="Run `bctrainctl init` again and reselect the Aliyun CLI profile.",
            )
        profile = AliyunCliConfigStore().get_profile(profile_name)
        return ResolvedCredentials(
            access_key_id=profile.access_key_id,
            access_key_secret=profile.access_key_secret,
            source="aliyun_cli",
            profile=profile.name,
        )

    if config.access_key_id and config.access_key_secret is not None:
        return ResolvedCredentials(
            access_key_id=config.access_key_id,
            access_key_secret=config.access_key_secret.get_secret_value(),
            source="file",
        )

    cli_store = AliyunCliConfigStore()
    if cli_store.exists():
        profile = cli_store.load_current_profile()
        return ResolvedCredentials(
            access_key_id=profile.access_key_id,
            access_key_secret=profile.access_key_secret,
            source="aliyun_cli",
            profile=profile.name,
        )

    raise ValidationFailedError(
        "No usable Alibaba Cloud credentials were found.",
        hint="Run `bctrainctl init`, or configure ~/.aliyun/config.json with an AK profile.",
    )
