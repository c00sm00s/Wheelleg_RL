"""Local YAML config storage."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from bctrainctl.core.models import AppConfig
from bctrainctl.utils.errors import ConfigNotFoundError, ValidationFailedError
from bctrainctl.utils.paths import ensure_app_dirs


class ConfigStore:
    """Read and write the local bctrainctl config file."""

    def __init__(self, path: Path | None = None) -> None:
        paths = ensure_app_dirs()
        self.path = path or paths.config_file

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> AppConfig | None:
        if not self.path.exists():
            return None
        try:
            payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ValidationFailedError(
                f"Failed to parse config file {self.path}: {exc}",
                hint="Fix the YAML or rerun `bctrainctl init` to recreate the config.",
            ) from exc
        try:
            return AppConfig.model_validate(payload)
        except ValidationError as exc:
            raise ValidationFailedError(
                f"Config file validation failed:\n{exc}",
                hint="Update the config file or rerun `bctrainctl init`.",
            ) from exc

    def load_required(self) -> AppConfig:
        config = self.load()
        if config is None:
            raise ConfigNotFoundError(
                f"Config file not found: {self.path}",
                hint="Run `bctrainctl init` before using this command.",
            )
        return config

    def save(self, config: AppConfig) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = config.plain_dict()
        self.path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return self.path

