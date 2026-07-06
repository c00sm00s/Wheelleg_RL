"""Filesystem path helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BCTRAINCTL_HOME_ENV = "BCTRAINCTL_HOME"


@dataclass(frozen=True, slots=True)
class AppPaths:
    home: Path
    config_file: Path
    jobs_db: Path
    cache_dir: Path
    packages_dir: Path
    logs_dir: Path


def get_app_home() -> Path:
    value = os.environ.get(BCTRAINCTL_HOME_ENV)
    if value:
        return Path(value).expanduser().resolve()
    return Path.home() / ".bctrainctl"


def get_app_paths() -> AppPaths:
    home = get_app_home()
    return AppPaths(
        home=home,
        config_file=home / "config.yaml",
        jobs_db=home / "jobs.db",
        cache_dir=home / "cache",
        packages_dir=home / "packages",
        logs_dir=home / "logs",
    )


def ensure_app_dirs() -> AppPaths:
    paths = get_app_paths()
    for directory in (
        paths.home,
        paths.cache_dir,
        paths.packages_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths
