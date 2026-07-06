"""Job template storage for the interactive submit TUI.

Templates are plain YAML job manifests stored under the app home so users can
build a job once in the TUI, save it, and reuse it later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bctrainctl.utils.paths import get_app_home


def templates_dir() -> Path:
    directory = get_app_home() / "templates"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_name(name: str) -> str:
    cleaned = name.strip().replace("/", "-").replace(" ", "-")
    if not cleaned:
        raise ValueError("Template name must not be empty.")
    return cleaned


def template_path(name: str) -> Path:
    return templates_dir() / f"{_safe_name(name)}.yaml"


def list_templates() -> list[str]:
    return sorted(path.stem for path in templates_dir().glob("*.yaml"))


def load_template(name: str) -> dict[str, Any]:
    path = template_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"Template not found: {name}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_template(name: str, data: dict[str, Any]) -> Path:
    path = template_path(name)
    return _dump_yaml(path, data)


def looks_like_path(value: str) -> bool:
    """Heuristic: treat a save/load target as a filesystem path, not a bare name."""

    value = value.strip()
    return (
        "/" in value
        or "\\" in value
        or value.endswith((".yaml", ".yml"))
        or Path(value).is_absolute()
    )


def resolve_path(value: str, *, base_dir: Path | None = None) -> Path:
    """Resolve a user-entered path relative to the given base (default: cwd)."""

    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    return path


def load_manifest_file(path: str | Path) -> dict[str, Any]:
    """Load a job manifest from an arbitrary YAML file (e.g. one in the project)."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    return yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}


def save_template_to_path(path: str | Path, data: dict[str, Any]) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return _dump_yaml(file_path, data)


def _dump_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
