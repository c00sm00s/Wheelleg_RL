"""Textual screen for building and submitting a job interactively."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Switch, TextArea

from bctrainctl.tui.templates import (
    list_templates,
    load_manifest_file,
    load_template,
    looks_like_path,
    resolve_path,
    save_template,
    save_template_to_path,
)

# Single-line text fields: (key, label, default, placeholder)
TEXT_FIELDS: list[tuple[str, str, str, str]] = [
    ("name", "Job name", "", "my-training-job"),
    ("project", "Project", "", "team-project"),
    ("image", "Image (blank = config default)", "", "registry/repo:tag"),
    ("resource_id", "DLC quota (blank = config default)", "", "quota-4090"),
    ("priority", "Priority 1-9 (blank = default)", "", "1"),
    ("gpu", "GPU count", "", "1"),
    ("cpu", "CPU count", "", "16"),
    ("memory_gb", "Memory (GB)", "", "60"),
    ("project_path", "Project path", ".", "."),
    ("workdir", "Container workdir", "/workspace/code", "/workspace/code"),
    ("entrypoint", "Entrypoint", "train.sh", "train.sh"),
]

# Multi-line text areas: (key, label, placeholder)
AREA_FIELDS: list[tuple[str, str]] = [
    ("exclude", "Exclude patterns (one per line)"),
    ("gitignore_include", "Keep-even-if-ignored patterns (one per line)"),
    ("env", "Environment vars (KEY=VALUE per line)"),
    ("extra_uploads", "Extra uploads (local:target per line)"),
    ("storage", "Storage mounts (type source target [ro] per line)"),
]


class SubmitScreen(Screen):
    """Compose a DLCJob manifest, optionally save it as a template, then submit."""

    BINDINGS = [
        ("ctrl+s", "submit", "Submit"),
        ("escape", "cancel", "Back"),
    ]

    def __init__(self, *, standalone: bool = True) -> None:
        super().__init__()
        self.standalone = standalone

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="submit-form"):
            yield Label("[b]New training job[/b]")
            templates = list_templates()
            yield Label("Load saved template")
            yield Select(
                [(name, name) for name in templates],
                prompt="(no template)",
                id="template-select",
                allow_blank=True,
            )
            yield Label("Or load a YAML file (path relative to current directory)")
            yield Input(value="job.yaml", placeholder="./job.yaml", id="field-load_path")
            for key, label, default, placeholder in TEXT_FIELDS:
                yield Label(label)
                yield Input(value=default, placeholder=placeholder, id=f"field-{key}")
            yield Label("Use .gitignore filtering")
            yield Switch(value=True, id="field-use_gitignore")
            for key, label in AREA_FIELDS:
                yield Label(label)
                yield TextArea(id=f"area-{key}")
            yield Label("Save as (template name, or a path like ./my-job.yaml)")
            yield Input(placeholder="template-name  或  ./my-job.yaml", id="field-template_name")
        with Horizontal(id="submit-buttons"):
            yield Button("Submit", variant="success", id="submit")
            yield Button("Load file", variant="primary", id="load-file")
            yield Button("Save", variant="primary", id="save-template")
            yield Button("Back", variant="default", id="cancel")
        yield Footer()

    # --- events -----------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self.action_submit()
        elif event.button.id == "load-file":
            self._load_file()
        elif event.button.id == "save-template":
            self._save_template()
        elif event.button.id == "cancel":
            self.action_cancel()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "template-select" and event.value not in (None, Select.BLANK):
            self._apply_template(str(event.value))

    # --- actions ----------------------------------------------------------
    def action_submit(self) -> None:
        try:
            manifest = self._build_manifest()
        except ValueError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        with NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="bctrainctl-tui-", delete=False, encoding="utf-8"
        ) as handle:
            yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)
            path = Path(handle.name)
        self.app.exit(path)

    def action_cancel(self) -> None:
        if self.standalone:
            self.app.exit(None)
        else:
            self.app.pop_screen()

    # --- helpers ----------------------------------------------------------
    def _save_template(self) -> None:
        target = self.query_one("#field-template_name", Input).value.strip()
        if not target:
            self.notify("Enter a template name or a save path first.", severity="warning")
            return
        try:
            manifest = self._build_manifest(require_name=False)
        except ValueError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        try:
            if looks_like_path(target):
                path = save_template_to_path(resolve_path(target), manifest)
            else:
                path = save_template(target, manifest)
        except OSError as exc:
            self.notify(f"Save failed: {exc}", severity="error", timeout=8)
            return
        self.notify(f"Saved: {path}", severity="information", timeout=6)

    def _load_file(self) -> None:
        raw = self.query_one("#field-load_path", Input).value.strip()
        if not raw:
            self.notify("Enter a YAML file path to load.", severity="warning")
            return
        path = resolve_path(raw)
        try:
            data = load_manifest_file(path)
        except FileNotFoundError:
            self.notify(f"File not found: {path}", severity="error", timeout=8)
            return
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Failed to read YAML: {exc}", severity="error", timeout=8)
            return
        self._populate_from_manifest(data)
        self.notify(f"Loaded: {path}", severity="information", timeout=6)

    def _apply_template(self, name: str) -> None:
        try:
            data = load_template(name)
        except FileNotFoundError:
            self.notify(f"Template not found: {name}", severity="error")
            return
        self._populate_from_manifest(data)
        self.notify(f"Loaded template: {name}", severity="information")

    def _text(self, key: str) -> str:
        return self.query_one(f"#field-{key}", Input).value.strip()

    def _area(self, key: str) -> str:
        return self.query_one(f"#area-{key}", TextArea).text

    def _build_manifest(self, *, require_name: bool = True) -> dict[str, Any]:
        name = self._text("name")
        project = self._text("project")
        if require_name and not name:
            raise ValueError("Job name is required.")
        if require_name and not project:
            raise ValueError("Project is required.")

        spec: dict[str, Any] = {}
        if self._text("image"):
            spec["image"] = self._text("image")
        if self._text("resource_id"):
            spec["resource_id"] = self._text("resource_id")
        if self._text("priority"):
            spec["priority"] = _to_int("priority", self._text("priority"))

        resources = {
            field: _to_int(field, self._text(field))
            for field in ("gpu", "cpu", "memory_gb")
            if self._text(field)
        }
        if resources:
            if len(resources) != 3:
                raise ValueError("Provide GPU, CPU, and Memory together, or leave all blank.")
            spec["resources"] = resources

        code: dict[str, Any] = {
            "project_path": self._text("project_path") or ".",
            "use_gitignore": self.query_one("#field-use_gitignore", Switch).value,
        }
        exclude = _parse_lines(self._area("exclude"))
        if exclude:
            code["exclude"] = exclude
        include = _parse_lines(self._area("gitignore_include"))
        if include:
            code["gitignore_include"] = include
        spec["code"] = code

        runtime: dict[str, Any] = {
            "workdir": self._text("workdir") or "/workspace/code",
            "entrypoint": self._text("entrypoint") or "train.sh",
        }
        env = _parse_env(self._area("env"))
        if env:
            runtime["env"] = env
        spec["runtime"] = runtime

        mounts = _parse_mounts(self._area("storage"))
        if mounts:
            spec["storage"] = {"mounts": mounts}

        extra = _parse_extra_uploads(self._area("extra_uploads"))
        if extra:
            spec["extra_uploads"] = extra

        return {
            "api_version": "v1",
            "kind": "DLCJob",
            "metadata": {"name": name, "project": project},
            "spec": spec,
        }

    def _populate_from_manifest(self, data: dict[str, Any]) -> None:
        metadata = data.get("metadata") or {}
        spec = data.get("spec") or {}
        resources = spec.get("resources") or {}
        code = spec.get("code") or {}
        runtime = spec.get("runtime") or {}

        values = {
            "name": metadata.get("name", ""),
            "project": metadata.get("project", ""),
            "image": spec.get("image", ""),
            "resource_id": spec.get("resource_id", ""),
            "priority": spec.get("priority", ""),
            "gpu": resources.get("gpu", ""),
            "cpu": resources.get("cpu", ""),
            "memory_gb": resources.get("memory_gb", ""),
            "project_path": code.get("project_path", "."),
            "workdir": runtime.get("workdir", "/workspace/code"),
            "entrypoint": runtime.get("entrypoint", "train.sh"),
        }
        for key, value in values.items():
            self.query_one(f"#field-{key}", Input).value = "" if value is None else str(value)

        self.query_one("#field-use_gitignore", Switch).value = bool(code.get("use_gitignore", True))
        self.query_one("#area-exclude", TextArea).text = "\n".join(code.get("exclude", []) or [])
        self.query_one("#area-gitignore_include", TextArea).text = "\n".join(
            code.get("gitignore_include", []) or []
        )
        self.query_one("#area-env", TextArea).text = "\n".join(
            f"{key}={value}" for key, value in (runtime.get("env") or {}).items()
        )
        self.query_one("#area-extra_uploads", TextArea).text = "\n".join(
            f"{item.get('local', '')}:{item.get('target', '')}"
            for item in (spec.get("extra_uploads") or [])
        )
        self.query_one("#area-storage", TextArea).text = "\n".join(
            _format_mount(mount) for mount in (spec.get("storage", {}) or {}).get("mounts", [])
        )


def _to_int(field: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer, got: {value!r}") from exc


def _parse_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_env(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in _parse_lines(text):
        if "=" not in line:
            raise ValueError(f"Env line must be KEY=VALUE, got: {line!r}")
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _parse_extra_uploads(text: str) -> list[dict[str, str]]:
    uploads: list[dict[str, str]] = []
    for line in _parse_lines(text):
        if ":" not in line:
            raise ValueError(f"Extra upload must be local:target, got: {line!r}")
        local, target = line.split(":", 1)
        uploads.append({"local": local.strip(), "target": target.strip()})
    return uploads


def _parse_mounts(text: str) -> list[dict[str, Any]]:
    mounts: list[dict[str, Any]] = []
    for index, line in enumerate(_parse_lines(text)):
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Mount must be 'type source target [ro]', got: {line!r}")
        mount_type, source, target = parts[0], parts[1], parts[2]
        read_only = len(parts) >= 4 and parts[3].lower() == "ro"
        mounts.append(
            {
                "name": f"{mount_type}-{index}",
                "type": mount_type,
                "source": source,
                "target": target,
                "read_only": read_only,
            }
        )
    return mounts


def _format_mount(mount: dict[str, Any]) -> str:
    base = f"{mount.get('type', '')} {mount.get('source', '')} {mount.get('target', '')}"
    return f"{base} ro" if mount.get("read_only") else base
