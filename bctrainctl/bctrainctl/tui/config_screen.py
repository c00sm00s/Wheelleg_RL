"""Textual screen for viewing and editing bctrainctl configuration."""

from __future__ import annotations

from pydantic import ValidationError
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label

from bctrainctl.cli.init import INIT_DEFAULTS
from bctrainctl.core.models import AppConfig
from bctrainctl.storage.config_store import ConfigStore

# (field key, display label, is_secret)
CONFIG_FIELDS: list[tuple[str, str, bool]] = [
    ("access_key_id", "Alibaba Cloud AccessKey ID", False),
    ("access_key_secret", "Alibaba Cloud AccessKey Secret", True),
    ("region", "Region", False),
    ("workspace_id", "Workspace ID", False),
    ("oss_bucket", "OSS Bucket", False),
    ("oss_prefix", "OSS Prefix", False),
    ("default_resource_id", "Default DLC Quota", False),
    ("default_image", "Default Image", False),
    ("wandb_api_key", "Weights & Biases API Key", True),
    ("swanlab_api_key", "SwanLab API Key", True),
    ("acr_registry", "ACR Registry", False),
    ("acr_namespace", "ACR Namespace", False),
    ("acr_username", "ACR Username", False),
    ("acr_password", "ACR Password", True),
]

# Defaults pre-filled in the form when there is no existing config yet, mirroring
# the `bctrainctl init` defaults so TUI-based first-time setup is just as quick.
NEW_CONFIG_DEFAULTS = {
    "region": INIT_DEFAULTS["region"],
    "workspace_id": INIT_DEFAULTS["workspace_id"],
    "oss_bucket": INIT_DEFAULTS["oss_bucket"],
    "oss_prefix": INIT_DEFAULTS["oss_prefix"],
    "default_resource_id": INIT_DEFAULTS["default_resource_id"],
    "default_image": INIT_DEFAULTS["default_image"],
    "acr_registry": INIT_DEFAULTS["acr_registry"],
    "acr_namespace": INIT_DEFAULTS["acr_namespace"],
    "credential_backend": "file",
}


class ConfigScreen(Screen):
    """Edit the aliyun / wandb / swanlab / acr credentials and defaults."""

    BINDINGS = [
        ("ctrl+s", "save", "Save"),
        ("escape", "cancel", "Back"),
    ]

    def __init__(self, *, standalone: bool = True) -> None:
        super().__init__()
        self.standalone = standalone

    def compose(self) -> ComposeResult:
        yield Header()
        existing = ConfigStore().load()
        prefill = existing.plain_dict() if existing is not None else dict(NEW_CONFIG_DEFAULTS)
        with VerticalScroll(id="config-form"):
            yield Label("[b]Configuration[/b]  (secrets are stored locally, masked on display)")
            for key, label, is_secret in CONFIG_FIELDS:
                yield Label(label)
                value = prefill.get(key)
                yield Input(
                    value="" if value is None else str(value),
                    password=is_secret,
                    id=f"field-{key}",
                )
        with Horizontal(id="config-buttons"):
            yield Button("Save", variant="success", id="save")
            yield Button("Back", variant="default", id="cancel")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.action_save()
        elif event.button.id == "cancel":
            self.action_cancel()

    def action_save(self) -> None:
        existing = ConfigStore().load()
        payload = existing.plain_dict() if existing is not None else dict(NEW_CONFIG_DEFAULTS)
        for key, _label, _is_secret in CONFIG_FIELDS:
            raw = self.query_one(f"#field-{key}", Input).value.strip()
            payload[key] = raw or None
        try:
            config = AppConfig.model_validate(payload)
        except ValidationError as exc:
            self.notify(_format_validation_error(exc), severity="error", timeout=8)
            return
        ConfigStore().save(config)
        self.notify("Config saved.", severity="information")
        self._finish()

    def action_cancel(self) -> None:
        self._finish()

    def _finish(self) -> None:
        if self.standalone:
            self.app.exit()
        else:
            self.app.pop_screen()


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        lines.append(f"{location}: {error.get('msg', '')}")
    return "Invalid config:\n" + "\n".join(lines)
