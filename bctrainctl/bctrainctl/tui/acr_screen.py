"""Textual screen for assembling an ACR docker push."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Switch

from bctrainctl.clients.acr_client import (
    DEFAULT_NAMESPACE,
    DEFAULT_REGISTRY,
    AcrClient,
    AcrPushRequest,
    split_image_ref,
)
from bctrainctl.storage.config_store import ConfigStore


class AcrPushScreen(Screen):
    """Pick a local image and ACR target; the push itself runs after exit."""

    BINDINGS = [
        ("ctrl+s", "push", "Push"),
        ("escape", "cancel", "Back"),
    ]

    def __init__(self, *, standalone: bool = True) -> None:
        super().__init__()
        self.standalone = standalone

    def compose(self) -> ComposeResult:
        yield Header()
        config = ConfigStore().load()
        registry = (config.acr_registry if config else None) or DEFAULT_REGISTRY
        namespace = (config.acr_namespace if config else None) or DEFAULT_NAMESPACE
        username = (config.acr_username if config else None) or ""
        password = (
            config.acr_password.get_secret_value() if config and config.acr_password else ""
        )
        with VerticalScroll(id="acr-form"):
            yield Label("[b]Push Docker image to ACR[/b]")
            yield Label("Local image (pick one)")
            yield Select([], prompt="(refresh to list local images)", id="acr-select", allow_blank=True)
            yield Label("...or type an image ref")
            yield Input(placeholder="repo:tag", id="field-image")
            yield Label("Registry")
            yield Input(value=registry, id="field-registry")
            yield Label("Namespace")
            yield Input(value=namespace, id="field-namespace")
            yield Label("Username")
            yield Input(value=username, id="field-username")
            yield Label("Password")
            yield Input(value=password, password=True, id="field-password")
            yield Label("Remote image name (blank = same as local)")
            yield Input(id="field-name")
            yield Label("Remote tag (blank = same as local)")
            yield Input(id="field-tag")
            yield Label("Use sudo for docker")
            yield Switch(value=False, id="field-sudo")
        with Horizontal(id="acr-buttons"):
            yield Button("Push", variant="success", id="push")
            yield Button("Refresh images", variant="primary", id="refresh")
            yield Button("Back", variant="default", id="cancel")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_images()

    # --- events -----------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "push":
            self.action_push()
        elif event.button.id == "refresh":
            self._refresh_images()
        elif event.button.id == "cancel":
            self.action_cancel()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "acr-select" and event.value not in (None, Select.BLANK):
            name, tag = split_image_ref(str(event.value))
            self.query_one("#field-name", Input).value = name
            self.query_one("#field-tag", Input).value = tag

    # --- actions ----------------------------------------------------------
    def action_push(self) -> None:
        try:
            request = self._build_request()
        except ValueError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.app.exit(request)

    def action_cancel(self) -> None:
        if self.standalone:
            self.app.exit(None)
        else:
            self.app.pop_screen()

    # --- helpers ----------------------------------------------------------
    def _refresh_images(self) -> None:
        use_sudo = self.query_one("#field-sudo", Switch).value
        try:
            images = AcrClient(use_sudo=use_sudo).list_images()
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Cannot list local images: {exc}", severity="warning", timeout=6)
            images = []
        self.query_one("#acr-select", Select).set_options((ref, ref) for ref in images)
        if not images:
            self.notify("No local images found. Type an image ref instead.", severity="information")

    def _selected_source(self) -> str:
        manual = self.query_one("#field-image", Input).value.strip()
        if manual:
            return manual
        value = self.query_one("#acr-select", Select).value
        return "" if value in (None, Select.BLANK) else str(value)

    def _build_request(self) -> AcrPushRequest:
        source = self._selected_source()
        if not source:
            raise ValueError("Select a local image or type an image ref.")
        registry = self.query_one("#field-registry", Input).value.strip() or DEFAULT_REGISTRY
        namespace = self.query_one("#field-namespace", Input).value.strip() or DEFAULT_NAMESPACE
        username = self.query_one("#field-username", Input).value.strip()
        password = self.query_one("#field-password", Input).value
        if not username:
            raise ValueError("ACR username is required.")
        if not password:
            raise ValueError("ACR password is required.")
        default_name, default_tag = split_image_ref(source)
        name = self.query_one("#field-name", Input).value.strip() or default_name
        tag = self.query_one("#field-tag", Input).value.strip() or default_tag
        return AcrPushRequest(
            source=source,
            registry=registry,
            namespace=namespace,
            username=username,
            password=password,
            name=name,
            tag=tag,
            use_sudo=self.query_one("#field-sudo", Switch).value,
        )
