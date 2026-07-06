"""Textual application shell and entry points for the bctrainctl TUI."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, OptionList, TextArea

from bctrainctl.clients.acr_client import AcrPushRequest
from bctrainctl.tui.acr_screen import AcrPushScreen
from bctrainctl.tui.config_screen import ConfigScreen
from bctrainctl.tui.submit_screen import SubmitScreen


class MenuScreen(Screen):
    """Main menu: pick configuration editing, a new job, or an ACR push."""

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="menu"):
            yield Label("[b]bctrainctl[/b] — interactive console\n")
            yield Button("New training job", variant="success", id="submit")
            yield Button("Push Docker image to ACR", variant="warning", id="acr")
            yield Button("Edit configuration", variant="primary", id="config")
            yield Button("Quit", variant="default", id="quit")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "config":
            self.app.push_screen(ConfigScreen(standalone=False))
        elif event.button.id == "submit":
            self.app.push_screen(SubmitScreen(standalone=False))
        elif event.button.id == "acr":
            self.app.push_screen(AcrPushScreen(standalone=False))
        elif event.button.id == "quit":
            self.action_quit()

    def action_quit(self) -> None:
        # Defined here because the binding lives on this pushed screen, where the
        # App's built-in `action_quit` is not in scope.
        self.app.exit(None)


class BctrainctlApp(App):
    """Single textual app that can boot into the menu, config, or submit screen."""

    # Up/Down move focus between fields (in addition to Tab/Shift+Tab). These are
    # priority bindings so they win over the form's scroll container, which would
    # otherwise consume the arrows. `check_action` disables them while a TextArea
    # or an open dropdown is focused, where the arrows have their own meaning.
    BINDINGS = [
        Binding("down", "focus_next", "Next field", priority=True),
        Binding("up", "focus_previous", "Prev field", priority=True),
    ]

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool:
        if action in ("focus_next", "focus_previous"):
            return not isinstance(self.focused, (TextArea, OptionList))
        return True

    CSS = """
    #menu { padding: 2 4; }
    #menu Button { margin: 1 0; width: 40; }
    #config-form, #submit-form, #acr-form { padding: 1 2; }
    #config-buttons, #submit-buttons, #acr-buttons { height: auto; padding: 1 2; }
    #config-buttons Button, #submit-buttons Button, #acr-buttons Button { margin: 0 1; }
    Label { margin: 1 0 0 0; }
    /* TextArea collapses to height 0 inside a scroll container unless sized. */
    TextArea { height: 5; border: tall $accent; }
    """

    TITLE = "bctrainctl"

    def __init__(self, start: str = "menu") -> None:
        super().__init__()
        self._start = start

    def on_mount(self) -> None:
        if self._start == "config":
            self.push_screen(ConfigScreen(standalone=True))
        elif self._start == "submit":
            self.push_screen(SubmitScreen(standalone=True))
        elif self._start == "acr":
            self.push_screen(AcrPushScreen(standalone=True))
        else:
            self.push_screen(MenuScreen())


def run_config_tui() -> None:
    """Launch the TUI directly on the configuration screen."""

    BctrainctlApp(start="config").run()


def run_submit_tui() -> Path | None:
    """Launch the submit screen; return a manifest path to submit, or None."""

    result = BctrainctlApp(start="submit").run()
    return result if isinstance(result, Path) else None


def run_acr_tui() -> AcrPushRequest | None:
    """Launch the ACR push screen; return a push request, or None."""

    result = BctrainctlApp(start="acr").run()
    return result if isinstance(result, AcrPushRequest) else None


def run_menu_tui() -> Path | AcrPushRequest | None:
    """Launch the main menu; return a submit manifest path or an ACR push request."""

    result = BctrainctlApp(start="menu").run()
    if isinstance(result, (Path, AcrPushRequest)):
        return result
    return None
