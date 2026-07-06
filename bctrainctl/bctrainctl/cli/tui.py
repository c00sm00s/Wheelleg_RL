"""`bctrainctl tui` command: launch the interactive console."""

from __future__ import annotations

from pathlib import Path

import typer

from bctrainctl.cli.acr import run_acr_push_request
from bctrainctl.cli.common import get_debug
from bctrainctl.cli.submit import execute_submit
from bctrainctl.clients.acr_client import AcrPushRequest


def register_tui_command(app: typer.Typer) -> None:
    @app.command("tui")
    def tui_command(ctx: typer.Context) -> None:
        """Open the interactive console (config, job submission, ACR push)."""

        from bctrainctl.tui.app import run_menu_tui

        result = run_menu_tui()
        if isinstance(result, Path):
            execute_submit(ctx, file=result)
        elif isinstance(result, AcrPushRequest):
            run_acr_push_request(result, debug=get_debug(ctx))
