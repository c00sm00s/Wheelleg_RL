"""`bctrainctl logs` command."""

from __future__ import annotations

import typer

from bctrainctl.cli.common import get_debug, get_dlc_client
from bctrainctl.storage.config_store import ConfigStore
from bctrainctl.utils.console import console


def register_logs_command(app: typer.Typer) -> None:
    @app.command("logs")
    def logs_command(
        ctx: typer.Context,
        job_id: str,
        follow: bool = typer.Option(
            True,
            "--follow/--no-follow",
            help="Poll logs continuously by default. Use --no-follow for a single fetch.",
        ),
        lines: int = typer.Option(200, "--lines", min=1, help="Maximum log lines to fetch."),
    ) -> None:
        """Fetch logs from the first usable pod in a DLC job."""

        config = ConfigStore().load_required()
        dlc_client = get_dlc_client(config, debug=get_debug(ctx))
        logs = dlc_client.get_job_logs(job_id, lines=lines, follow=follow)

        if not follow:
            console.print(logs)
            return

        previous = ""
        for chunk in logs:
            if chunk.startswith(previous):
                delta = chunk[len(previous) :].lstrip("\n")
            else:
                delta = chunk
            if delta:
                console.print(delta)
            previous = chunk
