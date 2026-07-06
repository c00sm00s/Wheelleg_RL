"""`bctrainctl config` commands."""

from __future__ import annotations

import typer
from pydantic import ValidationError

from bctrainctl.core.models import AppConfig
from bctrainctl.storage.config_store import ConfigStore
from bctrainctl.utils.console import console, print_mapping_table
from bctrainctl.utils.errors import ValidationFailedError

# Fields editable via `config set`, mapped to whether they hold a secret value.
EDITABLE_FIELDS: dict[str, bool] = {
    "access_key_id": False,
    "access_key_secret": True,
    "region": False,
    "workspace_id": False,
    "oss_bucket": False,
    "oss_prefix": False,
    "default_resource_id": False,
    "default_image": False,
    "wandb_api_key": True,
    "swanlab_api_key": True,
    "acr_registry": False,
    "acr_namespace": False,
    "acr_username": False,
    "acr_password": True,
}


def _show_current_config() -> None:
    config = ConfigStore().load_required()
    print_mapping_table("Current Config", config.masked_dict())


def register_config_commands(app: typer.Typer) -> None:
    @app.callback(invoke_without_command=True)
    def config_root(ctx: typer.Context) -> None:
        """Inspect and manage local configuration.

        Running `bctrainctl config` with no subcommand prints the current config.
        """

        if ctx.invoked_subcommand is not None:
            return
        _show_current_config()
        console.print(
            "\n[dim]Usage: bctrainctl config [OPTIONS] COMMAND [ARGS]...[/dim]\n"
            "[dim]Commands: show | set | edit[/dim]"
        )

    @app.command("show")
    def show_config() -> None:
        """Print the current config with secrets redacted."""

        _show_current_config()

    @app.command("set")
    def set_config(
        key: str = typer.Argument(..., help="Config field to update."),
        value: str = typer.Argument(..., help="New value for the field."),
    ) -> None:
        """Update a single config field, e.g. `config set wandb_api_key xxxx`."""

        if key not in EDITABLE_FIELDS:
            raise ValidationFailedError(
                f"Unknown or read-only config field: {key}",
                hint=f"Editable fields: {', '.join(sorted(EDITABLE_FIELDS))}.",
            )

        store = ConfigStore()
        config = store.load_required()
        payload = config.plain_dict()
        payload[key] = value
        try:
            updated = AppConfig.model_validate(payload)
        except ValidationError as exc:
            raise ValidationFailedError(
                f"Config update failed:\n{exc}",
                hint="Check the field value and try again.",
            ) from exc
        store.save(updated)

        shown = "********" if EDITABLE_FIELDS[key] else value
        console.print(f"[green]Updated[/green] {key} = {shown}")

    @app.command("edit")
    def edit_config() -> None:
        """Open the interactive TUI to edit configuration."""

        from bctrainctl.tui.app import run_config_tui

        run_config_tui()
