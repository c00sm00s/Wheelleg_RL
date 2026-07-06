"""CLI entrypoint for bctrainctl."""

from __future__ import annotations

import traceback
from dataclasses import dataclass

import typer

from bctrainctl.cli.acr import register_acr_commands
from bctrainctl.cli.config import register_config_commands
from bctrainctl.cli.delete import register_delete_command
from bctrainctl.cli.init import register_init_command
from bctrainctl.cli.list import register_list_command
from bctrainctl.cli.logs import register_logs_command
from bctrainctl.cli.show import register_show_command
from bctrainctl.cli.stop import register_stop_command
from bctrainctl.cli.submit import register_submit_command
from bctrainctl.cli.sync import register_sync_command
from bctrainctl.cli.tui import register_tui_command
from bctrainctl.utils.console import error_console, render_bctrainctl_error
from bctrainctl.utils.errors import BCTrainctlError


DEBUG_MODE = False


@dataclass(slots=True)
class AppContext:
    debug: bool = False


# `-h` is aliased to `--help` everywhere; the setting is inherited by all
# subcommands through the click context.
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    no_args_is_help=True,
    help="Manage Alibaba Cloud PAI DLC training jobs from local CLI.",
    pretty_exceptions_enable=False,
    context_settings=CONTEXT_SETTINGS,
)
config_app = typer.Typer(help="Inspect and manage local configuration.")
app.add_typer(config_app, name="config")
acr_app = typer.Typer(
    no_args_is_help=True,
    help="Push Docker images to Alibaba Cloud ACR.",
)
app.add_typer(acr_app, name="acr")


@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Print debug details."),
) -> None:
    """Initialize global CLI context."""
    global DEBUG_MODE
    DEBUG_MODE = debug
    ctx.obj = AppContext(debug=debug)


# `tui` is registered first so it shows at the top of `bctrainctl --help`.
register_tui_command(app)
register_init_command(app)
register_config_commands(config_app)
register_acr_commands(acr_app)
register_submit_command(app)
register_list_command(app)
register_show_command(app)
register_logs_command(app)
register_stop_command(app)
register_delete_command(app)
register_sync_command(app)


def run() -> None:
    # This is the console-script entry point, called from `sys.exit(run())`. We
    # exit via SystemExit (not typer.Exit / click.exceptions.Exit, which is only
    # meaningful inside the typer invocation); raising typer.Exit here would
    # escape as an uncaught exception and dump a full traceback on the user.
    try:
        app()
    except SystemExit:
        raise
    except BCTrainctlError as exc:
        if DEBUG_MODE:
            traceback.print_exc()
        render_bctrainctl_error(exc)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        error_console.print("[yellow]Interrupted by user.[/yellow]")
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001
        if DEBUG_MODE:
            traceback.print_exc()
        error_console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise SystemExit(1) from None
