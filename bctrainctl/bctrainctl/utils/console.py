"""Rich console helpers."""

from __future__ import annotations

from collections.abc import Mapping

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bctrainctl.utils.errors import BCTrainctlError

console = Console()
error_console = Console(stderr=True)


def print_mapping_table(title: str, values: Mapping[str, object]) -> None:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(key, "-" if value is None else str(value))
    console.print(table)


def render_bctrainctl_error(exc: BCTrainctlError) -> None:
    body = f"[bold red]{exc.message}[/bold red]"
    if exc.hint:
        body = f"{body}\n\n[cyan]Next step:[/cyan] {exc.hint}"
    error_console.print(Panel.fit(body, title="bctrainctl error", border_style="red"))

