"""Shared CLI helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import typer
from rich.table import Table

from bctrainctl.clients.dlc_client import DlcClient
from bctrainctl.clients.oss_client import OssClient
from bctrainctl.storage.config_store import ConfigStore
from bctrainctl.storage.job_store import JobStore
from bctrainctl.utils.console import console


def get_debug(ctx: typer.Context) -> bool:
    return bool(getattr(ctx.obj, "debug", False))


def get_config_store() -> ConfigStore:
    return ConfigStore()


def get_job_store() -> JobStore:
    return JobStore()


def get_config_required() -> Any:
    return get_config_store().load_required()


def get_dlc_client(config: Any, *, debug: bool) -> DlcClient:
    return DlcClient(config, debug=debug)


def get_oss_client(config: Any, *, debug: bool) -> OssClient:
    return OssClient(config, debug=debug)


def print_json(title: str, payload: Mapping[str, Any]) -> None:
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))


def render_jobs_table(rows: list[dict[str, object]]) -> None:
    table = Table(title="Jobs", header_style="bold cyan")
    table.add_column("Job ID", no_wrap=True)
    table.add_column("Name")
    table.add_column("Project")
    table.add_column("Priority")
    table.add_column("Status")
    table.add_column("Updated")
    for row in rows:
        table.add_row(
            str(row.get("job_id", "-")),
            str(row.get("job_name", "-")),
            str(row.get("project", "-")),
            str(row.get("priority", "-")),
            str(row.get("status", "-")),
            str(row.get("updated_at", "-")),
        )
    console.print(table)
