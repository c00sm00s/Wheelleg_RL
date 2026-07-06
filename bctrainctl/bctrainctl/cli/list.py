"""`bctrainctl list` command."""

from __future__ import annotations

import json

import typer

from bctrainctl.clients.dlc_client import DlcClient
from bctrainctl.core.models import JobRecord
from bctrainctl.cli.common import get_config_store, get_debug, get_dlc_client, get_job_store, render_jobs_table


def _remote_updated_at(job: dict[str, object]) -> str:
    for field in ("GmtFinishTime", "GmtRunningTime", "GmtSubmittedTime", "GmtCreateTime"):
        value = job.get(field)
        if value:
            return str(value)
    return "-"


def _local_priority(record: JobRecord) -> object:
    try:
        spec = json.loads(record.raw_spec_json)
    except json.JSONDecodeError:
        return "-"
    priority = ((spec.get("spec") or {}).get("priority"))
    return priority if priority is not None else "DEFAULT(1)"


def _priority_for_record(
    record: JobRecord,
    *,
    remote_by_id: dict[str, dict[str, object]],
) -> object:
    remote_job = remote_by_id.get(record.job_id)
    if remote_job is not None and remote_job.get("Priority") is not None:
        return remote_job.get("Priority")
    return _local_priority(record)


def _refresh_local_records(
    records: list[JobRecord],
    *,
    remote_jobs: list[dict[str, object]],
    dlc_client: DlcClient,
) -> list[JobRecord]:
    if not records:
        return []

    job_store = get_job_store()
    remote_by_id = {str(job.get("JobId")): job for job in remote_jobs if job.get("JobId")}
    refreshed: list[JobRecord] = []
    for record in records:
        payload = remote_by_id.get(record.job_id)
        if payload is None:
            refreshed.append(record)
            continue
        status = dlc_client.map_remote_status(str(payload.get("Status") or ""))
        refreshed.append(
            job_store.update_job_status(
                job_id=record.job_id,
                status=status,
                remote_payload=payload,
            )
        )
    return refreshed


def register_list_command(app: typer.Typer) -> None:
    @app.command("list")
    def list_command(
        ctx: typer.Context,
        remote: bool = typer.Option(
            False,
            "--remote",
            help="List all jobs from the current workspace, not only locally tracked jobs.",
        ),
    ) -> None:
        """List jobs, syncing local status from DLC by default."""

        job_store = get_job_store()
        records = job_store.list_jobs()
        dlc_client = None
        remote_jobs: list[dict[str, object]] = []

        if remote or records:
            config = get_config_store().load_required()
            dlc_client = get_dlc_client(config, debug=get_debug(ctx))
            remote_jobs = dlc_client.list_all_jobs(page_size=max(100, len(records) * 2 or 100))
            records = _refresh_local_records(records, remote_jobs=remote_jobs, dlc_client=dlc_client)

        if remote and dlc_client is not None:
            local_by_id = {record.job_id: record for record in records}
            rows = []
            for job in remote_jobs:
                job_id = str(job.get("JobId") or "-")
                local_record = local_by_id.get(job_id)
                if local_record is not None:
                    rows.append(
                        {
                            "job_id": local_record.job_id,
                            "job_name": local_record.job_name,
                            "project": local_record.project,
                            "priority": job.get("Priority", _local_priority(local_record)),
                            "status": local_record.status.value,
                            "updated_at": local_record.updated_at,
                        }
                    )
                    continue
                rows.append(
                    {
                        "job_id": job_id,
                        "job_name": str(job.get("DisplayName") or job_id),
                        "project": "-",
                        "priority": job.get("Priority", "-"),
                        "status": dlc_client.map_remote_status(str(job.get("Status") or "")).value,
                        "updated_at": _remote_updated_at(job),
                    }
                )
            render_jobs_table(rows)
            return

        remote_by_id = {str(job.get("JobId")): job for job in remote_jobs if job.get("JobId")}
        rows = [
            {
                "job_id": record.job_id,
                "job_name": record.job_name,
                "project": record.project,
                "priority": _priority_for_record(record, remote_by_id=remote_by_id),
                "status": record.status.value,
                "updated_at": record.updated_at,
            }
            for record in records
        ]
        render_jobs_table(rows)
