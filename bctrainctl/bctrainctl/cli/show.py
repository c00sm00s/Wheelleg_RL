"""`bctrainctl show` command."""

from __future__ import annotations

import json
from typing import Any

import typer

from bctrainctl.cli.common import get_config_required, get_debug, get_dlc_client, get_job_store
from bctrainctl.utils.console import print_mapping_table


def _extract_local_spec(record: Any) -> dict[str, Any] | None:
    if record is None:
        return None
    return json.loads(record.raw_spec_json)


def _extract_remote_job_spec(remote_job: dict[str, Any]) -> dict[str, Any]:
    job_specs = remote_job.get("JobSpecs") or []
    if isinstance(job_specs, list) and job_specs:
        first = job_specs[0]
        if isinstance(first, dict):
            return first
    return {}


def _extract_resources(
    *,
    local_spec: dict[str, Any] | None,
    remote_job: dict[str, Any],
) -> Any:
    if local_spec is not None:
        resources = ((local_spec.get("spec") or {}).get("resources"))
        if resources:
            return resources

    remote_job_spec = _extract_remote_job_spec(remote_job)
    remote_resource_config = remote_job_spec.get("ResourceConfig") or {}
    if remote_resource_config:
        return {
            "gpu": remote_resource_config.get("GPU"),
            "cpu": remote_resource_config.get("CPU"),
            "memory": remote_resource_config.get("Memory"),
        }

    return {
        "gpu": remote_job.get("RequestGPU"),
        "cpu": remote_job.get("RequestCPU"),
        "memory": remote_job.get("RequestMemory"),
    }


def _extract_image(
    *,
    local_record: Any,
    remote_job: dict[str, Any],
) -> str | None:
    if local_record is not None and local_record.image:
        return local_record.image
    remote_job_spec = _extract_remote_job_spec(remote_job)
    return remote_job_spec.get("Image")


def _extract_entrypoint(local_record: Any) -> str | None:
    if local_record is None:
        return None
    return local_record.entrypoint


def _extract_quota_selector(
    *,
    local_spec: dict[str, Any] | None,
    remote_job: dict[str, Any],
) -> str | None:
    if local_spec is not None:
        resource_id = (local_spec.get("spec") or {}).get("resource_id")
        if resource_id:
            return resource_id
    return remote_job.get("ResourceQuotaName") or remote_job.get("ResourceName")


def _extract_priority(
    *,
    local_spec: dict[str, Any] | None,
    remote_job: dict[str, Any],
) -> Any:
    remote_priority = remote_job.get("Priority")
    if remote_priority is not None:
        return remote_priority
    if local_spec is not None:
        return (local_spec.get("spec") or {}).get("priority")
    return None


def register_show_command(app: typer.Typer) -> None:
    @app.command("show")
    def show_command(ctx: typer.Context, job_id: str) -> None:
        """Show the latest cloud details for a job, merged with local metadata if tracked."""

        config = get_config_required()
        store = get_job_store()
        local_record = store.get_job(job_id)

        dlc_client = get_dlc_client(config, debug=get_debug(ctx))
        remote_job = dlc_client.get_job(job_id)
        remote_status = dlc_client.map_remote_status(str(remote_job.get("Status") or ""))

        if local_record is not None:
            local_record = store.update_job_status(
                job_id=job_id,
                status=remote_status,
                remote_payload=remote_job,
            )

        local_spec = _extract_local_spec(local_record)
        image = _extract_image(local_record=local_record, remote_job=remote_job)
        entrypoint = _extract_entrypoint(local_record)
        resources = _extract_resources(local_spec=local_spec, remote_job=remote_job)
        quota_selector = _extract_quota_selector(local_spec=local_spec, remote_job=remote_job)
        priority = _extract_priority(local_spec=local_spec, remote_job=remote_job)

        print_mapping_table(
            "Job Details",
            {
                "Tracked Locally": "YES" if local_record is not None else "NO",
                "Local ID": local_record.local_id if local_record is not None else None,
                "Job ID": str(remote_job.get("JobId") or job_id),
                "Job Name": str(
                    (local_record.job_name if local_record is not None else None)
                    or remote_job.get("DisplayName")
                    or job_id
                ),
                "Project": local_record.project if local_record is not None else "-",
                "Status": remote_status.value,
                "Remote Status": remote_job.get("Status"),
                "Remote SubStatus": remote_job.get("SubStatus"),
                "Reason Code": remote_job.get("ReasonCode"),
                "Reason Message": remote_job.get("ReasonMessage"),
                "Job Type": remote_job.get("JobType"),
                "Workspace ID": remote_job.get("WorkspaceId") or config.workspace_id,
                "DLC Quota": quota_selector,
                "DLC Quota ID": remote_job.get("ResourceId"),
                "DLC Resource Name": remote_job.get("ResourceName"),
                "Priority": priority,
                "Image": image,
                "EntryPoint": entrypoint,
                "Resources": resources,
                "OSS Code URI": local_record.oss_code_uri if local_record is not None else None,
                "Package Hash": local_record.package_hash if local_record is not None else None,
                "Created At": remote_job.get("GmtCreateTime") or (
                    local_record.created_at if local_record is not None else None
                ),
                "Submitted At": remote_job.get("GmtSubmittedTime"),
                "Running At": remote_job.get("GmtRunningTime"),
                "Finished At": remote_job.get("GmtFinishTime"),
                "Remote Updated At": remote_job.get("GmtModifiedTime"),
                "Last Synced At": local_record.last_synced_at if local_record is not None else None,
            },
        )
