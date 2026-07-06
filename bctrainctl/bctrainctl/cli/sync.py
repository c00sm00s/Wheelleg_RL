"""`bctrainctl sync` command."""

from __future__ import annotations

import typer

from bctrainctl.cli.common import get_debug, get_dlc_client, get_job_store
from bctrainctl.storage.config_store import ConfigStore
from bctrainctl.utils.console import print_mapping_table


def register_sync_command(app: typer.Typer) -> None:
    @app.command("sync")
    def sync_command(ctx: typer.Context) -> None:
        """Refresh all locally tracked jobs from DLC."""

        store = get_job_store()
        records = store.list_jobs()
        config = ConfigStore().load_required()
        dlc_client = get_dlc_client(config, debug=get_debug(ctx))
        remote_jobs = {
            job.get("JobId"): job
            for job in dlc_client.list_all_jobs(page_size=max(100, len(records) * 2 or 100))
        }

        updated = 0
        for record in records:
            payload = remote_jobs.get(record.job_id)
            if payload is None:
                continue
            status = dlc_client.map_remote_status(payload.get("Status"))
            store.update_job_status(job_id=record.job_id, status=status, remote_payload=payload)
            updated += 1

        print_mapping_table(
            "Sync Complete",
            {
                "Tracked Jobs": len(records),
                "Updated Jobs": updated,
            },
        )
