"""`bctrainctl stop` command."""

from __future__ import annotations

import typer

from bctrainctl.cli.common import get_debug, get_dlc_client, get_job_store
from bctrainctl.core.enums import JobStatus
from bctrainctl.storage.config_store import ConfigStore
from bctrainctl.utils.console import print_mapping_table


def register_stop_command(app: typer.Typer) -> None:
    @app.command("stop")
    def stop_command(ctx: typer.Context, job_id: str) -> None:
        """Stop a remote DLC job and update the local index if present."""

        config = ConfigStore().load_required()
        dlc_client = get_dlc_client(config, debug=get_debug(ctx))
        dlc_client.stop_job(job_id)

        store = get_job_store()
        record = store.get_job(job_id)
        if record:
            record = store.update_job_status(job_id=job_id, status=JobStatus.STOPPED)
            print_mapping_table(
                "Job Stopped",
                {
                    "Job ID": record.job_id,
                    "Job Name": record.job_name,
                    "Status": record.status.value,
                    "Updated At": record.updated_at,
                },
            )
            return

        print_mapping_table(
            "Remote Stop Requested",
            {
                "Job ID": job_id,
                "Status": "STOP_REQUESTED",
            },
        )

