"""`bctrainctl delete` command."""

from __future__ import annotations

import typer

from bctrainctl.cli.common import get_debug, get_dlc_client, get_job_store, get_oss_client
from bctrainctl.storage.config_store import ConfigStore
from bctrainctl.utils.console import print_mapping_table


def register_delete_command(app: typer.Typer) -> None:
    @app.command("delete")
    def delete_command(ctx: typer.Context, job_id: str) -> None:
        """Delete a DLC job remotely and remove its local record."""

        config = ConfigStore().load_required()
        store = get_job_store()
        existing_record = store.get_job(job_id)
        dlc_client = get_dlc_client(config, debug=get_debug(ctx))
        dlc_client.delete_job(job_id)

        oss_status = "SKIPPED"
        if existing_record is not None and existing_record.oss_code_uri:
            oss_client = get_oss_client(config, debug=get_debug(ctx))
            oss_client.delete_oss_uri(existing_record.oss_code_uri)
            oss_status = "DELETED"

        deleted_record = store.delete_job(job_id)

        print_mapping_table(
            "Job Deleted",
            {
                "Job ID": job_id,
                "Remote": "DELETED",
                "OSS Code Package": oss_status,
                "Local Record": "DELETED" if deleted_record is not None else "NOT_FOUND",
                "Job Name": deleted_record.job_name if deleted_record is not None else "-",
            },
        )
