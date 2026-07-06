"""`bctrainctl submit` command."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer

from bctrainctl.cli.common import get_debug, get_dlc_client, get_job_store, get_oss_client, print_json
from bctrainctl.clients.dlc_client import ExtraDownload
from bctrainctl.core.enums import JobStatus
from bctrainctl.core.models import JobSubmission
from bctrainctl.core.packaging import package_extra_upload, package_project
from bctrainctl.core.validators import load_job_manifest
from bctrainctl.storage.config_store import ConfigStore
from bctrainctl.utils.console import console, print_mapping_table
from bctrainctl.utils.time import compact_utc_timestamp, utc_now_iso


def register_submit_command(app: typer.Typer) -> None:
    @app.command("submit")
    def submit_command(
        ctx: typer.Context,
        file: Path | None = typer.Option(
            None,
            "-f",
            "--file",
            exists=True,
            readable=True,
            help="Job manifest YAML. Omit to fill one in interactively (TUI).",
        ),
        dry_run: bool = typer.Option(False, "--dry-run", help="Validate and preview only."),
        name: str | None = typer.Option(None, "--name", help="Override metadata.name."),
    ) -> None:
        """Validate, package, upload, submit, and record a DLC training job."""

        if file is None:
            from bctrainctl.tui.app import run_submit_tui

            file = run_submit_tui()
            if file is None:
                console.print("[yellow]No job provided. Nothing submitted.[/yellow]")
                raise typer.Exit(code=0)

        execute_submit(ctx, file=file, dry_run=dry_run, name=name)


def execute_submit(
    ctx: typer.Context,
    *,
    file: Path,
    dry_run: bool = False,
    name: str | None = None,
) -> None:
    """Package, upload, and submit a job from a manifest file."""

    config = ConfigStore().load_required()
    manifest = load_job_manifest(file, config=config, name_override=name)
    if manifest.spec.resource_id and manifest.spec.storage.mounts:
        console.print(
            "[yellow]Warning:[/yellow] current workspace-bound quota jobs may fail on OSS/NAS "
            "storage mounts due to a cluster-side Fluid webhook issue. For initial debugging, "
            "prefer running without `spec.storage.mounts`."
        )
        for mount in manifest.spec.storage.mounts:
            if mount.type.value == "oss":
                console.print(
                    f"[yellow]OSS mount normalized source:[/yellow] {mount.source}"
                )
    package_result = package_project(
        project_path=manifest.spec.code.project_path,
        job_name=manifest.metadata.name,
        exclude=manifest.spec.code.exclude,
        use_gitignore=manifest.spec.code.use_gitignore,
        gitignore_include=manifest.spec.code.gitignore_include,
    )

    debug = get_debug(ctx)
    oss_client = get_oss_client(config, debug=debug)
    dlc_client = get_dlc_client(config, debug=debug)

    safe_job_name = manifest.metadata.name.replace("/", "-").replace(" ", "-")
    timestamp = compact_utc_timestamp()
    object_key = (
        f"{config.oss_prefix.strip('/')}/code_packages/"
        f"{safe_job_name}-{timestamp}-{package_result.sha256}.tar.gz"
    )
    preview_code_uri = oss_client.build_oss_uri(object_key)
    preview_download_url = oss_client.sign_download_url(object_key)

    # Package each extra upload up-front so previews and the real submission
    # share the same archives and OSS object keys.
    extra_artifacts: list[dict[str, object]] = []
    for index, upload in enumerate(manifest.spec.extra_uploads):
        extra_result = package_extra_upload(
            local_path=upload.local,
            label=f"{safe_job_name}-{index}",
        )
        extra_key = (
            f"{config.oss_prefix.strip('/')}/extra_uploads/"
            f"{safe_job_name}-{timestamp}-{index}-{extra_result.sha256}.tar.gz"
        )
        extra_artifacts.append(
            {
                "upload": upload,
                "result": extra_result,
                "object_key": extra_key,
                "download_url": oss_client.sign_download_url(extra_key),
            }
        )

    extra_downloads = [
        ExtraDownload(
            filename=f"extra_{index}.tar.gz",
            download_url=str(artifact["download_url"]),
            target=artifact["upload"].target,  # type: ignore[union-attr]
        )
        for index, artifact in enumerate(extra_artifacts)
    ]

    _, request_snapshot = dlc_client.build_create_job_request(
        manifest,
        preview_code_uri,
        preview_download_url,
        extra_downloads,
    )

    if dry_run:
        print_mapping_table(
            "Submit Dry Run",
            {
                "Job Name": manifest.metadata.name,
                "Project": manifest.metadata.project,
                "Package Path": package_result.package_path,
                "Package SHA256": package_result.sha256,
                "Package Files": package_result.file_count,
                "Preview OSS URI": preview_code_uri,
                "Extra Uploads": len(extra_artifacts) or "-",
                "Image": manifest.spec.image,
                "Entrypoint": manifest.spec.runtime.entrypoint,
            },
        )
        print_json("CreateJob Request", request_snapshot)
        return

    oss_code_uri = oss_client.upload_file(package_result.package_path, object_key)
    for artifact in extra_artifacts:
        oss_client.upload_file(
            artifact["result"].package_path,  # type: ignore[union-attr]
            str(artifact["object_key"]),
        )
        console.print(
            f"[dim]Uploaded extra:[/dim] {artifact['upload'].local} "  # type: ignore[union-attr]
            f"-> {artifact['upload'].target}"  # type: ignore[union-attr]
        )
    for mount in manifest.spec.storage.mounts:
        if mount.type.value != "oss":
            continue
        placeholder_uri = oss_client.ensure_prefix_placeholder(mount.source)
        console.print(f"[dim]Prepared OSS mount prefix:[/dim] {placeholder_uri}")
    code_download_url = oss_client.sign_download_url(object_key)
    job_id, request_snapshot, create_response = dlc_client.create_job(
        manifest,
        oss_code_uri,
        code_download_url,
        extra_downloads,
    )
    submission = JobSubmission(
        submission_id=f"sub-{uuid4().hex[:12]}",
        package_path=package_result.package_path,
        package_hash=package_result.sha256,
        oss_code_uri=oss_code_uri,
        submitted_at=utc_now_iso(),
        dlc_job_id=job_id,
        request_snapshot=request_snapshot,
    )
    record = get_job_store().create_record(
        manifest=manifest,
        submission=submission,
        status=JobStatus.SUBMITTED,
        remote_payload=create_response,
    )

    print_mapping_table(
        "Submit Result",
        {
            "Job Name": record.job_name,
            "DLC Job ID": record.job_id,
            "Status": record.status.value,
            "OSS Code URI": record.oss_code_uri,
            "Entrypoint": record.entrypoint,
            "Image": record.image,
            "DLC Quota": manifest.spec.resource_id,
            "Priority": manifest.spec.priority if manifest.spec.priority is not None else "DEFAULT(1)",
            "Resources": manifest.spec.resources,
            "Package Path": record.package_path,
            "Logs Command": f"bctrainctl logs {record.job_id}",
        },
    )
