"""PAI DLC wrapper built on top of the Alibaba Cloud Python SDK."""

from __future__ import annotations

import shlex
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from alibabacloud_pai_dlc20201203.client import Client as DlcSdkClient
from alibabacloud_pai_dlc20201203.models import (
    CreateJobRequest,
    CreateJobRequestDataSources,
    GetJobRequest,
    GetPodLogsRequest,
    JobSpec,
    ListJobsRequest,
    ResourceConfig,
)
from alibabacloud_tea_openapi.models import Config as OpenApiConfig

from bctrainctl.clients.oss_client import OssClient
from bctrainctl.clients.workspace_client import WorkspaceClient
from bctrainctl.core.enums import JobStatus
from bctrainctl.core.models import AppConfig, JobManifest
from bctrainctl.core.status_mapper import map_remote_status
from bctrainctl.utils.console import console
from bctrainctl.utils.credentials import resolve_credentials
from bctrainctl.utils.errors import ClientOperationError


@dataclass(frozen=True, slots=True)
class ExtraDownload:
    """An extra archive to fetch and extract into the container before launch."""

    filename: str
    download_url: str
    target: str


class DlcClient:
    """Encapsulate DLC SDK operations and request assembly."""

    def __init__(self, config: AppConfig, *, debug: bool = False) -> None:
        self.config = config
        self.debug = debug
        credentials = resolve_credentials(config)
        self.workspace_client = WorkspaceClient(config, debug=debug)
        sdk_config = OpenApiConfig(
            access_key_id=credentials.access_key_id,
            access_key_secret=credentials.access_key_secret,
            region_id=config.region,
        )
        self.client = DlcSdkClient(sdk_config)

    def test_connectivity(self) -> None:
        self.list_jobs(page_size=1, page_number=1)

    def build_create_job_request(
        self,
        manifest: JobManifest,
        code_uri: str,
        code_download_url: str,
        extra_downloads: list[ExtraDownload] | None = None,
    ) -> tuple[CreateJobRequest, dict[str, Any]]:
        resource_spec = manifest.spec.resources
        if resource_spec is None:
            raise ClientOperationError(
                "Job resources are missing after validation.",
                hint="This indicates a local validation bug. Re-check the manifest resolution step.",
            )
        quota_selector = manifest.spec.resource_id
        if not quota_selector:
            raise ClientOperationError(
                "A workspace-bound DLC quota is required.",
                hint="Set `spec.resource_id` in job.yaml or `default_resource_id` in bctrainctl config. Use a bound quota such as `quota-4090`.",
            )
        resolved_resource_id, resolved_resource_name = self.workspace_client.resolve_dlc_quota_id(
            quota_selector
        )

        code_filename = OssClient.basename_from_oss_uri(code_uri)

        data_sources = [
            CreateJobRequestDataSources(
                uri=mount.source,
                mount_path=mount.target,
                mount_access="RO" if mount.read_only else "RW",
            )
            for mount in manifest.spec.storage.mounts
        ]

        user_command = self._build_user_command(
            workdir=manifest.spec.runtime.workdir,
            entrypoint=manifest.spec.runtime.entrypoint,
            code_filename=code_filename,
            code_download_url=code_download_url,
            extra_downloads=extra_downloads or [],
        )

        job_spec = JobSpec(
            type="Worker",
            image=manifest.spec.image,
            pod_count=1,
            is_chief=True,
            restart_policy="Never",
            resource_config=ResourceConfig(
                cpu=str(resource_spec.cpu),
                gpu=str(resource_spec.gpu),
                memory=f"{resource_spec.memory_gb}Gi",
            ),
        )

        request = CreateJobRequest(
            display_name=manifest.metadata.name,
            job_type="PyTorchJob",
            workspace_id=self.config.workspace_id,
            resource_id=resolved_resource_id,
            priority=manifest.spec.priority,
            envs=manifest.spec.runtime.env,
            user_command=user_command,
            data_sources=data_sources,
            job_specs=[job_spec],
        )
        snapshot = request.to_map()
        snapshot["ResolvedResource"] = {
            "selector": quota_selector,
            "resource_id": resolved_resource_id,
            "resource_name": resolved_resource_name,
        }
        return request, snapshot

    def create_job(
        self,
        manifest: JobManifest,
        code_uri: str,
        code_download_url: str,
        extra_downloads: list[ExtraDownload] | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        request, snapshot = self.build_create_job_request(
            manifest, code_uri, code_download_url, extra_downloads
        )
        self._debug("CreateJob request", snapshot)
        try:
            response = self.client.create_job(request)
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                f"Failed to create DLC job for {manifest.metadata.name}: {exc}",
                hint="Check RAM permissions, workspace ID, image address, selected EcsSpec, and DLC quota availability.",
            ) from exc

        body = response.body.to_map()
        self._debug("CreateJob response", body)
        job_id = body.get("JobId")
        if not job_id:
            raise ClientOperationError(
                "DLC create_job succeeded but no JobId was returned.",
                hint="Re-run with `--debug` and inspect the raw response.",
            )
        return job_id, snapshot, body

    def get_job(self, job_id: str) -> dict[str, Any]:
        request = GetJobRequest(need_detail=True)
        self._debug("GetJob request", {"job_id": job_id, "need_detail": True})
        try:
            response = self.client.get_job(job_id, request)
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                f"Failed to fetch job details: {job_id}",
                hint="Check whether the job exists in the configured workspace and whether your RAM user can access it.",
            ) from exc
        body = response.body.to_map()
        self._debug("GetJob response", body)
        return body

    def list_jobs(
        self,
        *,
        page_size: int = 100,
        page_number: int = 1,
        job_id: str | None = None,
    ) -> list[dict[str, Any]]:
        request = ListJobsRequest(
            workspace_id=self.config.workspace_id,
            show_own=True,
            sort_by="GmtCreateTime",
            order="desc",
            start_time="1970-01-01T00:00:00Z",
            page_size=page_size,
            page_number=page_number,
            job_id=job_id,
        )
        self._debug("ListJobs request", request.to_map())
        try:
            response = self.client.list_jobs(request)
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                "Failed to query DLC job list.",
                hint="Check AccessKey permissions, region, workspace ID, and DLC API reachability.",
            ) from exc
        body = response.body.to_map()
        self._debug("ListJobs response", body)
        return body.get("Jobs", [])

    def list_all_jobs(self, *, page_size: int = 100) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        page_number = 1
        while True:
            page = self.list_jobs(page_size=page_size, page_number=page_number)
            jobs.extend(page)
            if len(page) < page_size:
                break
            page_number += 1
        return jobs

    def stop_job(self, job_id: str) -> None:
        self._debug("StopJob request", {"job_id": job_id})
        try:
            response = self.client.stop_job(job_id)
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                f"Failed to stop job: {job_id}",
                hint="Check whether the job exists and is still in a stoppable state.",
            ) from exc
        self._debug("StopJob response", response.body.to_map())

    def delete_job(self, job_id: str) -> None:
        self._debug("DeleteJob request", {"job_id": job_id})
        try:
            response = self.client.delete_job(job_id)
        except Exception as exc:  # noqa: BLE001
            raise ClientOperationError(
                f"Failed to delete job: {job_id}",
                hint="Check whether the job exists, whether it has already been deleted, and whether your RAM user can delete DLC jobs in this workspace.",
            ) from exc
        self._debug("DeleteJob response", response.body.to_map())

    def get_job_logs(
        self,
        job_id: str,
        *,
        lines: int = 200,
        follow: bool = True,
    ) -> str | Iterator[str]:
        job = self.get_job(job_id)
        pods = job.get("Pods") or []
        if not pods:
            raise ClientOperationError(
                f"No running or historical pods were found for job: {job_id}",
                hint="The job may not have started yet, or pod details may be unavailable in the current workspace.",
            )

        pod = next((item for item in pods if item.get("PodId")), None)
        if pod is None:
            raise ClientOperationError(
                f"No usable pod with PodId was found for job: {job_id}",
                hint="Re-run with `--debug` to inspect the raw GetJob response and pod payloads.",
            )
        pod_id = pod.get("PodId")
        pod_uid = pod.get("PodUid")
        if not pod_id:
            raise ClientOperationError(
                f"Pod information is incomplete for job: {job_id}",
                hint="Re-run with `--debug` to inspect the raw GetJob response.",
            )

        def fetch() -> str:
            request = GetPodLogsRequest(max_lines=lines, pod_uid=pod_uid)
            self._debug(
                "GetPodLogs request",
                {"job_id": job_id, "pod_id": pod_id, "max_lines": lines, "pod_uid": pod_uid},
            )
            try:
                response = self.client.get_pod_logs(job_id, pod_id, request)
            except Exception as exc:  # noqa: BLE001
                raise ClientOperationError(
                    f"Failed to fetch logs for job: {job_id}",
                    hint="Check whether the job has started and whether the pod still exists.",
                ) from exc
            body = response.body.to_map()
            self._debug("GetPodLogs response", body)
            logs = body.get("Logs") or []
            return "\n".join(logs)

        if not follow:
            return fetch()

        def iterator() -> Iterator[str]:
            previous = ""
            while True:
                current = fetch()
                if current != previous:
                    previous = current
                    yield current
                time.sleep(3)

        return iterator()

    def map_remote_status(self, remote_status: str | None) -> JobStatus:
        return map_remote_status(remote_status)

    def _build_user_command(
        self,
        *,
        workdir: str,
        entrypoint: str,
        code_filename: str,
        code_download_url: str,
        extra_downloads: list[ExtraDownload] | None = None,
    ) -> str:
        command = self._normalize_entrypoint(entrypoint)
        quoted_workdir = shlex.quote(workdir)
        quoted_tar = shlex.quote(f"/tmp/{code_filename}")
        quoted_url = shlex.quote(code_download_url)
        steps = [
            "set -euo pipefail",
            f"mkdir -p {quoted_workdir}",
            f"curl -fsSL -o {quoted_tar} {quoted_url}",
            f"tar -xzf {quoted_tar} -C {quoted_workdir}",
        ]
        for index, extra in enumerate(extra_downloads or []):
            extra_tar = shlex.quote(f"/tmp/extra_{index}.tar.gz")
            extra_target = shlex.quote(extra.target)
            steps.extend(
                [
                    f"mkdir -p {extra_target}",
                    f"curl -fsSL -o {extra_tar} {shlex.quote(extra.download_url)}",
                    f"tar -xzf {extra_tar} -C {extra_target}",
                ]
            )
        steps.extend([f"cd {quoted_workdir}", command])
        inner_command = " && ".join(steps)
        return f"bash -lc {shlex.quote(inner_command)}"

    @staticmethod
    def _normalize_entrypoint(entrypoint: str) -> str:
        stripped = entrypoint.strip()
        if " " in stripped:
            return stripped
        if stripped.endswith(".sh"):
            return f"bash {shlex.quote(stripped)}"
        if stripped.endswith(".py"):
            return f"python {shlex.quote(stripped)}"
        return stripped

    def _debug(self, label: str, payload: dict[str, Any]) -> None:
        if self.debug:
            console.print(f"[dim]{label}[/dim]")
            console.print(payload)
