"""Workspace resource discovery for resolving bound DLC quota names."""

from __future__ import annotations

from typing import Any

from alibabacloud_aiworkspace20210204.client import Client as WorkspaceSdkClient
from alibabacloud_aiworkspace20210204.models import ListResourcesRequest
from alibabacloud_tea_openapi.models import Config as OpenApiConfig

from bctrainctl.core.models import AppConfig
from bctrainctl.utils.console import console
from bctrainctl.utils.credentials import resolve_credentials
from bctrainctl.utils.errors import ClientOperationError


class WorkspaceClient:
    """Resolve workspace-bound DLC quota names to actual quota IDs."""

    def __init__(self, config: AppConfig, *, debug: bool = False) -> None:
        self.config = config
        self.debug = debug
        credentials = resolve_credentials(config)
        sdk_config = OpenApiConfig(
            access_key_id=credentials.access_key_id,
            access_key_secret=credentials.access_key_secret,
            region_id=config.region,
        )
        self.client = WorkspaceSdkClient(sdk_config)

    def list_workspace_dlc_resources(self) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        for resource_type in ("DLC", "Lingjun"):
            request = ListResourcesRequest(
                workspace_id=self.config.workspace_id,
                option="ListResourceByWorkspace",
                page_size=100,
                page_number=1,
                verbose=True,
                verbose_fields="Quota,IsDefault",
                resource_types=resource_type,
            )
            self._debug("ListResources request", request.to_map())
            try:
                response = self.client.list_resources(request)
            except Exception as exc:  # noqa: BLE001
                raise ClientOperationError(
                    f"Failed to query workspace {resource_type} resources: {exc}",
                    hint="Check workspace ID, AIWorkspace permissions, and region settings.",
                ) from exc
            body = response.body.to_map()
            self._debug("ListResources response", body)
            resources.extend(body.get("Resources", []))
        return resources

    def resolve_dlc_quota_id(self, quota_selector: str) -> tuple[str, str | None]:
        """Resolve a quota selector to a workspace-bound quota ID."""

        resources = self.list_workspace_dlc_resources()
        for resource in resources:
            spec = resource.get("Spec") or {}
            resource_id = spec.get("resourceId")
            resource_name = spec.get("resourceName")
            if quota_selector in {resource_id, resource_name}:
                return resource_id, resource_name

        available = sorted(
            {
                spec.get("resourceName")
                for resource in resources
                for spec in [resource.get("Spec") or {}]
                if spec.get("resourceName")
            }
        )
        raise ClientOperationError(
            f"Workspace-bound DLC quota not found: {quota_selector}",
            hint=(
                "Use a quota name or quota ID that is already bound to the current workspace. "
                f"Currently visible DLC quotas: {', '.join(available) if available else 'none'}."
            ),
        )

    def _debug(self, label: str, payload: dict[str, Any]) -> None:
        if self.debug:
            console.print(f"[dim]{label}[/dim]")
            console.print(payload)
