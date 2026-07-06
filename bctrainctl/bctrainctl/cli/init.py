"""`bctrainctl init` command."""

from __future__ import annotations

import typer

from bctrainctl.cli.common import get_debug
from bctrainctl.clients.dlc_client import DlcClient
from bctrainctl.clients.oss_client import OssClient
from bctrainctl.core.models import AppConfig, DefaultResourceProfile
from bctrainctl.storage.aliyun_cli_store import AliyunCliConfigStore
from bctrainctl.storage.config_store import ConfigStore
from bctrainctl.utils.console import console, print_mapping_table
from bctrainctl.utils.errors import ValidationFailedError


REGION_ALIASES = {
    "乌兰察布": "cn-wulanchabu",
    "wulanchabu": "cn-wulanchabu",
    "ulanchab": "cn-wulanchabu",
}

INIT_DEFAULTS = {
    "region": "cn-wulanchabu",
    "workspace_id": "241942",
    "oss_bucket": "oss-pai-wulanchabu-stark",
    "oss_prefix": "bctrainctl",
    "default_resource_id": "quota-4090",
    "default_image": "pytorch:2.7.0-gpu-py311-cu128-ubuntu24.04-accl-b244fc94-1764399419",
    "default_gpu": "1",
    "default_cpu": "16",
    "default_memory": "60",
    "acr_registry": "acr-pai-wulanchabu-registry.cn-wulanchabu.cr.aliyuncs.com",
    "acr_namespace": "stark",
}


def _normalize_region(value: str) -> str:
    normalized = value.strip()
    return REGION_ALIASES.get(normalized.lower(), REGION_ALIASES.get(normalized, normalized))


def _configure_with_tui(ctx: typer.Context, *, store: ConfigStore, skip_checks: bool) -> None:
    """Run the TUI config editor, then validate cloud access like the CLI flow."""

    from bctrainctl.tui.app import run_config_tui

    run_config_tui()

    config = store.load()
    if config is None:
        console.print("[yellow]No config saved from the TUI. Nothing changed.[/yellow]")
        return

    if not skip_checks:
        debug = get_debug(ctx)
        OssClient(config, debug=debug).check_bucket_access()
        DlcClient(config, debug=debug).test_connectivity()

    print_mapping_table(
        "Config Saved",
        {
            "Config File": store.path,
            "Region": config.region,
            "Workspace ID": config.workspace_id,
            "OSS Bucket": config.oss_bucket,
            "OSS Prefix": config.oss_prefix,
            "Default DLC Quota": config.default_resource_id,
            "Default Image": config.default_image,
            "Connectivity Checked": not skip_checks,
        },
    )


def register_init_command(app: typer.Typer) -> None:
    @app.command("init")
    def init_command(
        ctx: typer.Context,
        force: bool = typer.Option(False, "--force", help="Overwrite existing config."),
        skip_checks: bool = typer.Option(
            False,
            "--skip-checks",
            help="Skip DLC and OSS connectivity checks.",
        ),
    ) -> None:
        """Initialize local config and validate cloud access."""

        store = ConfigStore()
        if store.exists() and not force:
            overwrite = typer.confirm(
                f"Config already exists at {store.path}. Overwrite it?",
                default=False,
            )
            if not overwrite:
                console.print(
                    "[yellow]Keeping existing config. Nothing changed.[/yellow]"
                )
                raise typer.Exit(code=0)

        if typer.confirm("Use the TUI to configure?", default=True):
            _configure_with_tui(ctx, store=store, skip_checks=skip_checks)
            return

        aliyun_cli_store = AliyunCliConfigStore()
        imported_profile = aliyun_cli_store.load_current_profile() if aliyun_cli_store.exists() else None
        use_aliyun_cli = False
        if imported_profile is not None:
            use_aliyun_cli = typer.confirm(
                f"Use credentials from ~/.aliyun/config.json current profile `{imported_profile.name}`?",
                default=True,
            )

        access_key_id = None
        access_key_secret = None
        credential_backend = "file"
        aliyun_profile = None
        if use_aliyun_cli and imported_profile is not None:
            credential_backend = "aliyun_cli"
            aliyun_profile = imported_profile.name
        else:
            access_key_id = typer.prompt("Alibaba Cloud AccessKey ID")
            access_key_secret = typer.prompt("Alibaba Cloud AccessKey Secret", hide_input=True)

        region = _normalize_region(typer.prompt("Region", default=INIT_DEFAULTS["region"]))
        workspace_id = (
            typer.prompt("Workspace ID", default=INIT_DEFAULTS["workspace_id"]).strip() or None
        )
        oss_bucket = typer.prompt("Default OSS Bucket", default=INIT_DEFAULTS["oss_bucket"])
        oss_prefix = typer.prompt("Default OSS Prefix", default=INIT_DEFAULTS["oss_prefix"])
        default_resource_id = typer.prompt(
            "Default DLC Quota",
            default=INIT_DEFAULTS["default_resource_id"],
        ).strip() or None
        default_image = (
            typer.prompt("Default Image", default=INIT_DEFAULTS["default_image"]).strip() or None
        )
        default_gpu = typer.prompt("Default GPU count", default=INIT_DEFAULTS["default_gpu"]).strip()
        default_cpu = typer.prompt("Default CPU count", default=INIT_DEFAULTS["default_cpu"]).strip()
        default_memory = typer.prompt(
            "Default memory (GB)",
            default=INIT_DEFAULTS["default_memory"],
        ).strip()

        default_resource_profile = None
        if any([default_gpu, default_cpu, default_memory]):
            if not all([default_gpu, default_cpu, default_memory]):
                raise ValidationFailedError(
                    "Default resource profile is incomplete.",
                    hint="Either leave all resource prompts blank or provide GPU, CPU, and memory together.",
                )
            default_resource_profile = DefaultResourceProfile(
                gpu=int(default_gpu),
                cpu=int(default_cpu),
                memory_gb=int(default_memory),
            )

        wandb_api_key = (
            typer.prompt(
                "Weights & Biases API Key (optional, blank to skip)",
                default="",
                hide_input=True,
            ).strip()
            or None
        )
        swanlab_api_key = (
            typer.prompt(
                "SwanLab API Key (optional, blank to skip)",
                default="",
                hide_input=True,
            ).strip()
            or None
        )

        acr_registry = acr_namespace = acr_username = acr_password = None
        if typer.confirm(
            "Configure ACR docker push credentials now?",
            default=False,
        ):
            acr_registry = (
                typer.prompt("ACR Registry", default=INIT_DEFAULTS["acr_registry"]).strip() or None
            )
            acr_namespace = (
                typer.prompt("ACR Namespace", default=INIT_DEFAULTS["acr_namespace"]).strip()
                or None
            )
            acr_username = typer.prompt("ACR Username").strip() or None
            acr_password = (
                typer.prompt("ACR Password", hide_input=True).strip() or None
            )

        config = AppConfig(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region=region,
            workspace_id=workspace_id,
            oss_bucket=oss_bucket,
            oss_prefix=oss_prefix,
            default_resource_id=default_resource_id,
            default_image=default_image,
            default_resource_profile=default_resource_profile,
            credential_backend=credential_backend,
            aliyun_profile=aliyun_profile,
            wandb_api_key=wandb_api_key,
            swanlab_api_key=swanlab_api_key,
            acr_registry=acr_registry,
            acr_namespace=acr_namespace,
            acr_username=acr_username,
            acr_password=acr_password,
        )

        if not skip_checks:
            debug = get_debug(ctx)
            OssClient(config, debug=debug).check_bucket_access()
            DlcClient(config, debug=debug).test_connectivity()

        # Save only after connectivity checks pass so a failed init never leaves
        # a half-written config that blocks a clean retry.
        store.save(config)

        print_mapping_table(
            "Config Saved",
            {
                "Config File": store.path,
                "Credential Backend": config.credential_backend,
                "Aliyun Profile": config.aliyun_profile,
                "Region": config.region,
                "Workspace ID": config.workspace_id,
                "OSS Bucket": config.oss_bucket,
                "OSS Prefix": config.oss_prefix,
                "Default DLC Quota": config.default_resource_id,
                "Default Image": config.default_image,
                "Default Resource Profile": config.default_resource_profile,
                "Connectivity Checked": not skip_checks,
            },
        )
