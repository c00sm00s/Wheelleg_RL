"""`bctrainctl acr` commands for pushing Docker images to ACR."""

from __future__ import annotations

import typer

from bctrainctl.cli.common import get_debug
from bctrainctl.clients.acr_client import (
    DEFAULT_NAMESPACE,
    DEFAULT_REGISTRY,
    AcrClient,
    AcrPushRequest,
    build_target_ref,
    split_image_ref,
)
from bctrainctl.storage.config_store import ConfigStore
from bctrainctl.utils.console import console, print_mapping_table
from bctrainctl.utils.errors import ValidationFailedError


def _select_local_image(client: AcrClient) -> str:
    images = client.list_images()
    if not images:
        raise ValidationFailedError(
            "No taggable local Docker images were found.",
            hint="Build an image first (`docker build -t name:tag .`), then retry. "
            "BuildKit cache shown by `docker buildx du` cannot be tagged directly.",
        )
    console.print("\n[bold]Local Docker images:[/bold]")
    for index, ref in enumerate(images, start=1):
        console.print(f"  {index}) {ref}")
    while True:
        choice = typer.prompt("Select image number")
        if choice.isdigit() and 1 <= int(choice) <= len(images):
            return images[int(choice) - 1]
        console.print(f"[yellow]Enter a number from 1 to {len(images)}.[/yellow]")


def _push(client: AcrClient, *, source: str, target: str, registry: str, username: str, password: str) -> None:
    """Run the actual docker login / tag / push sequence."""

    console.print(f"[dim]Logging in to {registry} ...[/dim]")
    client.login(registry=registry, username=username, password=password)
    console.print("[dim]Tagging image ...[/dim]")
    client.tag(source=source, target=target)
    console.print("[dim]Pushing image ...[/dim]")
    client.push(target=target)
    console.print(f"[green]Done:[/green] {target}")


def run_acr_push_request(request: AcrPushRequest, *, debug: bool = False) -> None:
    """Execute an ACR push assembled by the TUI, outside the textual app."""

    client = AcrClient(use_sudo=request.use_sudo, debug=debug)
    client.require_docker()
    target = build_target_ref(
        registry=request.registry,
        namespace=request.namespace,
        name=request.name,
        tag=request.tag,
    )
    print_mapping_table(
        "ACR Push Plan",
        {"Local Image": request.source, "Target": target, "Registry": request.registry},
    )
    _push(
        client,
        source=request.source,
        target=target,
        registry=request.registry,
        username=request.username,
        password=request.password,
    )


def register_acr_commands(app: typer.Typer) -> None:
    @app.command("push")
    def push_command(
        ctx: typer.Context,
        image: str | None = typer.Option(
            None, "--image", help="Local image ref to push. Prompts if omitted."
        ),
        registry: str | None = typer.Option(None, "--registry", help="Override ACR registry."),
        namespace: str | None = typer.Option(None, "--namespace", help="Override ACR namespace."),
        username: str | None = typer.Option(None, "--username", help="Override ACR username."),
        password: str | None = typer.Option(None, "--password", help="Override ACR password."),
        remote_name: str | None = typer.Option(None, "--name", help="Remote image name."),
        remote_tag: str | None = typer.Option(None, "--tag", help="Remote image tag."),
        use_sudo: bool = typer.Option(False, "--sudo/--no-sudo", help="Run docker via sudo."),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
        tui: bool = typer.Option(False, "--tui", help="Fill the push in via the TUI."),
    ) -> None:
        """Tag a local Docker image and push it to Alibaba Cloud ACR."""

        debug = get_debug(ctx)

        if tui:
            from bctrainctl.tui.app import run_acr_tui

            request = run_acr_tui()
            if request is None:
                console.print("[yellow]Cancelled. Nothing pushed.[/yellow]")
                raise typer.Exit(code=0)
            run_acr_push_request(request, debug=debug)
            return

        config = ConfigStore().load()
        client = AcrClient(use_sudo=use_sudo, debug=debug)
        client.require_docker()

        source = image or _select_local_image(client)
        default_name, default_tag = split_image_ref(source)

        registry = registry or (config.acr_registry if config else None) or DEFAULT_REGISTRY
        namespace = namespace or (config.acr_namespace if config else None) or DEFAULT_NAMESPACE
        username = username or (config.acr_username if config else None)
        if not username:
            username = typer.prompt("ACR username")
        if not password:
            password = (
                config.acr_password.get_secret_value()
                if config and config.acr_password
                else typer.prompt("ACR password", hide_input=True)
            )

        name = remote_name or default_name
        tag = remote_tag or default_tag
        target = build_target_ref(registry=registry, namespace=namespace, name=name, tag=tag)

        print_mapping_table(
            "ACR Push Plan",
            {"Local Image": source, "Target": target, "Registry": registry},
        )
        if not yes and not typer.confirm("Continue?", default=True):
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(code=0)

        _push(client, source=source, target=target, registry=registry, username=username, password=password)
