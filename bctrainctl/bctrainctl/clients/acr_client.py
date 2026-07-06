"""Docker / Alibaba Cloud Container Registry (ACR) helpers.

A small Python port of the `aliyun_acr_tools/acr_shell/acr_push.sh` workflow:
list local images, log in, tag, and push to ACR.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from bctrainctl.utils.console import console
from bctrainctl.utils.errors import ClientOperationError

DEFAULT_REGISTRY = "acr-pai-wulanchabu-registry.cn-wulanchabu.cr.aliyuncs.com"
DEFAULT_NAMESPACE = "stark"


@dataclass(frozen=True, slots=True)
class AcrPushRequest:
    """A fully-resolved ACR push request produced by the TUI."""

    source: str
    registry: str
    namespace: str
    username: str
    password: str
    name: str
    tag: str
    use_sudo: bool = False


class AcrClient:
    """Thin wrapper around the local `docker` CLI for ACR pushes."""

    def __init__(self, *, use_sudo: bool = False, debug: bool = False) -> None:
        self.docker = ["sudo", "docker"] if use_sudo else ["docker"]
        self.debug = debug

    def require_docker(self) -> None:
        if shutil.which("docker") is None:
            raise ClientOperationError(
                "Required command not found: docker",
                hint="Install Docker and make sure the `docker` CLI is on your PATH.",
            )
        if self.docker[0] == "sudo" and shutil.which("sudo") is None:
            raise ClientOperationError(
                "Required command not found: sudo",
                hint="Re-run with `--no-sudo` or install sudo.",
            )

    def list_images(self) -> list[str]:
        """Return taggable local image references (`repository:tag`)."""

        result = self._run(
            [*self.docker, "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
            capture=True,
        )
        images: list[str] = []
        for line in result.stdout.splitlines():
            ref = line.strip()
            if not ref or ref == "<none>:<none>":
                continue
            images.append(ref)
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(images))

    def login(self, *, registry: str, username: str, password: str) -> None:
        self._debug(f"docker login {registry} as {username}")
        try:
            subprocess.run(
                [*self.docker, "login", "--username", username, "--password-stdin", registry],
                input=password.encode("utf-8"),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ClientOperationError(
                f"Failed to log in to ACR registry: {registry}",
                hint="Check the registry address, username, and password.",
            ) from exc

    def tag(self, *, source: str, target: str) -> None:
        self._debug(f"docker tag {source} {target}")
        self._run([*self.docker, "tag", source, target])

    def push(self, *, target: str) -> None:
        self._debug(f"docker push {target}")
        self._run([*self.docker, "push", target])

    def _run(self, args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                args,
                check=True,
                text=True,
                capture_output=capture,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()
            raise ClientOperationError(
                f"Docker command failed: {' '.join(args)}",
                hint=detail or "Check Docker daemon status and the image reference.",
            ) from exc

    def _debug(self, message: str) -> None:
        if self.debug:
            console.print(f"[dim]{message}[/dim]")


def build_target_ref(*, registry: str, namespace: str, name: str, tag: str) -> str:
    return f"{registry.rstrip('/')}/{namespace.strip('/')}/{name}:{tag}"


def split_image_ref(image_ref: str) -> tuple[str, str]:
    """Split `repo/name:tag` into a default remote (name, tag)."""

    if ":" in image_ref and "/" not in image_ref.rsplit(":", 1)[1]:
        repo, tag = image_ref.rsplit(":", 1)
    else:
        repo, tag = image_ref, "latest"
    name = repo.rsplit("/", 1)[-1]
    return name, tag
