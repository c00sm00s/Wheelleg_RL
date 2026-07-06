"""Read Alibaba Cloud CLI credentials from ~/.aliyun/config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bctrainctl.utils.errors import ValidationFailedError


@dataclass(frozen=True, slots=True)
class AliyunCliProfile:
    name: str
    mode: str
    access_key_id: str
    access_key_secret: str


class AliyunCliConfigStore:
    """Load AK credentials from the local Alibaba Cloud CLI config file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (Path.home() / ".aliyun" / "config.json")

    def exists(self) -> bool:
        return self.path.exists()

    def load_current_profile(self) -> AliyunCliProfile:
        payload = self._load()
        current = payload.get("current")
        if not current:
            raise ValidationFailedError(
                f"`current` profile is missing in {self.path}",
                hint="Run `aliyun configure` or edit the config file to set a current profile.",
            )
        return self.get_profile(current)

    def get_profile(self, profile_name: str) -> AliyunCliProfile:
        payload = self._load()
        profiles = payload.get("profiles") or []
        for item in profiles:
            if item.get("name") != profile_name:
                continue
            mode = item.get("mode")
            if mode != "AK":
                raise ValidationFailedError(
                    f"Aliyun CLI profile `{profile_name}` uses unsupported mode `{mode}`.",
                    hint="Current bctrainctl only supports AK mode from ~/.aliyun/config.json.",
                )
            access_key_id = item.get("access_key_id")
            access_key_secret = item.get("access_key_secret")
            if not access_key_id or not access_key_secret:
                raise ValidationFailedError(
                    f"Aliyun CLI profile `{profile_name}` is missing AccessKey fields.",
                    hint="Run `aliyun configure` again or switch to a valid AK profile.",
                )
            return AliyunCliProfile(
                name=profile_name,
                mode=mode,
                access_key_id=access_key_id,
                access_key_secret=access_key_secret,
            )
        raise ValidationFailedError(
            f"Aliyun CLI profile `{profile_name}` not found in {self.path}",
            hint="Check the profile name or inspect ~/.aliyun/config.json.",
        )

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValidationFailedError(
                f"Aliyun CLI config not found: {self.path}",
                hint="Run `aliyun configure` first, or use `bctrainctl init` with manual AccessKey input.",
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValidationFailedError(
                f"Aliyun CLI config is invalid JSON: {self.path}",
                hint="Fix ~/.aliyun/config.json or recreate it with `aliyun configure`.",
            ) from exc

