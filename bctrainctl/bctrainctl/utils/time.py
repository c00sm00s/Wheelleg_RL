"""Time helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_utc_timestamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")

