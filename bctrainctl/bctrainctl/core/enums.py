"""Shared enums."""

from __future__ import annotations

from enum import Enum


class JobStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"


class MountType(str, Enum):
    NAS = "nas"
    OSS = "oss"

