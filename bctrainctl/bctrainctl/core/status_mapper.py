"""Remote DLC status mapping."""

from __future__ import annotations

from bctrainctl.core.enums import JobStatus


REMOTE_STATUS_MAP: dict[str, JobStatus] = {
    "Creating": JobStatus.PENDING,
    "Queued": JobStatus.PENDING,
    "Queuing": JobStatus.PENDING,
    "Bidding": JobStatus.PENDING,
    "EnvPreparing": JobStatus.PENDING,
    "SanityChecking": JobStatus.PENDING,
    "Pending": JobStatus.PENDING,
    "Submitted": JobStatus.SUBMITTED,
    "Running": JobStatus.RUNNING,
    "Restarting": JobStatus.RUNNING,
    "Stopping": JobStatus.STOPPED,
    "Stopped": JobStatus.STOPPED,
    "SucceededReserving": JobStatus.SUCCEEDED,
    "Succeeded": JobStatus.SUCCEEDED,
    "FailedReserving": JobStatus.FAILED,
    "Failed": JobStatus.FAILED,
    "Unknown": JobStatus.UNKNOWN,
}


def map_remote_status(remote_status: str | None) -> JobStatus:
    if not remote_status:
        return JobStatus.UNKNOWN
    return REMOTE_STATUS_MAP.get(remote_status, JobStatus.UNKNOWN)

