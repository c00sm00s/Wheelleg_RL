"""SQLite-backed local job index."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from bctrainctl.core.enums import JobStatus
from bctrainctl.core.models import JobManifest, JobRecord, JobSubmission
from bctrainctl.utils.errors import NotFoundError
from bctrainctl.utils.paths import ensure_app_dirs
from bctrainctl.utils.time import utc_now_iso


class JobStore:
    """Persist local job metadata for fast lookup and debugging."""

    def __init__(self, db_path: Path | None = None) -> None:
        paths = ensure_app_dirs()
        self.db_path = db_path or paths.jobs_db
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    local_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT NOT NULL,
                    job_id TEXT NOT NULL UNIQUE,
                    job_name TEXT NOT NULL,
                    project TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_synced_at TEXT,
                    oss_code_uri TEXT NOT NULL,
                    package_path TEXT NOT NULL,
                    package_hash TEXT NOT NULL,
                    entrypoint TEXT NOT NULL,
                    image TEXT NOT NULL,
                    raw_spec_json TEXT NOT NULL,
                    request_snapshot_json TEXT NOT NULL,
                    remote_payload_json TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)"
            )

    def create_record(
        self,
        *,
        manifest: JobManifest,
        submission: JobSubmission,
        status: JobStatus,
        remote_payload: dict | None = None,
    ) -> JobRecord:
        now = submission.submitted_at
        record = JobRecord(
            submission_id=submission.submission_id,
            job_id=submission.dlc_job_id or "",
            job_name=manifest.metadata.name,
            project=manifest.metadata.project,
            status=status,
            created_at=now,
            updated_at=now,
            last_synced_at=now if remote_payload else None,
            oss_code_uri=submission.oss_code_uri,
            package_path=str(submission.package_path),
            package_hash=submission.package_hash,
            entrypoint=manifest.spec.runtime.entrypoint,
            image=manifest.spec.image or "",
            raw_spec_json=manifest.model_dump_json(by_alias=True),
            request_snapshot_json=json.dumps(
                submission.request_snapshot,
                ensure_ascii=False,
                sort_keys=True,
            ),
            remote_payload_json=(
                json.dumps(remote_payload, ensure_ascii=False, sort_keys=True)
                if remote_payload
                else None
            ),
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO jobs (
                    submission_id, job_id, job_name, project, status, created_at, updated_at,
                    last_synced_at, oss_code_uri, package_path, package_hash, entrypoint, image,
                    raw_spec_json, request_snapshot_json, remote_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.submission_id,
                    record.job_id,
                    record.job_name,
                    record.project,
                    record.status.value,
                    record.created_at,
                    record.updated_at,
                    record.last_synced_at,
                    record.oss_code_uri,
                    record.package_path,
                    record.package_hash,
                    record.entrypoint,
                    record.image,
                    record.raw_spec_json,
                    record.request_snapshot_json,
                    record.remote_payload_json,
                ),
            )
            record.local_id = int(cursor.lastrowid)
        return record

    def list_jobs(self) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_job_required(self, job_id: str) -> JobRecord:
        record = self.get_job(job_id)
        if record is None:
            raise NotFoundError(
                f"Local job not found: {job_id}",
                hint="Run `bctrainctl list` to inspect locally tracked jobs, or submit the job first.",
            )
        return record

    def update_job_status(
        self,
        *,
        job_id: str,
        status: JobStatus,
        remote_payload: dict | None = None,
    ) -> JobRecord:
        existing = self.get_job_required(job_id)
        now = utc_now_iso()
        payload_json = (
            json.dumps(remote_payload, ensure_ascii=False, sort_keys=True)
            if remote_payload
            else existing.remote_payload_json
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?, last_synced_at = ?, remote_payload_json = ?
                WHERE job_id = ?
                """,
                (
                    status.value,
                    now,
                    now,
                    payload_json,
                    job_id,
                ),
            )
        return self.get_job_required(job_id)

    def delete_job(self, job_id: str) -> JobRecord | None:
        record = self.get_job(job_id)
        if record is None:
            return None
        with self._connect() as connection:
            connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        return record

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            local_id=row["local_id"],
            submission_id=row["submission_id"],
            job_id=row["job_id"],
            job_name=row["job_name"],
            project=row["project"],
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_synced_at=row["last_synced_at"],
            oss_code_uri=row["oss_code_uri"],
            package_path=row["package_path"],
            package_hash=row["package_hash"],
            entrypoint=row["entrypoint"],
            image=row["image"],
            raw_spec_json=row["raw_spec_json"],
            request_snapshot_json=row["request_snapshot_json"],
            remote_payload_json=row["remote_payload_json"],
        )
