"""Ostrá fronta nad Supabase. Atomický zámek a requeue dělají SQL funkce (sql/001_*.sql)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .models import (
    ALL_STATUSES, STATUS_DONE, STATUS_FAILED, STATUS_PROCESSING, Job, JobResult,
    JobStateError,
)
from .store import JobStore, RequeueReport

# [DOPLNIT] názvy sloupců ověř podle skutečného schématu scan_jobs
COL_STATUS = "status"
COL_FINISHED_AT = "finished_at"


class SupabaseJobStore(JobStore):
    def __init__(self, client: Any, table: str = "scan_jobs"):
        self._db = client
        self._table = table

    def claim_next(self) -> Job | None:
        data = self._db.rpc("claim_next_scan_job").execute().data
        if isinstance(data, dict):
            data = [data]
        if not data or not data[0].get("id"):
            return None
        return Job.from_row(data[0])

    def requeue_stale(self, timeout_s: int, max_attempts: int) -> RequeueReport:
        data = self._db.rpc(
            "requeue_stale_scan_jobs",
            {"p_timeout_seconds": timeout_s, "p_max_attempts": max_attempts},
        ).execute().data
        if isinstance(data, list):
            data = data[0] if data else {}
        data = data or {}
        return RequeueReport(int(data.get("requeued", 0)), int(data.get("failed", 0)))

    def _update_own(self, job_id: str, values: dict) -> None:
        # Podmínka na stav: pokud úlohu mezitím vzal requeue, update nic nezmění.
        res = (
            self._db.table(self._table)
            .update(values)
            .eq("id", job_id)
            .eq(COL_STATUS, STATUS_PROCESSING)
            .execute()
        )
        if not res.data:
            raise JobStateError(f"Úloha {job_id} není ve stavu '{STATUS_PROCESSING}'")

    def mark_done(self, job_id: str, result: JobResult) -> None:
        self._update_own(job_id, {
            COL_STATUS: STATUS_DONE,
            "result_spz_path": result.spz_path,
            "result_ply_path": result.ply_path,
            "result_spz_bytes": result.spz_bytes,
            "result_ply_bytes": result.ply_bytes,
            "gaussian_count": result.gaussian_count,
            "processing_seconds": round(result.processing_seconds, 1),
            COL_FINISHED_AT: datetime.now(timezone.utc).isoformat(),
        })

    def mark_failed(self, job_id: str, reason: str, detail: str = "") -> None:
        self._update_own(job_id, {
            COL_STATUS: STATUS_FAILED,
            "failure_reason": reason,
            "failure_detail": detail[:1000],
            COL_FINISHED_AT: datetime.now(timezone.utc).isoformat(),
        })

    def status_counts(self) -> dict[str, int]:
        counts = {}
        for status in ALL_STATUSES:
            res = (
                self._db.table(self._table)
                .select("id", count="exact")
                .eq(COL_STATUS, status)
                .limit(1)
                .execute()
            )
            counts[status] = int(res.count or 0)
        return counts
