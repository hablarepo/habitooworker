"""Falešná fronta v paměti – pro testy a režim `synthetic`. Chová se stejně jako SQL funkce."""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .models import (
    ALL_STATUSES, FAIL_TIMEOUT, STATUS_DONE, STATUS_FAILED, STATUS_PENDING,
    STATUS_PROCESSING, Job, JobResult, JobStateError,
)
from .store import JobStore, RequeueReport


@dataclass
class _Row:
    id: str
    room_id: str
    video_path: str
    seq: int
    status: str = STATUS_PENDING
    attempts: int = 0
    claimed_at: float | None = None
    failure_reason: str | None = None
    failure_detail: str | None = None
    result: JobResult | None = None


class FakeJobStore(JobStore):
    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._rows: dict[str, _Row] = {}
        self._seq = itertools.count()

    # --- pomocné metody pro testy ---
    def add_job(self, room_id: str = "room-1", video_path: str = "v.mp4",
                job_id: str | None = None) -> str:
        with self._lock:
            n = next(self._seq)
            jid = job_id or f"job-{n}"
            self._rows[jid] = _Row(jid, room_id, video_path, seq=n)
            return jid

    def get(self, job_id: str) -> _Row:
        return self._rows[job_id]

    # --- JobStore ---
    def claim_next(self) -> Job | None:
        with self._lock:
            pending = [r for r in self._rows.values() if r.status == STATUS_PENDING]
            if not pending:
                return None
            row = min(pending, key=lambda r: r.seq)
            row.status = STATUS_PROCESSING
            row.attempts += 1
            row.claimed_at = self._clock()
            return Job(row.id, row.room_id, row.video_path, row.attempts)

    def requeue_stale(self, timeout_s: int, max_attempts: int) -> RequeueReport:
        requeued = failed = 0
        with self._lock:
            now = self._clock()
            for row in self._rows.values():
                if row.status != STATUS_PROCESSING or row.claimed_at is None:
                    continue
                if now - row.claimed_at <= timeout_s:
                    continue
                if row.attempts >= max_attempts:
                    row.status = STATUS_FAILED
                    row.failure_reason = FAIL_TIMEOUT
                    failed += 1
                else:
                    row.status = STATUS_PENDING
                    row.claimed_at = None
                    requeued += 1
        return RequeueReport(requeued, failed)

    def _own(self, job_id: str) -> _Row:
        row = self._rows.get(job_id)
        if row is None or row.status != STATUS_PROCESSING:
            raise JobStateError(f"Úloha {job_id} není ve stavu '{STATUS_PROCESSING}'")
        return row

    def mark_done(self, job_id: str, result: JobResult) -> None:
        with self._lock:
            row = self._own(job_id)
            row.status = STATUS_DONE
            row.result = result

    def mark_failed(self, job_id: str, reason: str, detail: str = "") -> None:
        with self._lock:
            row = self._own(job_id)
            row.status = STATUS_FAILED
            row.failure_reason = reason
            row.failure_detail = detail

    def status_counts(self) -> dict[str, int]:
        with self._lock:
            counts = {s: 0 for s in ALL_STATUSES}
            for r in self._rows.values():
                counts[r.status] += 1
            return counts
