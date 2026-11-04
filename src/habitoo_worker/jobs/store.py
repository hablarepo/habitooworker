"""Rozhraní fronty úloh. Worker zná jen toto – Supabase i falešná fronta ho implementují."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import Job, JobResult


@dataclass(frozen=True)
class RequeueReport:
    requeued: int  # vráceno na "čeká"
    failed: int    # označeno "nepovedlo se" (vyčerpány pokusy)


class JobStore(ABC):
    @abstractmethod
    def claim_next(self) -> Job | None:
        """Atomicky vezme nejstarší úlohu "čeká", přepne na "zpracovává se", zvýší attempts."""

    @abstractmethod
    def requeue_stale(self, timeout_s: int, max_attempts: int) -> RequeueReport:
        """Úlohy "zpracovává se" starší než timeout: attempts < max -> "čeká", jinak "nepovedlo se"."""

    @abstractmethod
    def mark_done(self, job_id: str, result: JobResult) -> None:
        """"zpracovává se" -> "hotovo" + výsledky. Vyhodí JobStateError, pokud úloha už není naše."""

    @abstractmethod
    def mark_failed(self, job_id: str, reason: str, detail: str = "") -> None:
        """"zpracovává se" -> "nepovedlo se" + důvod. Vyhodí JobStateError, pokud úloha už není naše."""

    @abstractmethod
    def status_counts(self) -> dict[str, int]:
        """Počet úloh podle stavu (pro scripts/queue_status.py)."""
