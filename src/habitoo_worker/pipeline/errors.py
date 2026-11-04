from __future__ import annotations

from ..jobs.models import FAIL_OTHER


class PipelineError(Exception):
    """Chyba zpracování s důvodem, který se zapíše do scan_jobs.failure_reason."""

    def __init__(self, reason: str = FAIL_OTHER, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class GpuOutOfMemory(Exception):
    """Došla paměť GPU při tréninku (runner zkusí jednou znovu s nižším nastavením)."""
