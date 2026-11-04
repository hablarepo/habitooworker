"""Stavy úloh, důvody selhání a datový model úlohy.

Hodnoty STATUS_* musí přesně odpovídat řetězcům v sql/001_scan_jobs_worker.sql.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

STATUS_PENDING = "čeká"
STATUS_PROCESSING = "zpracovává se"
STATUS_DONE = "hotovo"
STATUS_FAILED = "nepovedlo se"

ALL_STATUSES = (STATUS_PENDING, STATUS_PROCESSING, STATUS_DONE, STATUS_FAILED)

# Důvody selhání (ukládají se do sloupce failure_reason)
FAIL_TOO_FEW_FRAMES = "málo záběrů"
FAIL_BLURRY = "rozmazané video"
FAIL_POSES = "nepodařilo se spočítat polohy"
FAIL_TIMEOUT = "překročen časový limit"
FAIL_OOM = "nedostatek paměti GPU"
FAIL_DOWNLOAD = "video se nepodařilo stáhnout"
FAIL_OTHER = "jiná chyba"


class JobStateError(RuntimeError):
    """Úloha už není ve stavu, ve kterém ji worker očekával (např. ji mezitím vrátil requeue)."""


@dataclass(frozen=True)
class Job:
    id: str
    room_id: str
    video_path: str  # cesta k videu v bucketu VIDEO_BUCKET
    attempts: int = 0

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Job":
        # [DOPLNIT] názvy sloupců ověř podle skutečného schématu scan_jobs
        return cls(
            id=str(row["id"]),
            room_id=str(row["room_id"]),
            video_path=str(row["video_path"]),
            attempts=int(row.get("attempts") or 0),
        )


@dataclass(frozen=True)
class JobResult:
    spz_path: str
    ply_path: str
    spz_bytes: int
    ply_bytes: int
    gaussian_count: int
    processing_seconds: float
