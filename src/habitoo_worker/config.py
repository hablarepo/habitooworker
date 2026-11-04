"""Konfigurace workeru z proměnných prostředí (.env se načítá jen v main.py).

Hodnoty označené [DOPLNIT] nemají výchozí hodnotu – v režimu "real" je nutné je
nastavit, jinak worker odmítne nastartovat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

REAL = "real"
SYNTHETIC = "synthetic"


class ConfigError(ValueError):
    pass


def _int(env: Mapping[str, str], key: str, default: int | None) -> int | None:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{key} musí být celé číslo, je {raw!r}")


@dataclass(frozen=True)
class Config:
    mode: str
    supabase_url: str
    # repr=False: klíč se nesmí dostat do logů ani do tracebacků
    service_role_key: str = field(repr=False)
    jobs_table: str
    video_bucket: str
    model_bucket: str
    job_timeout_s: int
    video_retention_days: int  # 0 = smazat video hned po úspěšném zpracování
    poll_interval_s: int
    max_attempts: int
    work_dir: str
    target_frames: int
    min_frames: int
    max_image_side: int
    max_gaussians: int
    train_steps: int

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Config":
        mode = (env.get("WORKER_MODE") or REAL).strip().lower()
        if mode not in (REAL, SYNTHETIC):
            raise ConfigError(f"WORKER_MODE musí být '{REAL}' nebo '{SYNTHETIC}'")

        missing = []

        def required(key: str) -> str:
            val = (env.get(key) or "").strip()
            if not val:
                missing.append(key)
            return val

        url = required("SUPABASE_URL")
        key = required("SUPABASE_SERVICE_ROLE_KEY")
        video_bucket = required("VIDEO_BUCKET")
        model_bucket = required("MODEL_BUCKET")
        timeout = _int(env, "JOB_TIMEOUT_S", None)
        retention = _int(env, "VIDEO_RETENTION_DAYS", None)
        if timeout is None:
            missing.append("JOB_TIMEOUT_S")
        if retention is None:
            missing.append("VIDEO_RETENTION_DAYS")

        if missing and mode == REAL:
            raise ConfigError(
                "Chybí povinné nastavení [DOPLNIT]: " + ", ".join(missing)
            )

        cfg = cls(
            mode=mode,
            supabase_url=url,
            service_role_key=key,
            jobs_table=(env.get("SCAN_JOBS_TABLE") or "scan_jobs").strip(),
            video_bucket=video_bucket,
            model_bucket=model_bucket,
            job_timeout_s=timeout if timeout is not None else 3600,
            video_retention_days=retention if retention is not None else 0,
            poll_interval_s=_int(env, "POLL_INTERVAL_S", 10),
            max_attempts=_int(env, "MAX_ATTEMPTS", 2),
            work_dir=(env.get("WORK_DIR") or "/work").strip(),
            target_frames=_int(env, "TARGET_FRAMES", 150),
            min_frames=_int(env, "MIN_FRAMES", 30),
            max_image_side=_int(env, "MAX_IMAGE_SIDE", 960),
            max_gaussians=_int(env, "MAX_GAUSSIANS", 600_000),
            train_steps=_int(env, "TRAIN_STEPS", 7000),
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if self.job_timeout_s <= 0:
            raise ConfigError("JOB_TIMEOUT_S musí být > 0")
        if self.video_retention_days < 0:
            raise ConfigError("VIDEO_RETENTION_DAYS musí být >= 0")
        if self.max_attempts < 1:
            raise ConfigError("MAX_ATTEMPTS musí být >= 1")
        if not 1 <= self.min_frames <= self.target_frames:
            raise ConfigError("Musí platit 1 <= MIN_FRAMES <= TARGET_FRAMES")
        if self.train_steps < 1 or self.max_gaussians < 1:
            raise ConfigError("TRAIN_STEPS a MAX_GAUSSIANS musí být >= 1")
