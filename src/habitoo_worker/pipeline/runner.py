"""Zpracování jedné úlohy: stažení -> snímky -> polohy -> trénink -> export -> upload -> úklid."""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..jobs.models import (
    FAIL_DOWNLOAD, FAIL_OOM, FAIL_OTHER, FAIL_TIMEOUT, Job, JobResult, JobStateError,
)
from ..jobs.store import JobStore
from ..storage import Storage
from .errors import GpuOutOfMemory, PipelineError
from .export import write_ply, write_spz
from .frames import extract_frames
from .poses import PoseEstimator
from .train import Trainer, TrainSettings

log = logging.getLogger(__name__)


@dataclass
class Components:
    store: JobStore
    storage: Storage
    poses: PoseEstimator
    trainer: Trainer


def model_paths(job: Job) -> tuple[str, str]:
    """Cesty v bucketu modelů: <room_id>/<job_id>.spz / .ply"""
    return f"{job.room_id}/{job.id}.spz", f"{job.room_id}/{job.id}.ply"


def process_job(job: Job, cfg: Config, c: Components) -> bool:
    """Vrací True při úspěchu. Nikdy nevyhazuje – chyby zapíše do fronty."""
    t0 = time.monotonic()
    deadline = t0 + cfg.job_timeout_s
    workdir = Path(cfg.work_dir) / job.id
    log.info("úloha %s (místnost %s, pokus %d): start", job.id, job.room_id, job.attempts)

    def check_deadline() -> None:
        if time.monotonic() > deadline:
            raise PipelineError(FAIL_TIMEOUT, f"limit {cfg.job_timeout_s} s")

    try:
        workdir.mkdir(parents=True, exist_ok=True)
        video = workdir / "video.mp4"
        try:
            c.storage.download(cfg.video_bucket, job.video_path, video)
        except Exception as e:
            raise PipelineError(FAIL_DOWNLOAD, type(e).__name__) from e
        check_deadline()

        fs = extract_frames(video, workdir / "frames", cfg.target_frames, cfg.min_frames,
                            cfg.max_image_side)
        log.info("snímků: %d (z %d kandidátů, %d rozmazaných) za %.0f s",
                 len(fs.paths), fs.n_candidates, fs.n_blurry, time.monotonic() - t0)
        check_deadline()

        recon = c.poses.estimate(fs.paths, workdir)
        log.info("polohy: %d kamer, %d 3D bodů za %.0f s", len(recon.image_paths),
                 len(recon.points_xyz), time.monotonic() - t0)
        check_deadline()

        settings = TrainSettings(max_gaussians=cfg.max_gaussians, steps=cfg.train_steps)
        try:
            res = c.trainer.train(recon, settings, check_deadline)
        except GpuOutOfMemory:
            settings = settings.reduced()
            log.warning("došla paměť GPU, opakuji s %d gaussiány a měřítkem obrazu %.2f",
                        settings.max_gaussians, settings.image_scale)
            try:
                res = c.trainer.train(recon, settings, check_deadline)
            except GpuOutOfMemory as e:
                raise PipelineError(FAIL_OOM, "ani s sníženým nastavením") from e
        vram = f"{res.peak_vram_mb:.0f} MB" if res.peak_vram_mb is not None else "n/a"
        log.info("trénink: %d gaussiánů, %.0f s, špička VRAM %s",
                 res.splat.count, res.seconds, vram)
        check_deadline()

        spz_local, ply_local = workdir / "model.spz", workdir / "model.ply"
        spz_bytes = write_spz(res.splat, spz_local)
        ply_bytes = write_ply(res.splat, ply_local)
        spz_path, ply_path = model_paths(job)
        c.storage.upload(cfg.model_bucket, spz_path, spz_local, "application/octet-stream")
        c.storage.upload(cfg.model_bucket, ply_path, ply_local, "application/octet-stream")

        result = JobResult(spz_path, ply_path, spz_bytes, ply_bytes, res.splat.count,
                           time.monotonic() - t0)
        c.store.mark_done(job.id, result)
        log.info("úloha %s hotova za %.0f s (SPZ %.1f MB, PLY %.1f MB)", job.id,
                 result.processing_seconds, spz_bytes / 1e6, ply_bytes / 1e6)

        if cfg.video_retention_days == 0:
            _delete_video(c, cfg, job)
        return True
    except PipelineError as e:
        _fail(c, job, e.reason, e.detail)
    except JobStateError as e:
        log.warning("úloha %s už není naše (%s) – zahazuji výsledek", job.id, e)
    except Exception as e:  # noqa: BLE001 – worker nesmí spadnout kvůli jedné úloze
        log.exception("úloha %s: neočekávaná chyba", job.id)
        _fail(c, job, FAIL_OTHER, f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)  # krok 8: smazat dočasné soubory
    return False


def _fail(c: Components, job: Job, reason: str, detail: str) -> None:
    log.error("úloha %s selhala: %s %s", job.id, reason, detail)
    try:
        c.store.mark_failed(job.id, reason, detail)
    except JobStateError:
        log.warning("úloha %s už není naše, selhání nezapsáno", job.id)


def _delete_video(c: Components, cfg: Config, job: Job) -> None:
    try:
        c.storage.delete(cfg.video_bucket, job.video_path)
        c.store.mark_video_deleted(job.id)
    except Exception:  # noqa: BLE001 – smazání zkusí znovu úklid podle retence
        log.exception("smazání videa úlohy %s se nepovedlo", job.id)


def purge_old_videos(cfg: Config, c: Components) -> int:
    """Smaže videa hotových/selhaných úloh starší než VIDEO_RETENTION_DAYS."""
    n = 0
    for job_id, video_path in c.store.videos_to_purge(cfg.video_retention_days):
        try:
            c.storage.delete(cfg.video_bucket, video_path)
            c.store.mark_video_deleted(job_id)
            n += 1
        except Exception:  # noqa: BLE001
            log.exception("smazání videa úlohy %s se nepovedlo", job_id)
    if n:
        log.info("smazáno %d starých videí", n)
    return n
