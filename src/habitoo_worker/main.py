"""Vstupní bod workeru: smyčka dotazující frontu. Spuštění: python -m habitoo_worker.main"""
from __future__ import annotations

import logging
import os
import signal
import tempfile
import time
from pathlib import Path

from .config import SYNTHETIC, Config
from .jobs.fake_store import FakeJobStore
from .log import setup_logging
from .pipeline.runner import Components, process_job, purge_old_videos
from .storage import LocalStorage

log = logging.getLogger("habitoo_worker")

REQUEUE_EVERY_S = 60
PURGE_EVERY_S = 3600


def build_components(cfg: Config) -> Components:
    if cfg.mode == SYNTHETIC:
        from .pipeline.synthetic import SyntheticPoseEstimator, SyntheticTrainer
        store = FakeJobStore()
        storage = LocalStorage(Path(tempfile.mkdtemp(prefix="habitoo-synth-")))
        return Components(store, storage, SyntheticPoseEstimator(), SyntheticTrainer())

    from supabase import create_client
    from .jobs.supabase_store import SupabaseJobStore
    from .pipeline.poses import ColmapPoseEstimator
    from .pipeline.train import GsplatTrainer
    from .storage import SupabaseStorage
    client = create_client(cfg.supabase_url, cfg.service_role_key)
    return Components(SupabaseJobStore(client, cfg.jobs_table), SupabaseStorage(client),
                      ColmapPoseEstimator(), GsplatTrainer())


def run(cfg: Config, c: Components, stop: "Stop", max_jobs: int | None = None) -> int:
    """Hlavní smyčka. max_jobs jen pro testy. Vrací počet zpracovaných úloh."""
    done = 0
    last_requeue = last_purge = 0.0
    while not stop.flag:
        now = time.monotonic()
        if now - last_requeue >= REQUEUE_EVERY_S or last_requeue == 0.0:
            r = c.store.requeue_stale(cfg.job_timeout_s, cfg.max_attempts)
            if r.requeued or r.failed:
                log.warning("zaseknuté úlohy: vráceno do fronty %d, označeno jako selhané %d",
                            r.requeued, r.failed)
            last_requeue = now
        if now - last_purge >= PURGE_EVERY_S or last_purge == 0.0:
            purge_old_videos(cfg, c)
            last_purge = now

        job = c.store.claim_next()
        if job is None:
            if max_jobs is not None:
                break
            stop.sleep(cfg.poll_interval_s)
            continue
        process_job(job, cfg, c)
        done += 1
        if max_jobs is not None and done >= max_jobs:
            break
    return done


class Stop:
    def __init__(self) -> None:
        self.flag = False

    def set(self, *_args) -> None:
        log.info("přijat signál ukončení – dokončím aktuální úlohu a skončím")
        self.flag = True

    def sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self.flag and time.monotonic() < end:
            time.sleep(min(0.5, seconds))


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()  # lokální .env (není v repu ani v Docker obrazu)
    cfg = Config.from_env(os.environ)
    setup_logging([cfg.service_role_key])
    log.info("worker startuje, režim=%s, interval=%d s, limit úlohy=%d s",
             cfg.mode, cfg.poll_interval_s, cfg.job_timeout_s)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    stop = Stop()
    signal.signal(signal.SIGTERM, stop.set)
    signal.signal(signal.SIGINT, stop.set)
    run(cfg, build_components(cfg), stop)
    log.info("worker ukončen")


if __name__ == "__main__":
    main()
