import dataclasses
import gzip

import pytest

from habitoo_worker.jobs.models import (
    FAIL_BLURRY, FAIL_DOWNLOAD, FAIL_OOM, FAIL_OTHER, FAIL_POSES, FAIL_TIMEOUT,
    FAIL_TOO_FEW_FRAMES, STATUS_DONE, STATUS_FAILED, STATUS_PENDING,
)
from habitoo_worker.main import Stop, run
from habitoo_worker.pipeline.errors import GpuOutOfMemory, PipelineError
from habitoo_worker.pipeline.runner import process_job, purge_old_videos
from habitoo_worker.pipeline.synthetic import SyntheticTrainer


def test_success_end_to_end(env, cfg):
    comps, add, tmp = env
    job = add(n_frames=90)
    assert process_job(job, cfg, comps) is True
    row = comps.store.get("job-1")
    assert row.status == STATUS_DONE
    r = row.result
    assert r.spz_path == "room-1/job-1.spz" and r.ply_path == "room-1/job-1.ply"
    assert 0 < r.gaussian_count <= cfg.max_gaussians
    assert r.spz_bytes > 0 and r.ply_bytes > r.spz_bytes
    assert comps.storage.exists("models", r.spz_path) and comps.storage.exists("models", r.ply_path)
    gzip.decompress(comps.storage._p("models", r.spz_path).read_bytes())  # platný gzip
    # krok 8: dočasné soubory smazány
    assert not (tmp / "work" / "job-1").exists()
    # retence 0: původní video smazáno hned
    assert not comps.storage.exists("videos", "room-1/video.mp4")
    assert row.video_deleted


def test_gaussian_limit_is_respected(env, cfg):
    comps, add, _ = env
    job = add()
    process_job(job, dataclasses.replace(cfg, max_gaussians=50), comps)
    assert comps.store.get("job-1").result.gaussian_count == 50


def test_video_kept_when_retention_positive(env, cfg):
    comps, add, _ = env
    job = add()
    process_job(job, dataclasses.replace(cfg, video_retention_days=7), comps)
    assert comps.storage.exists("videos", "room-1/video.mp4")
    assert not comps.store.get("job-1").video_deleted


@pytest.mark.parametrize("kw,reason", [
    (dict(n_frames=4), FAIL_TOO_FEW_FRAMES),
    (dict(n_frames=60, blur_from=0), FAIL_BLURRY),
])
def test_bad_video_marks_failed_with_reason(env, cfg, kw, reason):
    comps, add, tmp = env
    job = add(**kw)
    assert process_job(job, cfg, comps) is False
    row = comps.store.get("job-1")
    assert row.status == STATUS_FAILED and row.failure_reason == reason
    assert not (tmp / "work" / "job-1").exists()
    assert not comps.storage.exists("models", "room-1/job-1.spz")


def test_missing_video_in_storage(env, cfg):
    comps, add, _ = env
    job = add()
    comps.storage.delete("videos", "room-1/video.mp4")
    process_job(job, cfg, comps)
    assert comps.store.get("job-1").failure_reason == FAIL_DOWNLOAD


def test_pose_failure(env, cfg):
    comps, add, _ = env

    class Failing:
        def estimate(self, frames, work_dir):
            raise PipelineError(FAIL_POSES, "zaregistrováno méně než 10 z 40 snímků")

    comps.poses = Failing()
    process_job(add(), cfg, comps)
    row = comps.store.get("job-1")
    assert row.failure_reason == FAIL_POSES and "zaregistrováno" in row.failure_detail


def test_unexpected_exception_is_contained(env, cfg):
    comps, add, tmp = env

    class Boom(SyntheticTrainer):
        def train(self, *a, **k):
            raise RuntimeError("kaboom")

    comps.trainer = Boom()
    assert process_job(add(), cfg, comps) is False
    row = comps.store.get("job-1")
    assert row.failure_reason == FAIL_OTHER and "kaboom" in row.failure_detail
    assert not (tmp / "work" / "job-1").exists()


def test_oom_retries_once_with_reduced_settings(env, cfg):
    comps, add, _ = env
    seen = []

    class OomOnce(SyntheticTrainer):
        def train(self, recon, settings, check_deadline):
            seen.append(settings)
            if len(seen) == 1:
                raise GpuOutOfMemory("oom")
            return super().train(recon, settings, check_deadline)

    comps.trainer = OomOnce()
    assert process_job(add(), dataclasses.replace(cfg, max_gaussians=400), comps)
    assert len(seen) == 2
    assert seen[1].max_gaussians == 200 and seen[1].image_scale < seen[0].image_scale


def test_oom_twice_fails_with_oom_reason(env, cfg):
    comps, add, _ = env

    class AlwaysOom(SyntheticTrainer):
        def train(self, *a, **k):
            raise GpuOutOfMemory("oom")

    comps.trainer = AlwaysOom()
    process_job(add(), cfg, comps)
    assert comps.store.get("job-1").failure_reason == FAIL_OOM


def test_timeout_during_training(env, cfg):
    comps, add, _ = env

    class Slow(SyntheticTrainer):
        def train(self, recon, settings, check_deadline):
            import time
            time.sleep(1.1)
            check_deadline()
            return super().train(recon, settings, check_deadline)

    comps.trainer = Slow()
    process_job(add(), dataclasses.replace(cfg, job_timeout_s=1), comps)
    assert comps.store.get("job-1").failure_reason == FAIL_TIMEOUT


def test_result_discarded_if_job_was_requeued_meanwhile(env, cfg):
    comps, add, tmp = env

    class Requeues(SyntheticTrainer):
        def train(self, recon, settings, check_deadline):
            comps.store.get("job-1").status = STATUS_PENDING  # simuluje cizí requeue
            return super().train(recon, settings, check_deadline)

    comps.trainer = Requeues()
    assert process_job(add(), cfg, comps) is False
    assert comps.store.get("job-1").status == STATUS_PENDING
    assert not (tmp / "work" / "job-1").exists()


def test_purge_old_videos(env, cfg):
    comps, add, _ = env
    job = add()
    process_job(job, dataclasses.replace(cfg, video_retention_days=7), comps)
    clock_row = comps.store.get("job-1")
    assert purge_old_videos(dataclasses.replace(cfg, video_retention_days=7), comps) == 0
    clock_row.finished_at -= 8 * 86400  # úloha je starší než retence
    assert purge_old_videos(dataclasses.replace(cfg, video_retention_days=7), comps) == 1
    assert not comps.storage.exists("videos", "room-1/video.mp4")
    assert purge_old_videos(dataclasses.replace(cfg, video_retention_days=7), comps) == 0


def test_main_loop_drains_queue_then_stops_on_idle(env, cfg):
    comps, add, _ = env
    add()  # job-1 je po add() už claimnutý – vrať ho do fronty
    comps.store.get("job-1").status = STATUS_PENDING
    comps.store.add_job(room_id="room-1", video_path="room-1/missing.mp4", job_id="job-2")

    class StopWhenIdle(Stop):
        def sleep(self, seconds):  # smyčka usíná = fronta je prázdná
            self.flag = True

    assert run(cfg, comps, StopWhenIdle()) == 2
    assert comps.store.get("job-1").status == STATUS_DONE          # nejstarší první
    assert comps.store.get("job-2").failure_reason == FAIL_DOWNLOAD
