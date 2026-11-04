import dataclasses

import pytest

from habitoo_worker.config import SYNTHETIC, Config
from habitoo_worker.jobs.fake_store import FakeJobStore
from habitoo_worker.pipeline.runner import Components
from habitoo_worker.pipeline.synthetic import (
    SyntheticPoseEstimator, SyntheticTrainer, make_synthetic_video,
)
from habitoo_worker.storage import LocalStorage


@pytest.fixture
def cfg(tmp_path):
    base = Config.from_env({"WORKER_MODE": SYNTHETIC})
    return dataclasses.replace(
        base, video_bucket="videos", model_bucket="models", job_timeout_s=600,
        video_retention_days=0, work_dir=str(tmp_path / "work"),
        target_frames=40, min_frames=10, max_gaussians=300)


@pytest.fixture
def env(tmp_path, cfg):
    store = FakeJobStore()
    storage = LocalStorage(tmp_path / "storage")
    comps = Components(store, storage, SyntheticPoseEstimator(), SyntheticTrainer())

    def add(video_path=None, name="room-1/video.mp4", **video_kw):
        if video_path is None:
            video_path = make_synthetic_video(tmp_path / "v.mp4", **video_kw)
        storage.upload("videos", name, video_path, "video/mp4")
        store.add_job(room_id="room-1", video_path=name, job_id="job-1")
        return store.claim_next()

    return comps, add, tmp_path
