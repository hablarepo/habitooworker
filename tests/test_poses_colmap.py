"""Skutečný COLMAP (pycolmap, CPU) na syntetické scéně – ověřuje volání API a čtení výsledku."""
import cv2
import numpy as np
import pytest
from colmap_scene import render_scene

pycolmap = pytest.importorskip("pycolmap")

from habitoo_worker.jobs.models import FAIL_POSES  # noqa: E402
from habitoo_worker.pipeline.errors import PipelineError  # noqa: E402
from habitoo_worker.pipeline.poses import ColmapPoseEstimator  # noqa: E402


def test_colmap_recovers_cameras_and_focal(tmp_path):
    frames = render_scene(tmp_path / "frames", n=40)
    rec = ColmapPoseEstimator().estimate(frames, tmp_path)
    assert len(rec.image_paths) >= 30
    assert rec.world_to_cam.shape == (len(rec.image_paths), 4, 4)
    assert len(rec.points_xyz) > 200 and rec.points_rgb.shape == rec.points_xyz.shape
    assert all(p.exists() for p in rec.image_paths)
    assert rec.Ks[0][0, 0] == pytest.approx(500, rel=0.05)  # skutečné f = 500 px
    R = rec.world_to_cam[:, :3, :3]
    np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.tile(np.eye(3), (len(R), 1, 1)), atol=1e-4)


def test_colmap_failure_on_featureless_images(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(20):
        cv2.imwrite(str(d / f"frame_{i:04d}.jpg"), np.full((240, 320, 3), 128, np.uint8))
    with pytest.raises(PipelineError) as e:
        ColmapPoseEstimator().estimate(sorted(d.glob("*.jpg")), tmp_path)
    assert e.value.reason == FAIL_POSES
