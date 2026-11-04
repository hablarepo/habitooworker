import cv2
import numpy as np
import pytest

from habitoo_worker.jobs.models import FAIL_BLURRY, FAIL_TOO_FEW_FRAMES
from habitoo_worker.pipeline.errors import PipelineError
from habitoo_worker.pipeline.frames import extract_frames
from habitoo_worker.pipeline.synthetic import make_synthetic_video


def test_extracts_target_count_and_resizes(tmp_path):
    v = make_synthetic_video(tmp_path / "v.mp4", n_frames=120, size=(640, 480))
    fs = extract_frames(v, tmp_path / "f", target=50, min_frames=10, max_side=320)
    assert len(fs.paths) == 50
    img = cv2.imread(str(fs.paths[0]))
    assert max(img.shape[:2]) == 320
    assert [p.name for p in fs.paths] == sorted(p.name for p in fs.paths)


def test_fewer_frames_than_target_keeps_all_sharp(tmp_path):
    v = make_synthetic_video(tmp_path / "v.mp4", n_frames=30)
    fs = extract_frames(v, tmp_path / "f", target=100, min_frames=10, max_side=320)
    assert len(fs.paths) == 30


def test_blurry_segment_is_dropped(tmp_path):
    v = make_synthetic_video(tmp_path / "v.mp4", n_frames=120, blur_from=0, blur_to=40)
    fs = extract_frames(v, tmp_path / "f", target=120, min_frames=10, max_side=320)
    assert fs.n_blurry >= 30
    assert len(fs.paths) <= 85


def test_fully_blurry_video_fails_with_blurry_reason(tmp_path):
    v = make_synthetic_video(tmp_path / "v.mp4", n_frames=60, blur_from=0)
    with pytest.raises(PipelineError) as e:
        extract_frames(v, tmp_path / "f", target=60, min_frames=10, max_side=320)
    assert e.value.reason == FAIL_BLURRY


def test_short_video_fails_too_few(tmp_path):
    v = make_synthetic_video(tmp_path / "v.mp4", n_frames=5)
    with pytest.raises(PipelineError) as e:
        extract_frames(v, tmp_path / "f", target=60, min_frames=10, max_side=320)
    assert e.value.reason == FAIL_TOO_FEW_FRAMES


def test_unreadable_file_fails(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(PipelineError) as e:
        extract_frames(bad, tmp_path / "f", target=60, min_frames=10, max_side=320)
    assert e.value.reason == FAIL_TOO_FEW_FRAMES
