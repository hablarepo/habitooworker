"""Odhad poloh kamer (COLMAP přes pycolmap, BSD-3)."""
from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..jobs.models import FAIL_POSES
from .errors import PipelineError

log = logging.getLogger(__name__)

MIN_REGISTERED_RATIO = 0.5   # alespoň polovina snímků musí mít spočítanou polohu
MIN_REGISTERED_ABS = 10      # [DOPLNIT] prahy doladit na reálných videích
SEQUENTIAL_OVERLAP = 20


@dataclass
class Reconstruction:
    """Výsledek odhadu poloh, připravený pro trénink (bez zkreslení čoček, pinhole)."""
    image_paths: list[Path]
    world_to_cam: np.ndarray   # (N,4,4) float32, OpenCV konvence (x vpravo, y dolů, z dopředu)
    Ks: np.ndarray             # (N,3,3) float32
    sizes: np.ndarray          # (N,2) int, (šířka, výška)
    points_xyz: np.ndarray     # (M,3) float32
    points_rgb: np.ndarray     # (M,3) uint8


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frames: list[Path], work_dir: Path) -> Reconstruction: ...


class ColmapPoseEstimator(PoseEstimator):
    def estimate(self, frames: list[Path], work_dir: Path) -> Reconstruction:
        import pycolmap  # import až tady: v testech bez COLMAPu se nenačítá

        frames_dir = frames[0].parent
        n = len(frames)
        need = max(MIN_REGISTERED_ABS, int(n * MIN_REGISTERED_RATIO))
        best = None
        for mode in ("sequential", "exhaustive"):
            db = work_dir / f"colmap_{mode}.db"
            sparse = work_dir / f"sparse_{mode}"
            shutil.rmtree(sparse, ignore_errors=True)
            sparse.mkdir(parents=True)
            try:
                reader = pycolmap.ImageReaderOptions()
                reader.camera_model = "SIMPLE_RADIAL"
                pycolmap.extract_features(db, frames_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                                          reader_options=reader)
                if mode == "sequential":
                    opts = pycolmap.SequentialPairingOptions()
                    opts.overlap = SEQUENTIAL_OVERLAP
                    pycolmap.match_sequential(db, pairing_options=opts)
                else:
                    pycolmap.match_exhaustive(db)
                recs = pycolmap.incremental_mapping(db, frames_dir, sparse)
            except Exception as e:  # pycolmap vyhazuje obecné RuntimeError
                log.warning("COLMAP (%s) selhal: %s", mode, e)
                continue
            if recs:
                rid, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
                log.info("COLMAP %s: %d/%d snímků zaregistrováno", mode, rec.num_reg_images(), n)
                if rec.num_reg_images() >= need:
                    best = (sparse / str(rid), rec)
                    break
        if best is None:
            raise PipelineError(FAIL_POSES, f"zaregistrováno méně než {need} z {n} snímků")

        model_dir, _ = best
        undist = work_dir / "undistorted"
        shutil.rmtree(undist, ignore_errors=True)
        pycolmap.undistort_images(undist, model_dir, frames_dir)
        return _read(pycolmap.Reconstruction(undist / "sparse"), undist / "images")


def _read(rec, images_dir: Path) -> Reconstruction:
    paths, w2c, Ks, sizes = [], [], [], []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        if not img.has_pose:
            continue
        m = np.eye(4, dtype=np.float32)
        m[:3, :] = img.cam_from_world().matrix()
        cam = rec.cameras[img.camera_id]
        paths.append(images_dir / img.name)
        w2c.append(m)
        Ks.append(np.asarray(cam.calibration_matrix(), dtype=np.float32))
        sizes.append((cam.width, cam.height))
    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts], dtype=np.float32).reshape(-1, 3)
    rgb = np.array([p.color for p in pts], dtype=np.uint8).reshape(-1, 3)
    return Reconstruction(paths, np.stack(w2c), np.stack(Ks), np.array(sizes, dtype=np.int64), xyz, rgb)
