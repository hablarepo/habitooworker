"""Náhrady těžkých kroků pro testování bez GPU (WORKER_MODE=synthetic).

Skutečně se zkouší tok dat, limity, export a úklid. COLMAP ani gsplat se tu NEPOUŽÍVAJÍ,
kvalita výsledku nic neříká.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..poses import PoseEstimator, Reconstruction
from ..splat import Splat, sh_rest_count
from ..train import TrainResult, Trainer, TrainSettings


def make_synthetic_video(path: Path, n_frames: int = 90, size=(320, 240), seed: int = 0,
                         blur_from: int | None = None, blur_to: int | None = None) -> Path:
    """Video s texturovanou scénou, kterou kamera pomalu posouvá. Snímky blur_from..blur_to jsou rozmazané."""
    rng = np.random.default_rng(seed)
    w, h = size
    base = cv2.resize(rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8), (w * 2, h * 2),
                      interpolation=cv2.INTER_CUBIC)
    base = cv2.add(base, rng.integers(0, 40, base.shape, dtype=np.uint8))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter nelze otevřít")
    for i in range(n_frames):
        dx = int(i * (w / max(n_frames, 1)))
        frame = base[h // 2:h // 2 + h, dx:dx + w].copy()
        if blur_from is not None and blur_from <= i < (blur_to or n_frames):
            frame = cv2.GaussianBlur(frame, (0, 0), 6)
        vw.write(frame)
    vw.release()
    return path


class SyntheticPoseEstimator(PoseEstimator):
    """Kamery na kruhu kolem počátku se známými parametry + náhodné body."""

    def estimate(self, frames: list[Path], work_dir: Path) -> Reconstruction:
        n = len(frames)
        rng = np.random.default_rng(0)
        w2c = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        for i in range(n):
            a = 2 * np.pi * i / n
            cam_pos = np.array([3 * np.cos(a), 0.0, 3 * np.sin(a)])
            fwd = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross(fwd, [0, -1, 0]); right /= np.linalg.norm(right)  # y dolů
            down = np.cross(fwd, right)
            R = np.stack([right, down, fwd])  # řádky = osy kamery
            w2c[i, :3, :3] = R
            w2c[i, :3, 3] = -R @ cam_pos
        img = cv2.imread(str(frames[0]))
        h, w = img.shape[:2]
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float32)
        pts = rng.normal(size=(500, 3)).astype(np.float32)
        rgb = rng.integers(0, 255, (500, 3), dtype=np.uint8)
        return Reconstruction(list(frames), w2c, np.tile(K, (n, 1, 1)),
                              np.tile(np.array([w, h]), (n, 1)), pts, rgb)


class SyntheticTrainer(Trainer):
    """Bez optimalizace: z bodů vyrobí splat a respektuje strop max_gaussians."""

    def train(self, recon: Reconstruction, settings: TrainSettings,
              check_deadline: Callable[[], None]) -> TrainResult:
        t0 = time.monotonic()
        check_deadline()
        rng = np.random.default_rng(1)
        n = min(len(recon.points_xyz), settings.max_gaussians)
        k = sh_rest_count(settings.sh_degree)
        q = rng.normal(size=(n, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        splat = Splat(
            means=recon.points_xyz[:n].copy(),
            log_scales=np.full((n, 3), -3.0, np.float32),
            quats=q,
            logit_opacity=rng.normal(size=n).astype(np.float32),
            sh0=((recon.points_rgb[:n] / 255.0 - 0.5) / 0.28209479).astype(np.float32),
            sh_rest=(rng.normal(size=(n, k, 3)) * 0.05).astype(np.float32),
            sh_degree=settings.sh_degree)
        return TrainResult(splat, time.monotonic() - t0, None)
