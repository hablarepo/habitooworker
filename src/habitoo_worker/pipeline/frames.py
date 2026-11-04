"""Extrakce snímků z videa: výběr, vyřazení rozmazaných, zmenšení."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..jobs.models import FAIL_BLURRY, FAIL_TOO_FEW_FRAMES
from .errors import PipelineError

log = logging.getLogger(__name__)

CANDIDATE_FACTOR = 3         # kolik kandidátů na jeden cílový snímek se vyhodnotí
BLUR_RELATIVE = 0.5          # snímek je rozmazaný, pokud ostrost < 50 % mediánu videa
BLUR_ABSOLUTE_MIN = 20.0     # a vždy pod touto absolutní hodnotou (rozptyl Laplaciánu)
# [DOPLNIT] prahy rozmazání je třeba doladit na reálném videu z telefonu


@dataclass(frozen=True)
class FrameSet:
    paths: list[Path]
    n_candidates: int
    n_blurry: int


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _resize(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def extract_frames(video: Path, out_dir: Path, target: int, min_frames: int,
                   max_side: int) -> FrameSet:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise PipelineError(FAIL_TOO_FEW_FRAMES, "video se nepodařilo otevřít")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise PipelineError(FAIL_TOO_FEW_FRAMES, "video nemá žádné snímky")
        n_cand = min(total, target * CANDIDATE_FACTOR)
        wanted = set(np.linspace(0, total - 1, n_cand).round().astype(int).tolist())

        cands: list[tuple[int, float, np.ndarray]] = []
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx in wanted:
                ok, img = cap.retrieve()
                if ok:
                    small = _resize(img, max_side)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    cands.append((idx, sharpness(gray), small))
            idx += 1
    finally:
        cap.release()

    if len(cands) < min_frames:
        raise PipelineError(
            FAIL_TOO_FEW_FRAMES, f"z videa lze získat jen {len(cands)} snímků, minimum je {min_frames}")

    median = float(np.median([s for _, s, _ in cands]))
    thr = max(median * BLUR_RELATIVE, BLUR_ABSOLUTE_MIN)
    sharp = [c for c in cands if c[1] >= thr]
    n_blurry = len(cands) - len(sharp)
    log.info("snímky: kandidátů %d, rozmazaných %d (práh %.1f, medián %.1f)",
             len(cands), n_blurry, thr, median)

    if len(sharp) < min_frames:
        raise PipelineError(
            FAIL_BLURRY, f"po vyřazení rozmazaných zbylo {len(sharp)} snímků, minimum je {min_frames}")

    if len(sharp) > target:  # rovnoměrně prořídit, zachovat pořadí
        keep = np.linspace(0, len(sharp) - 1, target).round().astype(int)
        sharp = [sharp[i] for i in keep]

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (_, _, img) in enumerate(sharp):
        p = out_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(p)
    return FrameSet(paths, len(cands), n_blurry)
