"""Datový model hotového splatu (parametry ve stejné podobě jako při tréninku gsplat)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sh_rest_count(sh_degree: int) -> int:
    """Počet SH koeficientů mimo DC pro daný stupeň: 0, 3, 8, 15."""
    return (sh_degree + 1) ** 2 - 1


@dataclass
class Splat:
    means: np.ndarray      # (N,3) float32, světové souřadnice (COLMAP/OpenCV: x vpravo, y dolů, z dopředu)
    log_scales: np.ndarray # (N,3) float32, log měřítko
    quats: np.ndarray      # (N,4) float32, pořadí w,x,y,z
    logit_opacity: np.ndarray  # (N,) float32
    sh0: np.ndarray        # (N,3) float32, DC koeficient
    sh_rest: np.ndarray    # (N,K,3) float32, K = sh_rest_count(sh_degree)
    sh_degree: int = 3

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    def validate(self) -> None:
        n = self.count
        k = sh_rest_count(self.sh_degree)
        shapes = {
            "means": (n, 3), "log_scales": (n, 3), "quats": (n, 4),
            "logit_opacity": (n,), "sh0": (n, 3), "sh_rest": (n, k, 3),
        }
        for name, shape in shapes.items():
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name}: tvar {arr.shape}, čekáno {shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} obsahuje NaN/Inf")
