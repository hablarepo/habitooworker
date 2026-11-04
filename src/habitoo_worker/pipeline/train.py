"""Trénink Gaussian splatu přes gsplat (Apache-2.0) s limitem počtu gaussiánů (MCMC strategie)."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from ..jobs.models import FAIL_OTHER
from .errors import GpuOutOfMemory, PipelineError
from .poses import Reconstruction
from .splat import Splat, sh_rest_count

log = logging.getLogger(__name__)

SH0_C = 0.28209479177387814  # konstanta SH stupně 0


@dataclass(frozen=True)
class TrainSettings:
    max_gaussians: int
    steps: int = 7000        # [DOPLNIT] počet kroků doladit podle kvality/času na GTX 1660 Super
    image_scale: float = 1.0  # další zmenšení snímků při tréninku (po OOM se sníží)
    sh_degree: int = 3

    def reduced(self) -> "TrainSettings":
        """Úspornější nastavení pro opakování po nedostatku paměti GPU."""
        return replace(self, max_gaussians=self.max_gaussians // 2,
                       image_scale=self.image_scale * 0.75)


@dataclass(frozen=True)
class TrainResult:
    splat: Splat
    seconds: float
    peak_vram_mb: float | None  # None = bez GPU (syntetický režim)


class Trainer(ABC):
    @abstractmethod
    def train(self, recon: Reconstruction, settings: TrainSettings,
              check_deadline: Callable[[], None]) -> TrainResult: ...


def init_scales(points: np.ndarray, k: int = 3, chunk: int = 512) -> np.ndarray:
    """Počáteční měřítko = průměrná vzdálenost ke k nejbližším sousedům (po částech, šetří paměť)."""
    n = len(points)
    if n <= k:
        return np.full(n, 0.01, np.float32)
    out = np.empty(n, np.float32)
    for i in range(0, n, chunk):
        d = np.linalg.norm(points[i:i + chunk, None, :] - points[None, :, :], axis=2)
        d.sort(axis=1)
        out[i:i + chunk] = d[:, 1:k + 1].mean(axis=1)
    return np.maximum(out, 1e-6)


class GsplatTrainer(Trainer):
    """Skutečný trénink na CUDA GPU. NEBYL ověřen bez GPU – viz README (první reálný běh)."""

    def train(self, recon, settings, check_deadline):
        import cv2
        import torch
        from gsplat.rendering import rasterization
        from gsplat.strategy import MCMCStrategy

        if not torch.cuda.is_available():
            raise PipelineError(FAIL_OTHER, "CUDA GPU není dostupná")
        dev = torch.device("cuda")
        t0 = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            return self._run(recon, settings, check_deadline, torch, cv2, rasterization,
                             MCMCStrategy, dev, t0)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise GpuOutOfMemory(str(e)) from e

    def _run(self, recon, s, check_deadline, torch, cv2, rasterization, MCMCStrategy, dev, t0):
        scale = s.image_scale
        images, Ks, sizes = [], [], []
        for p, K, (w, h) in zip(recon.image_paths, recon.Ks, recon.sizes):
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            if (nw, nh) != img.shape[1::-1]:
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            K = K.copy()
            K[0] *= nw / w
            K[1] *= nh / h
            images.append(torch.from_numpy(img))  # uint8 zůstává v RAM, na GPU po jednom
            Ks.append(K)
            sizes.append((nw, nh))
        viewmats = torch.from_numpy(recon.world_to_cam).to(dev)
        Ks_t = torch.from_numpy(np.stack(Ks)).to(dev)

        # poloha kamer -> velikost scény (pro learning rate pozic)
        centers = -np.einsum("nij,nj->ni", recon.world_to_cam[:, :3, :3].transpose(0, 2, 1),
                             recon.world_to_cam[:, :3, 3])
        scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1 or 1.0

        pts, rgb = recon.points_xyz, recon.points_rgb
        if len(pts) > s.max_gaussians:
            keep = np.random.default_rng(0).choice(len(pts), s.max_gaussians, replace=False)
            pts, rgb = pts[keep], rgb[keep]
        if len(pts) < 100:
            raise PipelineError(FAIL_OTHER, f"COLMAP vrátil jen {len(pts)} 3D bodů")
        dist = init_scales(pts)
        n = len(pts)
        k = sh_rest_count(s.sh_degree)
        sh0 = ((rgb.astype(np.float32) / 255.0 - 0.5) / SH0_C)[:, None, :]
        quats = np.random.default_rng(0).normal(size=(n, 4)).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)

        P = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.from_numpy(pts).to(dev)),
            "scales": torch.nn.Parameter(torch.log(torch.from_numpy(dist).to(dev))[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(torch.from_numpy(quats).to(dev)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=dev))),
            "sh0": torch.nn.Parameter(torch.from_numpy(sh0).to(dev)),
            "shN": torch.nn.Parameter(torch.zeros((n, k, 3), device=dev)),
        })
        lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
               "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
        opts = {name: torch.optim.Adam([{"params": P[name], "lr": lr, "name": name}], eps=1e-15)
                for name, lr in lrs.items()}
        sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / s.steps))
        strategy = MCMCStrategy(cap_max=s.max_gaussians, verbose=False)
        strategy.check_sanity(P, opts)
        state = strategy.initialize_state()

        order = np.random.default_rng(1)
        for step in range(s.steps):
            if step % 100 == 0:
                check_deadline()
            i = int(order.integers(len(images)))
            gt = images[i].to(dev).float().div_(255.0)[None]
            w, h = sizes[i]
            deg = min(step // 1000, s.sh_degree)
            colors = torch.cat([P["sh0"], P["shN"]], dim=1)
            render, _alpha, info = rasterization(
                P["means"], P["quats"], torch.exp(P["scales"]), torch.sigmoid(P["opacities"]),
                colors, viewmats[i:i + 1], Ks_t[i:i + 1], w, h, sh_degree=deg, packed=False)
            loss = torch.nn.functional.l1_loss(render, gt)
            loss = loss + 0.01 * torch.sigmoid(P["opacities"]).mean() + 0.01 * torch.exp(P["scales"]).mean()
            loss.backward()
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
            sched.step()
            strategy.step_post_backward(P, opts, state, step, info, lr=sched.get_last_lr()[0])
            if step % 500 == 0 or step == s.steps - 1:
                log.info("trénink %d/%d loss=%.4f gaussiánů=%d VRAM=%.0f MB", step, s.steps,
                         float(loss), P["means"].shape[0], torch.cuda.max_memory_allocated() / 2**20)

        peak = torch.cuda.max_memory_allocated() / 2**20
        g = {name: P[name].detach().cpu().numpy() for name in P.keys()}
        splat = Splat(g["means"], g["scales"], g["quats"], g["opacities"],
                      g["sh0"][:, 0, :], g["shN"], s.sh_degree)
        return TrainResult(splat, time.monotonic() - t0, peak)
