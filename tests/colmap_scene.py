"""Syntetická 3D scéna (barevné skvrny na stěnách krychle) s parallaxou pro test COLMAPu."""
from pathlib import Path

import cv2
import numpy as np


def render_scene(out: Path, n=40, W=640, H=480, seed=0):
    rng = np.random.default_rng(seed)
    # místnost: body na stěnách krychle 6x6x6 s texturou (barevné skvrny), uvnitř kamera na kruhu
    N = 6000
    pts = rng.uniform(-3, 3, (N, 3)); ax = rng.integers(0, 3, N); sg = rng.choice([-3, 3], N)
    pts[np.arange(N), ax] = sg
    cols = rng.integers(30, 255, (N, 3)); rad = rng.uniform(3, 9, N)
    f = 500.0; K = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]])
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        a = 0.25 * np.sin(2*np.pi*i/n)           # lehké kývání + posun => parallax
        cam = np.array([1.2*np.sin(2*np.pi*i/n), 0.3*np.cos(2*np.pi*i/n), -0.5])
        yaw = a
        R = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
        pc = (pts - cam) @ R.T
        z = pc[:, 2]; ok = z > 0.3
        uv = (pc[ok] @ K.T); uv = uv[:, :2] / uv[:, 2:3]
        order = np.argsort(-z[ok])
        img = np.full((H, W, 3), 128, np.uint8)
        for j in order:
            x, y = uv[j]
            if 0 <= x < W and 0 <= y < H:
                r = max(2, int(rad[ok][j] * 2.0 / z[ok][j]))
                c = tuple(int(v) for v in cols[ok][j])
                cv2.circle(img, (int(x), int(y)), r, c, -1, cv2.LINE_AA)
                cv2.circle(img, (int(x)+r//3, int(y)-r//3), max(1, r//3), (255,255,255) if j % 2 else (0,0,0), -1, cv2.LINE_AA)
        img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
        cv2.imwrite(str(out / f"frame_{i:04d}.jpg"), img)
    return sorted(out.glob("*.jpg"))
