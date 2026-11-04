"""Export splatu do PLY (standardní 3DGS) a SPZ (Niantic, legacy gzip v2).

SPZ formát podle referenčního zdroje https://github.com/nianticlabs/spz (MIT),
src/cc/load-spz.cc a load-spz.h. Zapisuje se verze 2: 16B hlavička + pole
(pozice, alfa, barvy, měřítka, rotace, SH), celé zabalené gzipem. Současný
loader Niantic verzi 4 (ZSTD) zapisuje, ale starší gzip verze pořád čte.
Vstup je v souřadnicích RDF (jako COLMAP/PLY), uloží se do RUB (Three.js),
jak formát SPZ předpokládá.
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

from .splat import Splat, sh_rest_count

NGSP_MAGIC = 0x5053474E
SPZ_VERSION = 2
FRACTIONAL_BITS = 12
COLOR_SCALE = 0.15
SH1_BITS = 5       # výchozí hodnoty Niantic (DEFAULT_SH1_BITS / DEFAULT_SH_REST_BITS)
SH_REST_BITS = 4

# Převod RDF -> RUB: osy y a z se obrátí (flipP = {1,-1,-1}, flipQ = {y*z,x*z,x*y}).
_FLIP_P = np.array([1.0, -1.0, -1.0], dtype=np.float32)
_FLIP_Q = np.array([1.0, -1.0, -1.0], dtype=np.float32)  # pro x,y,z složku kvaternionu
# Znaménka SH koeficientů po flipu (x=1, y=-1, z=-1), tabulka flipSh z splat-types.h, idx 0..14.
_FLIP_SH = np.array(
    [-1, -1, 1,            # stupeň 1: y, z, x
     -1, 1, 1, -1, 1,      # stupeň 2: x*y, y*z, 1, x*z, 1
     -1, 1, -1, -1, 1, -1, 1],  # stupeň 3: y, x*y*z, y, z, x, z, x
    dtype=np.float32)


def _round_half_away(x: np.ndarray) -> np.ndarray:
    """std::round z C++ (půlky od nuly); np.round zaokrouhluje půlky k sudému."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(_round_half_away(x), 0, 255).astype(np.uint8)


def _quantize_sh(x: np.ndarray, bits: int) -> np.ndarray:
    bucket = 1 << (8 - bits)
    q = _round_half_away(x * 128.0).astype(np.int32) + 128
    q = (q + bucket // 2) // bucket * bucket
    return np.clip(q, 0, 255).astype(np.uint8)


def write_ply(splat: Splat, path: Path) -> int:
    """Standardní 3DGS PLY (binary little endian, souřadnice beze změny). Vrací velikost v B."""
    splat.validate()
    n, k = splat.count, sh_rest_count(splat.sh_degree)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * k)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"

    rest = splat.sh_rest.transpose(0, 2, 1).reshape(n, 3 * k)  # kanálově (R..., G..., B...)
    data = np.concatenate(
        [splat.means, np.zeros((n, 3), np.float32), splat.sh0, rest,
         splat.logit_opacity[:, None], splat.log_scales, splat.quats],
        axis=1).astype("<f4")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    return Path(path).stat().st_size


def write_spz(splat: Splat, path: Path) -> int:
    """SPZ verze 2 (gzip). Vrací velikost v B."""
    splat.validate()
    n, deg = splat.count, splat.sh_degree
    k = sh_rest_count(deg)

    pos = _round_half_away(splat.means * _FLIP_P * float(1 << FRACTIONAL_BITS)).astype(np.int32)
    pos_bytes = np.stack([pos & 0xFF, (pos >> 8) & 0xFF, (pos >> 16) & 0xFF], axis=2)
    pos_bytes = pos_bytes.astype(np.uint8)  # (N,3,3): pro každou osu 3 bajty little endian

    alphas = _to_uint8(1.0 / (1.0 + np.exp(-splat.logit_opacity.astype(np.float64))) * 255.0)
    colors = _to_uint8(splat.sh0 * (COLOR_SCALE * 255.0) + 0.5 * 255.0)
    scales = _to_uint8((splat.log_scales + 10.0) * 16.0)

    q = splat.quats[:, [1, 2, 3, 0]].astype(np.float64)  # w,x,y,z -> x,y,z,w
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    q[:, :3] *= _FLIP_Q
    sign = np.where(q[:, 3] < 0, -127.5, 127.5)[:, None]
    rot = _to_uint8(q[:, :3] * sign + 127.5)

    parts = [
        struct.pack("<IIIBBBB", NGSP_MAGIC, SPZ_VERSION, n, deg, FRACTIONAL_BITS, 0, 0),
        pos_bytes.tobytes(), alphas.tobytes(), colors.tobytes(),
        scales.tobytes(), rot.tobytes(),
    ]
    if deg > 0:
        sh = splat.sh_rest * _FLIP_SH[:k, None]           # (N,K,3)
        out = np.empty((n, k, 3), np.uint8)
        out[:, :3, :] = _quantize_sh(sh[:, :3, :], SH1_BITS)
        if k > 3:
            out[:, 3:, :] = _quantize_sh(sh[:, 3:, :], SH_REST_BITS)
        parts.append(out.tobytes())

    with open(path, "wb") as f:
        f.write(gzip.compress(b"".join(parts), compresslevel=9))
    return Path(path).stat().st_size
