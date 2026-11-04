import gzip
import struct

import numpy as np
import pytest

from habitoo_worker.pipeline.export import (
    COLOR_SCALE, NGSP_MAGIC, write_ply, write_spz,
)
from habitoo_worker.pipeline.splat import Splat, sh_rest_count


def make_splat(n=200, deg=3, seed=0) -> Splat:
    r = np.random.default_rng(seed)
    q = r.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return Splat(
        means=r.uniform(-5, 5, (n, 3)).astype(np.float32),
        log_scales=r.uniform(-6, -1, (n, 3)).astype(np.float32),
        quats=q, logit_opacity=r.normal(size=n).astype(np.float32),
        sh0=r.uniform(-1, 1, (n, 3)).astype(np.float32),
        sh_rest=r.uniform(-0.3, 0.3, (n, sh_rest_count(deg), 3)).astype(np.float32),
        sh_degree=deg)


def read_ply(path):
    raw = path.read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    lines = head.decode().splitlines()
    n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    names = [l.split()[-1] for l in lines if l.startswith("property")]
    return names, np.frombuffer(body, "<f4").reshape(n, len(names))


def test_ply_roundtrip(tmp_path):
    s = make_splat()
    size = write_ply(s, tmp_path / "a.ply")
    assert size == (tmp_path / "a.ply").stat().st_size
    names, d = read_ply(tmp_path / "a.ply")
    assert len(names) == 3 + 3 + 3 + 45 + 1 + 3 + 4 and names[0] == "x" and names[-1] == "rot_3"
    col = {n: i for i, n in enumerate(names)}
    np.testing.assert_allclose(d[:, :3], s.means)
    np.testing.assert_allclose(d[:, col["f_dc_0"]:col["f_dc_2"] + 1], s.sh0)
    np.testing.assert_allclose(d[:, col["opacity"]], s.logit_opacity)
    np.testing.assert_allclose(d[:, col["scale_0"]:col["scale_2"] + 1], s.log_scales)
    np.testing.assert_allclose(d[:, col["rot_0"]:col["rot_3"] + 1], s.quats)
    # f_rest je řazen po kanálech: nejdřív 15 koeficientů R, pak G, pak B
    rest = d[:, col["f_rest_0"]:col["f_rest_44"] + 1].reshape(-1, 3, 15)
    np.testing.assert_allclose(rest.transpose(0, 2, 1), s.sh_rest)


def decode_spz(path):
    """Nezávislý dekodér podle unpack v nianticlabs/spz (load-spz.cc), výstup v RUB."""
    raw = gzip.decompress(path.read_bytes())
    magic, ver, n, deg, frac, flags, _ = struct.unpack("<IIIBBBB", raw[:16])
    assert magic == NGSP_MAGIC and ver == 2 and flags == 0
    k = (deg + 1) ** 2 - 1
    o = 16
    pos = np.frombuffer(raw, np.uint8, n * 9, o).reshape(n, 3, 3).astype(np.int32); o += n * 9
    fixed = pos[..., 0] | (pos[..., 1] << 8) | (pos[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    alpha = np.frombuffer(raw, np.uint8, n, o); o += n
    col = np.frombuffer(raw, np.uint8, n * 3, o).reshape(n, 3); o += n * 3
    sc = np.frombuffer(raw, np.uint8, n * 3, o).reshape(n, 3); o += n * 3
    rot = np.frombuffer(raw, np.uint8, n * 3, o).reshape(n, 3); o += n * 3
    sh = np.frombuffer(raw, np.uint8, n * k * 3, o).reshape(n, k, 3); o += n * k * 3
    assert o == len(raw)  # žádná data navíc ani chybějící
    xyz = rot / 127.5 - 1.0
    w = np.sqrt(np.maximum(0, 1 - (xyz ** 2).sum(1)))
    return dict(
        pos=fixed / float(1 << frac), alpha=alpha / 255.0,
        sh0=((col / 255.0) - 0.5) / COLOR_SCALE, scale=sc / 16.0 - 10.0,
        quat_xyzw=np.concatenate([xyz, w[:, None]], 1), sh=(sh.astype(np.float32) - 128) / 128.0)


def test_spz_roundtrip_within_quantization_error(tmp_path):
    s = make_splat()
    size = write_spz(s, tmp_path / "a.spz")
    assert size == (tmp_path / "a.spz").stat().st_size
    d = decode_spz(tmp_path / "a.spz")
    # RDF -> RUB: y a z se obrátí
    np.testing.assert_allclose(d["pos"], s.means * [1, -1, -1], atol=1 / 4096)
    np.testing.assert_allclose(d["alpha"], 1 / (1 + np.exp(-s.logit_opacity)), atol=1 / 255)
    np.testing.assert_allclose(d["sh0"], s.sh0, atol=0.5 / (255 * COLOR_SCALE) + 1e-6)
    np.testing.assert_allclose(d["scale"], s.log_scales, atol=1 / 16)
    # kvaternion: znaménko se normalizuje na w>=0, složky x,y,z po flipu (1,-1,-1)
    q = s.quats[:, [1, 2, 3, 0]].copy()
    q[:, 1:3] *= -1
    q *= np.where(q[:, 3] < 0, -1, 1)[:, None]
    np.testing.assert_allclose(d["quat_xyzw"][:, :3], q[:, :3], atol=1 / 127)
    # SH: stupeň 1 po flipu (-1,-1,+1), kvantizace 5 bitů -> bucket 8/128
    flip = np.array([-1, -1, 1], np.float32)[None, :, None]
    np.testing.assert_allclose(d["sh"][:, :3], s.sh_rest[:, :3] * flip, atol=8 / 128)


def test_spz_is_smaller_than_ply(tmp_path):
    s = make_splat(n=2000)
    assert write_spz(s, tmp_path / "a.spz") < write_ply(s, tmp_path / "a.ply") / 3


def test_spz_degree_zero(tmp_path):
    s = make_splat(n=50, deg=0)
    write_spz(s, tmp_path / "a.spz")
    assert decode_spz(tmp_path / "a.spz")["sh"].shape == (50, 0, 3)


def test_validate_rejects_nan(tmp_path):
    s = make_splat(n=10)
    s.means[0, 0] = np.nan
    with pytest.raises(ValueError):
        write_ply(s, tmp_path / "a.ply")
    with pytest.raises(ValueError):
        write_spz(s, tmp_path / "a.spz")
