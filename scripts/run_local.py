"""Lokální běh jednoho videa přes celou pipeline BEZ Supabase (falešná fronta + složka místo úložiště).

  python scripts/run_local.py --video samples/room.mp4 --mode real --out out/
  python scripts/run_local.py --synthetic --out out/          # bez GPU, jen test toku dat

Výsledek (model.spz, model.ply) je ve složce --out.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from habitoo_worker.config import REAL, SYNTHETIC, Config  # noqa: E402
from habitoo_worker.jobs.fake_store import FakeJobStore  # noqa: E402
from habitoo_worker.log import setup_logging  # noqa: E402
from habitoo_worker.pipeline.runner import Components, model_paths, process_job  # noqa: E402
from habitoo_worker.storage import LocalStorage  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, help="vstupní video (u --synthetic se vygeneruje)")
    ap.add_argument("--mode", choices=[REAL, SYNTHETIC], default=REAL)
    ap.add_argument("--synthetic", action="store_true", help="zkratka za --mode synthetic")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--work", type=Path, default=None, help="pracovní složka (výchozí: dočasná)")
    args = ap.parse_args()
    mode = SYNTHETIC if args.synthetic else args.mode
    setup_logging()

    cfg = dataclasses.replace(
        Config.from_env({"WORKER_MODE": SYNTHETIC}), mode=mode, video_bucket="videos",
        model_bucket="models", job_timeout_s=6 * 3600, video_retention_days=1,
        work_dir=str(args.work or tempfile.mkdtemp(prefix="habitoo-work-")))

    root = Path(tempfile.mkdtemp(prefix="habitoo-local-"))
    storage = LocalStorage(root)
    video = args.video
    if mode == SYNTHETIC and video is None:
        from habitoo_worker.pipeline.synthetic import make_synthetic_video
        video = make_synthetic_video(root / "synthetic.mp4")
    if video is None or not video.exists():
        print("Chybí video (--video)", file=sys.stderr)
        return 2
    storage.upload("videos", "room/video.mp4", video, "video/mp4")

    store = FakeJobStore()
    store.add_job(room_id="room", video_path="room/video.mp4", job_id="local")
    if mode == SYNTHETIC:
        from habitoo_worker.pipeline.synthetic import SyntheticPoseEstimator, SyntheticTrainer
        poses, trainer = SyntheticPoseEstimator(), SyntheticTrainer()
    else:
        from habitoo_worker.pipeline.poses import ColmapPoseEstimator
        from habitoo_worker.pipeline.train import GsplatTrainer
        poses, trainer = ColmapPoseEstimator(), GsplatTrainer()

    job = store.claim_next()
    ok = process_job(job, cfg, Components(store, storage, poses, trainer))
    row = store.get("local")
    if not ok:
        print(f"SELHALO: {row.failure_reason} – {row.failure_detail}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    spz, ply = model_paths(job)
    shutil.copyfile(root / "models" / spz, args.out / "model.spz")
    shutil.copyfile(root / "models" / ply, args.out / "model.ply")
    r = row.result
    print(f"HOTOVO: {r.gaussian_count} gaussiánů, {r.processing_seconds:.0f} s, "
          f"SPZ {r.spz_bytes / 1e6:.1f} MB, PLY {r.ply_bytes / 1e6:.1f} MB -> {args.out}/")
    shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
