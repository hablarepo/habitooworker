"""Úložiště souborů. Buckety zůstávají soukromé – worker používá service role, žádné veřejné URL."""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Storage(ABC):
    @abstractmethod
    def download(self, bucket: str, path: str, dest: Path) -> None: ...

    @abstractmethod
    def upload(self, bucket: str, path: str, src: Path, content_type: str) -> None: ...

    @abstractmethod
    def delete(self, bucket: str, path: str) -> None: ...


class SupabaseStorage(Storage):
    def __init__(self, client: Any):
        self._db = client

    def download(self, bucket: str, path: str, dest: Path) -> None:
        data = self._db.storage.from_(bucket).download(path)
        dest.write_bytes(data)

    def upload(self, bucket: str, path: str, src: Path, content_type: str) -> None:
        # [DOPLNIT] limit velikosti souboru v bucketu (Supabase) – PLY může mít >100 MB
        self._db.storage.from_(bucket).upload(
            path, src, file_options={"content-type": content_type, "upsert": "true"})

    def delete(self, bucket: str, path: str) -> None:
        self._db.storage.from_(bucket).remove([path])


class LocalStorage(Storage):
    """Úložiště v adresáři – pro testy a režim synthetic."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _p(self, bucket: str, path: str) -> Path:
        return self.root / bucket / path

    def download(self, bucket, path, dest):
        shutil.copyfile(self._p(bucket, path), dest)

    def upload(self, bucket, path, src, content_type):
        p = self._p(bucket, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, p)

    def delete(self, bucket, path):
        self._p(bucket, path).unlink(missing_ok=True)

    def exists(self, bucket: str, path: str) -> bool:
        return self._p(bucket, path).exists()
