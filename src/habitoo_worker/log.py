"""Logování na stdout (čte ho `docker logs`) s ochranou tajemství."""
from __future__ import annotations

import logging
import sys


class RedactFilter(logging.Filter):
    """Nahradí případný výskyt tajemství v hlášce za ***."""

    def __init__(self, secrets: list[str]):
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        if self._secrets:
            msg = record.getMessage()
            for s in self._secrets:
                msg = msg.replace(s, "***")
            record.msg, record.args = msg, ()
        return True


def setup_logging(secrets: list[str] | None = None, level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    handler.addFilter(RedactFilter(secrets or []))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
