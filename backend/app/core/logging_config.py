"""Structured, JSON-ish logging setup.

Emits one line per event with a stable key=value tail so logs are greppable and
ingestible by a log pipeline (Loki, ELK, CloudWatch) without a heavy dependency.
"""

import logging
import sys


class KeyValueFormatter(logging.Formatter):
    RESERVED = set(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    ) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S%z')} "
            f"{record.levelname:<7} {record.name} :: {record.getMessage()}"
        )
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self.RESERVED and not key.startswith("_")
        }
        if extras:
            tail = " ".join(f"{k}={v}" for k, v in extras.items())
            base = f"{base} | {tail}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(KeyValueFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
