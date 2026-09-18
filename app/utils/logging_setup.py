"""Rotating technical log.

Privacy rules enforced here (spec §19, §20):

* transcript text is **never** written to the log — helpers that need to
  mention a segment log its length and timing only;
* no audio samples, no device serial numbers, no user paths beyond the
  application's own directories;
* nothing leaves the machine. There is no network handler, by construction.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from app.utils.paths import logs_dir

LOG_FILENAME = "live-transcriber.log"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 5

_FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: int = logging.INFO, console: bool = True) -> Path:
    """Configure root logging once. Returns the log file path.

    Safe to call repeatedly; only the first call has an effect.
    """
    global _configured
    log_path = logs_dir() / LOG_FILENAME
    if _configured:
        return log_path

    logs_dir().mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # A read-only or locked app-data directory must not prevent startup.
        pass

    # A frozen GUI build has no console; sys.stderr may even be None.
    if console and sys.stderr is not None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(level)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    # Third-party libraries are chatty at DEBUG and can leak content.
    for noisy in ("faster_whisper", "huggingface_hub", "urllib3", "filelock", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def describe_segment(text: str, start: float, end: float) -> str:
    """Log-safe description of a transcript segment.

    Returns timing and size only — never the transcribed words.
    """
    return f"segment {start:.2f}-{end:.2f}s ({end - start:.2f}s, {len(text)} chars)"
