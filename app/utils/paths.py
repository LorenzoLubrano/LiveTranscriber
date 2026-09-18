"""Application directories.

All user data lives under a single root so the app is easy to inspect, back up
and uninstall. Nothing is written outside these directories.

Layout::

    %LOCALAPPDATA%\\LiveTranscriber\\
        models/                     Whisper models (CTranslate2 format)
        logs/live-transcriber.log   Rotating technical log
        config.json                 Settings (QSettings uses the registry)
        cache/

    %USERPROFILE%\\Documents\\LiveTranscriber\\Recordings\\
        2026-09-18_Lezione_QCED/
            audio.wav / pc_audio.wav / microphone.wav
            transcript.txt / .srt / .vtt
            session.json
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

APP_NAME = "LiveTranscriber"

_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_WIN_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def bundle_dir() -> Path:
    """Directory holding read-only resources (assets, bundled models).

    Under PyInstaller this is the extraction/one-dir root; in development it is
    the project root.
    """
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]


def app_data_dir() -> Path:
    """Root for models, logs, config and cache."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    return Path(base) / APP_NAME if base else Path.home() / f".{APP_NAME.lower()}"


def models_dir() -> Path:
    return app_data_dir() / "models"


def logs_dir() -> Path:
    return app_data_dir() / "logs"


def cache_dir() -> Path:
    return app_data_dir() / "cache"


def config_file() -> Path:
    return app_data_dir() / "config.json"


def default_recordings_dir() -> Path:
    """Default parent directory for session folders.

    Documents is used rather than LOCALAPPDATA because recordings are user
    content the user is expected to open, move and share.
    """
    docs = Path.home() / "Documents"
    if not docs.exists():
        docs = Path.home()
    return docs / APP_NAME / "Recordings"


def ensure_app_dirs() -> None:
    """Create every directory the application writes to. Idempotent."""
    for d in (app_data_dir(), models_dir(), logs_dir(), cache_dir()):
        d.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str, fallback: str = "Session") -> str:
    """Make ``name`` safe to use as a single Windows path component."""
    cleaned = _INVALID_FS_CHARS.sub("_", name).strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    # A title made entirely of separators sanitizes to something like "___",
    # which is a technically valid but useless folder name. Require at least one
    # alphanumeric character before accepting it.
    if not cleaned or not any(ch.isalnum() for ch in cleaned):
        return fallback
    if cleaned.upper().split(".")[0] in _RESERVED_WIN_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned[:80].strip(" .") or fallback


def unique_session_dir(
    parent: Path,
    title: str = "",
    when: datetime | None = None,
) -> Path:
    """Return a session directory path that does not exist yet.

    Never overwrites an existing recording: on collision a numeric suffix is
    appended (``..._2``, ``..._3``). The directory itself is not created here.
    """
    when = when or datetime.now()
    stamp = when.strftime("%Y-%m-%d_%H-%M")
    base = f"{stamp}_{sanitize_name(title)}" if title.strip() else stamp

    candidate = parent / base
    counter = 2
    while candidate.exists():
        candidate = parent / f"{base}_{counter}"
        counter += 1
    return candidate
