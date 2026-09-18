"""Structured JSON export, and the session record on disk.

Two different files, both JSON, with different jobs:

``transcript.json``
    The full transcript, ungrouped. Every confirmed segment exactly as the
    pipeline produced it, with the five fields the spec fixes (§12). This is the
    archival form: TXT, SRT and VTT are all derivable from it, and none of them
    is derivable from the others.

``session.json``
    What the recording *was* — when it ran, which devices and model, which audio
    files exist, and crucially whether it finished. Recovery reads this file to
    find sessions that were interrupted (§14).

Both are written atomically. A crash mid-write must not replace a good file with
half a file, which for the archival transcript would mean losing the recording's
only complete record.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.sessions.transcript import Transcript, TranscriptStats

logger = logging.getLogger(__name__)

EXTENSION = ".json"

TRANSCRIPT_FILENAME = "transcript.json"
SESSION_FILENAME = "session.json"

#: Bumped when the on-disk shape changes incompatibly.
SCHEMA_VERSION = 1


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON atomically: full file or previous file, never a partial one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON, returning None for anything unusable.

    A truncated or corrupt file is treated as absent rather than raised: this is
    called during recovery, where the whole point is coping with a bad shutdown.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %r", path, exc)
        return None


# --------------------------------------------------------------------------
# transcript.json
# --------------------------------------------------------------------------


def render(transcript: Transcript, title: str = "", extra: dict | None = None) -> dict:
    stats = TranscriptStats.of(transcript)
    return {
        "schema": SCHEMA_VERSION,
        "title": title,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "stats": stats.to_dict(),
        "segments": transcript.to_dicts(),
        **(extra or {}),
    }


def export(
    transcript: Transcript, path: Path, title: str = "", extra: dict | None = None
) -> Path:
    return write_json(path, render(transcript, title, extra))


def load_transcript(path: Path) -> Transcript | None:
    """Rebuild a transcript from ``transcript.json``."""
    payload = read_json(path)
    if not payload:
        return None
    transcript = Transcript()
    transcript.load(payload.get("segments", []))
    return transcript


# --------------------------------------------------------------------------
# session.json
# --------------------------------------------------------------------------


@dataclass
class SessionRecord:
    """Metadata describing one recording."""

    title: str = ""
    directory: str = ""
    started: str = ""
    finished: str = ""
    duration: float = 0.0

    mode: str = ""
    model: str = ""
    language: str = ""
    device: str = ""
    compute_type: str = ""
    loopback_device: str = ""
    microphone_device: str = ""

    audio_files: dict[str, str] = field(default_factory=dict)
    exports: dict[str, str] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    #: False until the session is stopped cleanly. Recovery looks at this.
    completed: bool = False
    #: Set when a session was rescued after a crash.
    recovered: bool = False
    app_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, **vars(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionRecord:
        known = {f for f in vars(cls()) if not f.startswith("_")}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def path(self) -> Path:
        return Path(self.directory)

    @property
    def started_at(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self.started)
        except (TypeError, ValueError):
            return None

    @property
    def display_name(self) -> str:
        return self.title or (self.path.name if self.directory else "Registrazione")


def write_session(directory: Path, record: SessionRecord) -> Path:
    return write_json(Path(directory) / SESSION_FILENAME, record.to_dict())


def read_session(directory: Path) -> SessionRecord | None:
    payload = read_json(Path(directory) / SESSION_FILENAME)
    if not payload:
        return None
    try:
        record = SessionRecord.from_dict(payload)
    except TypeError as exc:
        logger.warning("Unusable session record in %s: %r", directory, exc)
        return None
    if not record.directory:
        record.directory = str(directory)
    return record
