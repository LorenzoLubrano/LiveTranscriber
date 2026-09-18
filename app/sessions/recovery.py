"""Finding and rescuing interrupted recordings (spec §14).

A session that ended properly has ``completed: true`` in its ``session.json``.
Anything else — a power cut, a crash, a forced close — leaves that flag false,
which is precisely the signal recovery looks for.

What recovery does **not** do is overwrite anything. The rescued session keeps
its own directory and its own files; exports are written next to the audio that
produced them. A previous recording is never touched, which is the point of the
"do not overwrite" requirement: the user has already lost a session once, and
losing a second one to the rescue would be worse than not rescuing at all.

A session whose process is still running must not be offered for recovery, so
sessions saved within the last few seconds are left alone.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from app.export import export_all
from app.export.json_export import (
    TRANSCRIPT_FILENAME,
    SessionRecord,
    load_transcript,
    read_session,
    write_session,
)
from app.sessions.transcript import Transcript
from app.utils.paths import default_recordings_dir

logger = logging.getLogger(__name__)

#: A record saved more recently than this probably belongs to a live session.
LIVE_GRACE_S = 30.0

#: Sessions older than this are not offered; the user has moved on.
MAX_AGE_DAYS = 30


@dataclass
class RecoverableSession:
    """An interrupted recording that can still be salvaged."""

    record: SessionRecord
    directory: Path
    transcript_path: Path | None
    audio_paths: list[Path]
    segments: int
    duration: float
    modified: float

    @property
    def title(self) -> str:
        return self.record.display_name

    @property
    def has_transcript(self) -> bool:
        return self.segments > 0

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_paths)

    @property
    def audio_size_mb(self) -> float:
        return sum(p.stat().st_size for p in self.audio_paths if p.exists()) / 1_048_576

    @property
    def audio_size_label(self) -> str:
        """Readable size. A short recording is kilobytes, and "0 MB" reads as
        "nothing here" when there is in fact audio to rescue."""
        megabytes = self.audio_size_mb
        if megabytes >= 1.0:
            return f"{megabytes:.0f} MB"
        return f"{megabytes * 1024:.0f} KB"

    def summary(self) -> str:
        """One line describing what would be recovered."""
        from app.sessions.transcript import format_timestamp

        parts = [self.title]
        if self.duration:
            parts.append(format_timestamp(self.duration, always_hours=True))
        if self.segments:
            parts.append(f"{self.segments} segmenti")
        if self.audio_paths:
            parts.append(f"{self.audio_size_label} di audio")
        return "  ·  ".join(parts)


def _audio_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.wav") if p.stat().st_size > 44)


def find_incomplete(
    root: Path | None = None, grace_s: float = LIVE_GRACE_S
) -> list[RecoverableSession]:
    """Every interrupted session under ``root``, newest first.

    Anything that cannot be read is skipped rather than raised: this runs at
    startup, and a single unreadable folder must not stop the app from opening.
    """
    root = Path(root) if root else default_recordings_dir()
    if not root.is_dir():
        return []

    now = time.time()
    cutoff = now - MAX_AGE_DAYS * 86400
    found: list[RecoverableSession] = []

    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            record = read_session(directory)
            if record is None or record.completed:
                continue

            session_file = directory / "session.json"
            modified = session_file.stat().st_mtime
            if modified < cutoff:
                continue
            if now - modified < grace_s:
                # Probably still recording in another window.
                logger.debug("Skipping %s: saved %.0fs ago", directory.name, now - modified)
                continue

            transcript_path = directory / TRANSCRIPT_FILENAME
            segments = int(record.stats.get("segments", 0) or 0)
            duration = float(record.stats.get("duration", 0.0) or 0.0)
            audio = _audio_in(directory)

            if not segments and not audio:
                # Nothing worth rescuing: an empty folder from a failed start.
                continue

            found.append(
                RecoverableSession(
                    record=record,
                    directory=directory,
                    transcript_path=transcript_path if transcript_path.exists() else None,
                    audio_paths=audio,
                    segments=segments,
                    duration=duration,
                    modified=modified,
                )
            )
        except OSError as exc:
            logger.warning("Could not inspect %s: %r", directory, exc)

    found.sort(key=lambda s: s.modified, reverse=True)
    if found:
        logger.info("Found %d interrupted session(s)", len(found))
    return found


def recover(session: RecoverableSession) -> Transcript:
    """Rescue a session: load its transcript and write the exports.

    Writes into the session's own directory, alongside the audio it belongs
    with. Nothing outside that directory is touched.
    """
    # `... or Transcript()` would work here only by accident: Transcript defines
    # __bool__ as "has segments", so a successfully loaded but empty transcript
    # would be silently swapped for a different object. Harmless today, and the
    # same shape that once left the GUI watching an object nothing wrote to.
    loaded = (
        load_transcript(session.transcript_path) if session.transcript_path else None
    )
    transcript = loaded if loaded is not None else Transcript()

    if len(transcript):
        exports = export_all(
            transcript, session.directory, title=session.record.title
        )
        session.record.exports = {k: str(v) for k, v in exports.items()}
        logger.info(
            "Recovered %s: %d segments, %d exports",
            session.directory.name,
            len(transcript),
            len(exports),
        )
    else:
        logger.info("Recovered %s: audio only, no transcript", session.directory.name)

    session.record.completed = True
    session.record.recovered = True
    write_session(session.directory, session.record)
    return transcript


def dismiss(session: RecoverableSession) -> None:
    """Stop offering this session, without deleting anything.

    The recording stays on disk exactly as it is — the user asked not to be
    asked again, not to throw the audio away.
    """
    session.record.completed = True
    write_session(session.directory, session.record)
    logger.info("Dismissed interrupted session %s", session.directory.name)


def dismiss_all(sessions: list[RecoverableSession]) -> None:
    for session in sessions:
        dismiss(session)
