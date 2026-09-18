"""Periodic save of a running session (spec §14).

A four-hour lecture must survive a power cut, a driver fault or Task Manager.
Three things are written on a timer:

* ``transcript.json`` — every confirmed segment so far;
* ``session.json``    — metadata, with ``completed: false`` while it runs;
* the WAV headers      — flushed through the recorder.

``completed`` is the flag recovery keys on. It stays false for the whole
recording and is only set when the session stops cleanly, so a session that
died is exactly a session whose record still says false.

Two things keep the cost near zero. The transcript exposes a revision counter,
so an unchanged transcript is not rewritten; and both files are written to a
temporary sibling and moved into place, so an interrupted save leaves the
previous good file rather than a truncated one.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from app.export.json_export import (
    TRANSCRIPT_FILENAME,
    SessionRecord,
    write_session,
)
from app.export.json_export import (
    export as export_json,
)
from app.sessions.transcript import Transcript, TranscriptStats

logger = logging.getLogger(__name__)

#: Worst case, a crash costs this many seconds of transcript.
DEFAULT_INTERVAL_S = 5.0


class Autosave:
    """Saves a session's transcript and metadata on a timer.

    Runs on its own daemon thread. Every save is wrapped: a failure is logged
    and the next one is attempted, because a full disk or a locked file must
    not take the recording down with it.
    """

    def __init__(
        self,
        directory: Path,
        transcript: Transcript,
        record: SessionRecord,
        interval_s: float = DEFAULT_INTERVAL_S,
        on_sync: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.transcript = transcript
        self.record = record
        self.interval_s = max(1.0, float(interval_s))
        self.on_sync = on_sync
        self.on_error = on_error

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_revision = -1
        self._reported_failure = False

        self.saves = 0
        self.failures = 0
        self.last_saved_at = 0.0

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        # Write immediately, so a crash seconds into a recording still leaves a
        # record that recovery can find.
        self.save(force=True)

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="autosave", daemon=True)
        self._thread.start()
        logger.info("Autosave started (every %.0fs) in %s", self.interval_s, self.directory)

    def stop(self, completed: bool = True) -> None:
        """Stop saving and write a final record.

        ``completed=True`` is what tells recovery this session ended properly.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=15)

        self.record.completed = completed
        self.save(force=True)
        logger.info(
            "Autosave stopped (%d saves, %d failures, completed=%s)",
            self.saves,
            self.failures,
            completed,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.save()

    # -- saving -----------------------------------------------------------

    def save(self, force: bool = False) -> bool:
        """Write the transcript and metadata. Returns True if anything changed."""
        with self._lock:
            revision = self.transcript.revision
            if not force and revision == self._last_revision:
                return False

            try:
                export_json(
                    self.transcript,
                    self.directory / TRANSCRIPT_FILENAME,
                    self.record.title,
                )
                self.record.stats = TranscriptStats.of(self.transcript).to_dict()
                write_session(self.directory, self.record)

                if self.on_sync is not None:
                    self.on_sync()

                self._last_revision = revision
                self.saves += 1
                self.last_saved_at = time.time()
                self._reported_failure = False
                return True
            except Exception as exc:
                self.failures += 1
                logger.exception("Autosave failed")
                # Report once per run of failures, not once per tick: a full
                # disk would otherwise produce a message every few seconds.
                if self.on_error is not None and not self._reported_failure:
                    self._reported_failure = True
                    try:
                        self.on_error(_friendly_save_error(exc))
                    except Exception:
                        logger.exception("Autosave error handler raised")
                return False

    def __enter__(self) -> Autosave:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop(completed=exc_info[0] is None)


def _friendly_save_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "space" in text or "errno 28" in text:
        return (
            "Spazio su disco insufficiente per salvare la trascrizione. "
            "La registrazione audio continua."
        )
    if "permission" in text or "denied" in text or "being used" in text:
        return (
            "Impossibile salvare la trascrizione: la cartella non è accessibile. "
            "La registrazione audio continua."
        )
    return "Impossibile salvare automaticamente la trascrizione."
