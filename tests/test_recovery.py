"""Autosave and crash recovery.

The central scenario is simulated literally: write a session as if it were
running, never mark it complete, and check that a later startup finds it, can
rescue the text, and touches nothing else.
"""

from __future__ import annotations

import time

from app.export.json_export import (
    SESSION_FILENAME,
    TRANSCRIPT_FILENAME,
    SessionRecord,
    read_session,
    write_session,
)
from app.sessions.autosave import Autosave
from app.sessions.recovery import (
    LIVE_GRACE_S,
    dismiss,
    find_incomplete,
    recover,
)
from app.sessions.transcript import Source, Transcript


def make_transcript(segments: int = 3) -> Transcript:
    t = Transcript()
    for i in range(segments):
        t.add(Source.PC, i * 5.0, i * 5.0 + 4.0, f"Segmento numero {i}.")
    return t


def crashed_session(directory, title="Lezione interrotta", transcript=None, age_s=60.0):
    """Write a session that looks like it died mid-recording."""
    directory.mkdir(parents=True, exist_ok=True)
    transcript = transcript if transcript is not None else make_transcript()

    record = SessionRecord(
        title=title,
        directory=str(directory),
        started="2026-09-18T10:00:00",
        mode="pc",
        model="small",
        completed=False,          # the flag recovery keys on
    )
    autosave = Autosave(directory, transcript, record)
    autosave.save(force=True)

    # Audio the crash left behind.
    (directory / "audio.wav").write_bytes(b"RIFF" + b"\x00" * 2000)

    # Age the record past the live-session grace period.
    old = time.time() - age_s
    for name in (SESSION_FILENAME, TRANSCRIPT_FILENAME):
        path = directory / name
        if path.exists():
            import os

            os.utime(path, (old, old))
    return directory


# -- autosave --------------------------------------------------------------

def test_autosave_writes_both_files(tmp_path):
    transcript = make_transcript()
    record = SessionRecord(title="Prova", directory=str(tmp_path))

    autosave = Autosave(tmp_path, transcript, record)
    assert autosave.save(force=True)

    assert (tmp_path / TRANSCRIPT_FILENAME).exists()
    assert (tmp_path / SESSION_FILENAME).exists()


def test_autosave_marks_a_running_session_incomplete(tmp_path):
    """While recording, the record must say the session has not finished."""
    autosave = Autosave(tmp_path, make_transcript(), SessionRecord(directory=str(tmp_path)))
    autosave.save(force=True)

    assert read_session(tmp_path).completed is False


def test_stopping_marks_it_complete(tmp_path):
    autosave = Autosave(tmp_path, make_transcript(), SessionRecord(directory=str(tmp_path)))
    autosave.start()
    autosave.stop(completed=True)

    assert read_session(tmp_path).completed is True


def test_unchanged_transcript_is_not_rewritten(tmp_path):
    """The revision counter keeps idle ticks free."""
    transcript = make_transcript()
    autosave = Autosave(tmp_path, transcript, SessionRecord(directory=str(tmp_path)))

    assert autosave.save(force=True)
    assert not autosave.save(), "nothing changed; it should have skipped"

    transcript.add(Source.PC, 100.0, 102.0, "Nuovo testo.")
    assert autosave.save(), "a changed transcript must be written"


def test_autosave_leaves_no_partial_files(tmp_path):
    autosave = Autosave(tmp_path, make_transcript(), SessionRecord(directory=str(tmp_path)))
    autosave.save(force=True)
    assert not list(tmp_path.glob("*.part"))


def test_autosave_survives_a_failing_write(tmp_path, monkeypatch):
    """A full disk must not take the recording down."""
    messages = []
    autosave = Autosave(
        tmp_path,
        make_transcript(),
        SessionRecord(directory=str(tmp_path)),
        on_error=messages.append,
    )

    def explode(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("app.sessions.autosave.export_json", explode)

    assert autosave.save(force=True) is False
    assert autosave.failures == 1
    assert messages and "Spazio" in messages[0]


def test_repeated_failures_report_only_once(tmp_path, monkeypatch):
    messages = []
    autosave = Autosave(
        tmp_path,
        make_transcript(),
        SessionRecord(directory=str(tmp_path)),
        on_error=messages.append,
    )
    monkeypatch.setattr(
        "app.sessions.autosave.export_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError(28, "No space left on device")),
    )

    for _ in range(5):
        autosave.save(force=True)

    assert len(messages) == 1, "the user must not be told every few seconds"


def test_autosave_thread_starts_and_stops(tmp_path):
    autosave = Autosave(
        tmp_path, make_transcript(), SessionRecord(directory=str(tmp_path)), interval_s=1.0
    )
    autosave.start()
    assert autosave._thread is not None
    autosave.stop()
    assert autosave._thread is None
    assert autosave.saves >= 1


def test_autosave_as_a_context_manager(tmp_path):
    with Autosave(tmp_path, make_transcript(), SessionRecord(directory=str(tmp_path))):
        pass
    assert read_session(tmp_path).completed is True


# -- finding interrupted sessions ------------------------------------------

def test_a_crashed_session_is_found(tmp_path):
    crashed_session(tmp_path / "2026-09-18_10-00_Lezione")
    found = find_incomplete(tmp_path)

    assert len(found) == 1
    assert found[0].title == "Lezione interrotta"
    assert found[0].segments == 3
    assert found[0].has_audio


def test_a_completed_session_is_not_offered(tmp_path):
    directory = crashed_session(tmp_path / "finita")
    record = read_session(directory)
    record.completed = True
    write_session(directory, record)

    assert find_incomplete(tmp_path) == []


def test_a_session_still_running_is_left_alone(tmp_path):
    """A record saved seconds ago probably belongs to a live window."""
    crashed_session(tmp_path / "in-corso", age_s=0.0)
    assert find_incomplete(tmp_path, grace_s=LIVE_GRACE_S) == []


def test_an_empty_failed_start_is_not_offered(tmp_path):
    """Nothing recorded, nothing transcribed: nothing worth rescuing."""
    directory = tmp_path / "vuota"
    directory.mkdir()
    write_session(directory, SessionRecord(directory=str(directory), completed=False))

    import os

    old = time.time() - 300
    os.utime(directory / SESSION_FILENAME, (old, old))

    assert find_incomplete(tmp_path) == []


def test_audio_without_a_transcript_is_still_offered(tmp_path):
    """The audio alone is worth rescuing."""
    directory = tmp_path / "solo-audio"
    directory.mkdir()
    write_session(directory, SessionRecord(directory=str(directory), completed=False))
    (directory / "audio.wav").write_bytes(b"RIFF" + b"\x00" * 5000)

    import os

    old = time.time() - 300
    os.utime(directory / SESSION_FILENAME, (old, old))

    found = find_incomplete(tmp_path)
    assert len(found) == 1
    assert found[0].has_audio
    assert not found[0].has_transcript


def test_several_crashes_are_listed_newest_first(tmp_path):
    crashed_session(tmp_path / "vecchia", age_s=3000)
    crashed_session(tmp_path / "recente", age_s=60)

    found = find_incomplete(tmp_path)
    assert [f.directory.name for f in found] == ["recente", "vecchia"]


def test_unreadable_folders_are_skipped(tmp_path):
    (tmp_path / "spazzatura").mkdir()
    (tmp_path / "spazzatura" / SESSION_FILENAME).write_text("{corrotto", encoding="utf-8")
    crashed_session(tmp_path / "buona")

    found = find_incomplete(tmp_path)
    assert len(found) == 1
    assert found[0].directory.name == "buona"


def test_a_missing_root_is_not_an_error(tmp_path):
    assert find_incomplete(tmp_path / "non-esiste") == []


def test_loose_files_in_the_root_are_ignored(tmp_path):
    (tmp_path / "note.txt").write_text("nothing to see", encoding="utf-8")
    assert find_incomplete(tmp_path) == []


# -- recovering ------------------------------------------------------------

def test_recovering_restores_the_text_and_writes_exports(tmp_path):
    directory = crashed_session(tmp_path / "da-recuperare")
    session = find_incomplete(tmp_path)[0]

    transcript = recover(session)

    assert len(transcript) == 3
    assert "Segmento numero 0." in transcript.text()
    for name in ("transcript.txt", "transcript.srt", "transcript.vtt"):
        assert (directory / name).exists(), f"{name} was not written"


def test_a_recovered_session_is_not_offered_again(tmp_path):
    crashed_session(tmp_path / "una-volta")
    recover(find_incomplete(tmp_path)[0])

    assert find_incomplete(tmp_path) == []


def test_recovery_is_marked_in_the_record(tmp_path):
    directory = crashed_session(tmp_path / "segnata")
    recover(find_incomplete(tmp_path)[0])

    record = read_session(directory)
    assert record.completed is True
    assert record.recovered is True


def test_recovery_never_touches_another_recording(tmp_path):
    """Spec §14: previous recordings are never overwritten."""
    other = tmp_path / "registrazione-precedente"
    other.mkdir()
    precious = other / "audio.wav"
    precious.write_bytes(b"PRECIOUS AUDIO")
    write_session(other, SessionRecord(directory=str(other), completed=True))

    crashed_session(tmp_path / "interrotta")
    recover(find_incomplete(tmp_path)[0])

    assert precious.read_bytes() == b"PRECIOUS AUDIO"


def test_recovering_audio_without_a_transcript(tmp_path):
    directory = tmp_path / "muta"
    directory.mkdir()
    write_session(directory, SessionRecord(directory=str(directory), completed=False))
    (directory / "audio.wav").write_bytes(b"RIFF" + b"\x00" * 5000)

    import os

    old = time.time() - 300
    os.utime(directory / SESSION_FILENAME, (old, old))

    session = find_incomplete(tmp_path)[0]
    transcript = recover(session)

    assert len(transcript) == 0
    assert not (directory / "transcript.txt").exists(), "nothing to export"
    assert read_session(directory).completed is True


def test_dismissing_keeps_the_files(tmp_path):
    """"Ignora" means stop asking, not throw the recording away."""
    directory = crashed_session(tmp_path / "ignorata")
    session = find_incomplete(tmp_path)[0]

    dismiss(session)

    assert find_incomplete(tmp_path) == []
    assert (directory / "audio.wav").exists()
    assert (directory / TRANSCRIPT_FILENAME).exists()


def test_summary_describes_what_would_be_recovered(tmp_path):
    crashed_session(tmp_path / "descritta")
    summary = find_incomplete(tmp_path)[0].summary()

    assert "Lezione interrotta" in summary
    assert "segmenti" in summary


# -- the whole scenario ----------------------------------------------------

def test_crash_during_recording_loses_at_most_one_interval(tmp_path):
    """Write like a live session, kill it, and check what survived."""
    directory = tmp_path / "simulazione"
    transcript = Transcript()
    record = SessionRecord(title="Lezione lunga", directory=str(directory))
    autosave = Autosave(directory, transcript, record, interval_s=1.0)
    autosave.start()

    try:
        for i in range(6):
            transcript.add(Source.PC, i * 10.0, i * 10.0 + 9.0, f"Frase numero {i}.")
            autosave.save()
    finally:
        # No stop() call: this is the crash.
        autosave._stop.set()

    import os

    old = time.time() - 300
    os.utime(directory / SESSION_FILENAME, (old, old))

    found = find_incomplete(tmp_path)
    assert len(found) == 1, "an interrupted session must be found"

    rescued = recover(found[0])
    assert len(rescued) == 6, "every saved segment must survive"
    assert "Frase numero 5." in rescued.text()
