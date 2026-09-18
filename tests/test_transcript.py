"""Transcript store: ordering, two sources, search, serialisation."""

from __future__ import annotations

import threading

import pytest

from app.sessions.transcript import (
    Source,
    Transcript,
    TranscriptSegment,
    TranscriptStats,
    format_timestamp,
    looks_complete,
)


@pytest.fixture
def transcript() -> Transcript:
    return Transcript()


# -- timestamps ------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "00:00"),
        (59, "00:59"),
        (60, "01:00"),
        (599, "09:59"),
        (3600, "01:00:00"),
        (3661, "01:01:01"),
        (-5, "00:00"),
    ],
)
def test_timestamp_formatting(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_timestamps_can_force_hours():
    assert format_timestamp(62, always_hours=True) == "00:01:02"


# -- basics ----------------------------------------------------------------

def test_new_transcript_is_empty(transcript):
    assert len(transcript) == 0
    assert not transcript
    assert transcript.duration == 0.0
    assert transcript.text() == ""


def test_add_stores_a_segment(transcript):
    segment = transcript.add(Source.PC, 0.0, 2.0, "Ciao")
    assert segment is not None
    assert len(transcript) == 1
    assert transcript.segments[0].text == "Ciao"
    assert transcript.segments[0].confirmed


def test_blank_text_is_rejected(transcript):
    assert transcript.add(Source.PC, 0.0, 1.0, "   ") is None
    assert len(transcript) == 0


def test_text_is_stripped(transcript):
    transcript.add(Source.PC, 0.0, 1.0, "  Ciao  ")
    assert transcript.segments[0].text == "Ciao"


def test_duration_follows_the_last_segment(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "uno")
    transcript.add(Source.MIC, 3.0, 7.5, "due")
    assert transcript.duration == 7.5


def test_word_count(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "uno due tre")
    transcript.add(Source.MIC, 2.0, 4.0, "quattro cinque")
    assert transcript.word_count == 5


# -- ordering --------------------------------------------------------------

def test_segments_stay_in_timeline_order(transcript):
    transcript.add(Source.PC, 10.0, 12.0, "terzo")
    transcript.add(Source.MIC, 0.0, 2.0, "primo")
    transcript.add(Source.PC, 5.0, 7.0, "secondo")

    assert [s.text for s in transcript.segments] == ["primo", "secondo", "terzo"]


def test_two_sources_interleave_by_time(transcript):
    """A conversation reads in time order, not grouped by source."""
    transcript.add(Source.PC, 0.0, 3.0, "Consideriamo l'Hamiltoniana")
    transcript.add(Source.MIC, 3.5, 5.0, "Siamo nel limite adiabatico?")
    transcript.add(Source.PC, 5.5, 8.0, "Esatto")

    assert [s.source for s in transcript.segments] == [Source.PC, Source.MIC, Source.PC]


def test_appending_in_order_is_the_common_path(transcript):
    for i in range(200):
        transcript.add(Source.PC, float(i), float(i + 1), f"parola{i}")
    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts)


# -- rendering -------------------------------------------------------------

def test_text_includes_source_labels(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "Dal computer")
    transcript.add(Source.MIC, 2.0, 4.0, "Dal microfono")

    rendered = transcript.text()
    assert "[PC] Dal computer" in rendered
    assert "[MIC] Dal microfono" in rendered


def test_timestamps_are_optional(transcript):
    transcript.add(Source.PC, 75.0, 78.0, "Ciao")
    assert "[00:01:15]" in transcript.text(include_timestamps=True)
    assert "[00:01:15]" not in transcript.text(include_timestamps=False)


def test_source_labels_are_optional(transcript):
    transcript.add(Source.PC, 0.0, 1.0, "Ciao")
    assert transcript.text(include_source=False) == "Ciao"


def test_source_labels_match_the_spec():
    assert Source.PC.label == "PC"
    assert Source.MIC.label == "MIC"


def test_sources_used(transcript):
    transcript.add(Source.PC, 0.0, 1.0, "a")
    assert transcript.sources_used() == {Source.PC}
    transcript.add(Source.MIC, 1.0, 2.0, "b")
    assert transcript.sources_used() == {Source.PC, Source.MIC}


# -- provisional -----------------------------------------------------------

def test_provisional_is_separate_from_confirmed(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "confermato")
    transcript.set_provisional(Source.PC, 2.0, 3.0, "provvisorio")

    assert len(transcript) == 1, "provisional must not count as confirmed"
    assert transcript.provisional[0].text == "provvisorio"
    assert not transcript.provisional[0].confirmed


def test_provisional_is_replaced_not_appended(transcript):
    transcript.set_provisional(Source.PC, 0.0, 1.0, "primo tentativo")
    transcript.set_provisional(Source.PC, 0.0, 2.0, "secondo tentativo")

    assert len(transcript.provisional) == 1
    assert transcript.provisional[0].text == "secondo tentativo"


def test_each_source_keeps_its_own_provisional(transcript):
    transcript.set_provisional(Source.PC, 0.0, 1.0, "dal pc")
    transcript.set_provisional(Source.MIC, 0.0, 1.0, "dal mic")
    assert len(transcript.provisional) == 2


def test_empty_provisional_clears_it(transcript):
    transcript.set_provisional(Source.PC, 0.0, 1.0, "qualcosa")
    transcript.set_provisional(Source.PC, 0.0, 1.0, "")
    assert transcript.provisional == []


def test_provisional_never_reaches_exported_text(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "confermato")
    transcript.set_provisional(Source.PC, 2.0, 3.0, "provvisorio")

    assert "provvisorio" not in transcript.text()
    assert all(d["confirmed"] for d in transcript.to_dicts())


def test_all_segments_can_include_provisional(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "confermato")
    transcript.set_provisional(Source.PC, 2.0, 3.0, "provvisorio")

    assert len(transcript.all_segments()) == 1
    assert len(transcript.all_segments(include_provisional=True)) == 2


# -- search ----------------------------------------------------------------

def test_search_finds_text(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "Consideriamo l'Hamiltoniana del sistema")
    hits = transcript.search("hamiltoniana")

    assert len(hits) == 1
    assert hits[0].index == 0
    assert hits[0].segment.text.startswith("Consideriamo")


def test_search_ignores_accents(transcript):
    """Typing "perche" must find "perché" on an Italian transcript."""
    transcript.add(Source.PC, 0.0, 2.0, "Perché la variazione è lenta")
    assert transcript.search("perche")
    assert transcript.search("PERCHE")


def test_search_can_be_case_sensitive(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "Sistema e sistema")
    assert len(transcript.search("sistema", case_sensitive=True)) == 1
    assert len(transcript.search("sistema", case_sensitive=False)) == 2


def test_search_reports_every_occurrence(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "test test test")
    assert len(transcript.search("test")) == 3


def test_search_returns_offsets_for_highlighting(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "abc target def")
    hit = transcript.search("target")[0]
    assert hit.segment.text[hit.start_offset : hit.end_offset] == "target"


def test_search_of_nothing(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "qualcosa")
    assert transcript.search("") == []
    assert transcript.search("   ") == []


def test_search_skips_provisional(transcript):
    transcript.set_provisional(Source.PC, 0.0, 1.0, "provvisorio")
    assert transcript.search("provvisorio") == []


def test_segment_at_a_point_in_time(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "primo")
    transcript.add(Source.PC, 5.0, 7.0, "secondo")

    assert transcript.segment_at(1.0).text == "primo"
    assert transcript.segment_at(6.0).text == "secondo"
    assert transcript.segment_at(3.0) is None


# -- serialisation ---------------------------------------------------------

def test_segment_dict_matches_the_spec_shape():
    """Spec §12 fixes these five field names."""
    segment = TranscriptSegment(Source.PC, 123.42, 128.14, "Consideriamo adesso il sistema.")
    data = segment.to_dict()

    assert data == {
        "source": "pc",
        "start": 123.42,
        "end": 128.14,
        "text": "Consideriamo adesso il sistema.",
        "confirmed": True,
    }


def test_segment_round_trip():
    original = TranscriptSegment(Source.MIC, 1.5, 3.25, "Ciao", confirmed=False)
    assert TranscriptSegment.from_dict(original.to_dict()) == original


def test_from_dict_tolerates_missing_fields():
    segment = TranscriptSegment.from_dict({})
    assert segment.source is Source.PC
    assert segment.text == ""


def test_from_dict_accepts_the_long_source_name():
    assert TranscriptSegment.from_dict({"source": "microphone"}).source is Source.MIC


def test_transcript_round_trip(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "primo")
    transcript.add(Source.MIC, 2.0, 4.0, "secondo")

    restored = Transcript()
    restored.load(transcript.to_dicts())

    assert [s.text for s in restored.segments] == ["primo", "secondo"]
    assert [s.source for s in restored.segments] == [Source.PC, Source.MIC]


def test_load_sorts_what_it_is_given(transcript):
    transcript.load([
        {"source": "pc", "start": 10.0, "end": 12.0, "text": "dopo"},
        {"source": "pc", "start": 1.0, "end": 2.0, "text": "prima"},
    ])
    assert [s.text for s in transcript.segments] == ["prima", "dopo"]


def test_load_replaces_existing_content(transcript):
    transcript.add(Source.PC, 0.0, 1.0, "vecchio")
    transcript.load([{"source": "pc", "start": 0.0, "end": 1.0, "text": "nuovo"}])
    assert [s.text for s in transcript.segments] == ["nuovo"]


# -- revision --------------------------------------------------------------

def test_revision_changes_on_every_write(transcript):
    """Autosave uses this to skip writing unchanged state."""
    start = transcript.revision
    transcript.add(Source.PC, 0.0, 1.0, "a")
    after_add = transcript.revision
    assert after_add > start

    transcript.set_provisional(Source.PC, 1.0, 2.0, "b")
    assert transcript.revision > after_add


def test_revision_is_stable_without_writes(transcript):
    transcript.add(Source.PC, 0.0, 1.0, "a")
    first = transcript.revision

    _ = transcript.segments   # reading must not count as a change
    _ = transcript.search("a")

    assert transcript.revision == first


# -- stats -----------------------------------------------------------------

def test_stats(transcript):
    transcript.add(Source.PC, 0.0, 2.0, "uno due tre")
    transcript.add(Source.MIC, 2.0, 5.0, "quattro")
    transcript.add(Source.PC, 5.0, 6.0, "cinque")

    stats = TranscriptStats.of(transcript)
    assert stats.segments == 3
    assert stats.words == 5
    assert stats.duration == 6.0
    assert stats.by_source == {"pc": 2, "mic": 1}
    assert stats.to_dict()["by_source"] == {"pc": 2, "mic": 1}


def test_stats_of_an_empty_transcript(transcript):
    stats = TranscriptStats.of(transcript)
    assert stats.segments == 0
    assert stats.duration == 0.0


# -- helpers ---------------------------------------------------------------

@pytest.mark.parametrize(
    "text,complete",
    [("Fine.", True), ("Davvero?", True), ("Ecco!", True), ("continua", False), ("", False)],
)
def test_sentence_completeness(text, complete):
    assert looks_complete(text) is complete


# -- threading -------------------------------------------------------------

def test_concurrent_writers_lose_nothing(transcript):
    """Two transcription workers write at once, one per source."""
    def writer(source: Source, base: float) -> None:
        for i in range(200):
            transcript.add(source, base + i, base + i + 0.5, f"{source.value}{i}")

    threads = [
        threading.Thread(target=writer, args=(Source.PC, 0.0)),
        threading.Thread(target=writer, args=(Source.MIC, 0.25)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert all(not t.is_alive() for t in threads)
    assert len(transcript) == 400


# -- the falsy-object trap -------------------------------------------------


def test_empty_transcript_is_falsy_but_not_none(transcript):
    """Transcript.__bool__ means "has segments", which is a trap for callers.

    `x or Transcript()` silently discards a caller's empty transcript. The
    session used to do exactly that, so the GUI watched an object nothing ever
    wrote to while recording worked perfectly. Anything accepting a Transcript
    must test `is not None`.
    """
    assert not transcript
    assert transcript is not None

    replaced = transcript or Transcript()
    assert replaced is not transcript, "this is the trap being documented"

    kept = transcript if transcript is not None else Transcript()
    assert kept is transcript


def test_session_keeps_the_transcript_it_is_given():
    """The regression that produced an empty GUI during a working recording."""
    from app.sessions.session import RecordingSession, SessionConfig

    shared = Transcript()
    session = RecordingSession(SessionConfig(), transcript=shared)
    assert session.transcript is shared

    shared.add(Source.PC, 0.0, 1.0, "scritto dalla pipeline")
    assert len(session.transcript) == 1


def test_session_still_creates_one_when_given_none():
    from app.sessions.session import RecordingSession, SessionConfig

    session = RecordingSession(SessionConfig())
    assert isinstance(session.transcript, Transcript)
