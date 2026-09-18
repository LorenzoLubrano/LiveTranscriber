"""Export formats: valid files, correct grouping, nothing lost."""

from __future__ import annotations

import json
import re

import pytest

from app.export import (
    DEFAULT_FORMATS,
    export_all,
    get_format,
    json_export,
    list_formats,
    srt,
    txt,
    vtt,
)
from app.export.base import (
    ExportError,
    GroupingRules,
    clock_timestamp,
    group_segments,
    srt_timestamp,
    vtt_timestamp,
    wrap_cue,
)
from app.export.json_export import load_transcript
from app.sessions.transcript import Source, Transcript


@pytest.fixture
def transcript() -> Transcript:
    """A short two-speaker exchange, fragmented as streaming produces it."""
    t = Transcript()
    t.add(Source.PC, 12.31, 15.0, "Consideriamo adesso l'Hamiltoniana del sistema,")
    t.add(Source.PC, 15.0, 18.14, "dove il termine di interazione dipende dal tempo.")
    t.add(Source.MIC, 19.0, 22.4, "Quindi siamo nel limite adiabatico?")
    t.add(Source.PC, 23.0, 27.9, "Esatto, purche' la variazione sia lenta.")
    return t


# -- timestamps ------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "00:00:00,000"),
        (12.31, "00:00:12,310"),
        (61.5, "00:01:01,500"),
        (3661.001, "01:01:01,001"),
        (-3, "00:00:00,000"),
    ],
)
def test_srt_timestamps(seconds, expected):
    assert srt_timestamp(seconds) == expected


def test_vtt_uses_a_full_stop_not_a_comma():
    """The two formats are strict about this and reject the other's form."""
    assert vtt_timestamp(12.31) == "00:00:12.310"
    assert "," not in vtt_timestamp(12.31)


def test_clock_timestamps():
    assert clock_timestamp(3661) == "01:01:01"


def test_millisecond_rounding_does_not_overflow():
    assert srt_timestamp(0.9999) == "00:00:01,000"
    assert srt_timestamp(59.9996) == "00:01:00,000"


# -- grouping --------------------------------------------------------------

def test_fragments_from_one_speaker_merge(transcript):
    """Streaming confirms fragments; a document needs sentences."""
    groups = group_segments(transcript.segments, GroupingRules.for_paragraphs())

    assert len(groups) == 3, "the two PC fragments should have merged"
    assert groups[0].text.startswith("Consideriamo")
    assert groups[0].text.endswith("dipende dal tempo.")


def test_a_change_of_speaker_always_breaks(transcript):
    groups = group_segments(transcript.segments, GroupingRules.for_paragraphs())
    assert [g.source for g in groups] == [Source.PC, Source.MIC, Source.PC]


def test_a_long_pause_breaks_a_group():
    t = Transcript()
    t.add(Source.PC, 0.0, 2.0, "Prima parte.")
    t.add(Source.PC, 30.0, 32.0, "Molto dopo.")
    assert len(group_segments(t.segments, GroupingRules(max_gap=2.0))) == 2


def test_grouping_respects_a_duration_cap():
    t = Transcript()
    for i in range(20):
        t.add(Source.PC, i * 2.0, i * 2.0 + 2.0, f"frammento {i}")
    groups = group_segments(t.segments, GroupingRules(max_gap=5.0, max_duration=10.0))
    assert len(groups) > 1
    assert all(g.duration <= 12.0 for g in groups)


def test_subtitle_rules_produce_short_cues(transcript):
    groups = group_segments(transcript.segments, GroupingRules.for_subtitles())
    for group in groups:
        assert len(group.text) <= 84 + 40, f"cue too long: {group.text!r}"
        assert group.duration <= 8.0


def test_grouping_loses_no_words(transcript):
    """Regrouping must never drop text."""
    original = " ".join(s.text for s in transcript.segments).split()
    regrouped = " ".join(g.text for g in group_segments(transcript.segments)).split()
    assert regrouped == original


def test_grouping_of_nothing():
    assert group_segments([]) == []


def test_grouping_skips_blank_segments():
    t = Transcript()
    t.add(Source.PC, 0.0, 1.0, "vero")
    t._segments.append(type(t._segments[0])(Source.PC, 1.0, 2.0, "   "))
    assert len(group_segments(t.segments)) == 1


# -- cue wrapping ----------------------------------------------------------

def test_short_cues_are_not_wrapped():
    assert "\n" not in wrap_cue("Testo breve")


def test_long_cues_wrap_to_two_lines():
    text = "Consideriamo adesso l'Hamiltoniana del sistema dove il termine dipende"
    wrapped = wrap_cue(text, width=42)
    assert wrapped.count("\n") <= 1
    assert wrapped.replace("\n", " ") == text


def test_wrapping_never_loses_words():
    text = " ".join(f"parola{i}" for i in range(30))
    assert wrap_cue(text).replace("\n", " ").split() == text.split()


def test_wrapping_of_nothing():
    assert wrap_cue("") == ""


# -- TXT -------------------------------------------------------------------

def test_txt_matches_the_spec_layout(transcript):
    """Spec §13 fixes this shape: header line, then the prose."""
    rendered = txt.render(transcript)
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] \[PC\]\n", rendered)
    assert "[MIC]" in rendered


def test_txt_can_omit_timestamps(transcript):
    rendered = txt.render(transcript, include_timestamps=False)
    assert "[PC]" in rendered
    assert not re.search(r"\[\d{2}:\d{2}:\d{2}\]", rendered)


def test_txt_includes_a_title(transcript):
    assert "Lezione QCED" in txt.render(transcript, title="Lezione QCED")


def test_txt_of_an_empty_transcript():
    assert txt.render(Transcript()).strip() == ""


def test_txt_written_to_disk(tmp_path, transcript):
    path = txt.export(transcript, tmp_path / "t.txt", title="Prova")
    assert path.exists()
    assert "Consideriamo" in path.read_text(encoding="utf-8")


# -- SRT -------------------------------------------------------------------

def test_srt_structure_is_valid(transcript):
    blocks = [b for b in srt.render(transcript).split("\n\n") if b.strip()]
    assert blocks

    for index, block in enumerate(blocks, start=1):
        lines = block.strip().split("\n")
        assert lines[0] == str(index), "indices must be sequential from 1"
        assert re.fullmatch(
            r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", lines[1]
        ), f"bad timing line: {lines[1]!r}"
        assert len(lines) >= 3, "a cue needs text"


def test_srt_cues_never_overlap(transcript):
    """PC and microphone are transcribed independently and can overlap.

    Overlapping cues are invalid SRT, so they are pushed apart on export.
    """
    times = re.findall(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})",
        srt.render(transcript),
    )

    def to_seconds(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    previous_end = -1.0
    for h1, m1, s1, ms1, h2, m2, s2, ms2 in times:
        start = to_seconds(h1, m1, s1, ms1)
        end = to_seconds(h2, m2, s2, ms2)
        assert start >= previous_end, "cues overlap"
        assert end > start, "cue has no duration"
        previous_end = end


def test_overlapping_sources_are_separated(tmp_path):
    t = Transcript()
    t.add(Source.PC, 0.0, 10.0, "Il relatore parla a lungo.")
    t.add(Source.MIC, 5.0, 8.0, "Interruzione dal microfono.")

    rendered = srt.render(t)
    starts = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}) -->", rendered)
    assert starts == sorted(starts)


def test_srt_of_an_empty_transcript():
    assert srt.render(Transcript()) == ""


def test_srt_written_to_disk(tmp_path, transcript):
    path = srt.export(transcript, tmp_path / "t.srt")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("1\n")


# -- VTT -------------------------------------------------------------------

def test_vtt_starts_with_the_required_header(transcript):
    assert vtt.render(transcript).startswith("WEBVTT")


def test_vtt_uses_voice_spans(transcript):
    rendered = vtt.render(transcript)
    assert "<v PC>" in rendered
    assert "<v MIC>" in rendered


def test_vtt_timings_use_full_stops(transcript):
    rendered = vtt.render(transcript)
    assert re.search(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", rendered)
    timings = [ln for ln in rendered.split("\n") if "-->" in ln]
    assert timings and all("," not in ln for ln in timings)


def test_vtt_title_goes_in_a_note(transcript):
    rendered = vtt.render(transcript, title="Lezione QCED")
    assert "NOTE Lezione QCED" in rendered


def test_empty_vtt_still_has_its_header():
    assert vtt.render(Transcript()).startswith("WEBVTT")


# -- JSON ------------------------------------------------------------------

def test_json_round_trips(tmp_path, transcript):
    path = json_export.export(transcript, tmp_path / "t.json", title="Prova")
    restored = load_transcript(path)

    assert restored is not None
    assert len(restored) == len(transcript)
    assert [s.text for s in restored.segments] == [s.text for s in transcript.segments]
    assert [s.source for s in restored.segments] == [s.source for s in transcript.segments]


def test_json_keeps_full_fidelity(tmp_path, transcript):
    """JSON is the archival form: the others derive from it, not vice versa."""
    path = json_export.export(transcript, tmp_path / "t.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["segments"]) == len(transcript.segments), "must not group"
    assert payload["stats"]["segments"] == len(transcript)
    for segment in payload["segments"]:
        assert set(segment) == {"source", "start", "end", "text", "confirmed"}


def test_json_writes_are_atomic(tmp_path, transcript):
    path = tmp_path / "t.json"
    json_export.export(transcript, path)
    assert not list(tmp_path.glob("*.part")), "temporary file left behind"


def test_reading_corrupt_json_returns_none(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert json_export.read_json(bad) is None
    assert load_transcript(bad) is None


def test_reading_a_missing_file_returns_none(tmp_path):
    assert json_export.read_json(tmp_path / "nope.json") is None


# -- registry --------------------------------------------------------------

def test_every_spec_format_is_available():
    """Spec §13 requires these four."""
    assert {f.key for f in list_formats()} >= {"txt", "srt", "vtt", "json"}


def test_formats_declare_matching_extensions():
    for fmt in list_formats():
        assert fmt.extension.startswith(".")
        assert fmt.label and fmt.filter_string.endswith(f"(*{fmt.extension})")


def test_unknown_format_raises_a_user_facing_error():
    with pytest.raises(ExportError) as info:
        get_format("pdf")
    assert "pdf" in info.value.user_message


def test_export_all_writes_every_default_format(tmp_path, transcript):
    written = export_all(transcript, tmp_path, title="Lezione QCED")

    assert set(written) == set(DEFAULT_FORMATS)
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0


def test_export_all_survives_one_format_failing(tmp_path, transcript, monkeypatch):
    """Losing a transcript to a subtitle bug would be absurd."""
    def explode(*args, **kwargs):
        raise RuntimeError("srt is broken")

    monkeypatch.setattr("app.export.srt.export", explode)
    monkeypatch.setattr(
        "app.export.FORMATS",
        tuple(
            f if f.key != "srt" else type(f)(**{**vars(f), "writer": explode})
            for f in list_formats()
        ),
    )
    written = export_all(transcript, tmp_path)
    assert "txt" in written
    assert "json" in written


def test_export_all_creates_the_directory(tmp_path, transcript):
    target = tmp_path / "deep" / "nested"
    written = export_all(transcript, target)
    assert target.is_dir()
    assert written
