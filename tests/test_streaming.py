"""Streaming reconciliation.

These tests drive the LocalAgreement algorithm directly with synthetic
hypotheses, because that is the only way to reproduce the exact failure modes
the spec calls out (§5) — cut words, repetitions, duplicated phrases — on
demand rather than hoping a real recording happens to trigger them.
"""

from __future__ import annotations

import pytest

from app.transcription.engine import Segment, Word
from app.transcription.streaming import (
    HypothesisBuffer,
    StreamingSettings,
    StreamingUpdate,
    merge_words,
    normalise_word,
    words_to_segments,
)


def words(*specs: tuple[str, float, float]) -> list[Word]:
    """Build a word list from (text, start, end) triples."""
    return [Word(start, end, f" {text}") for text, start, end in specs]


def sentence(text: str, start: float = 0.0, step: float = 0.5) -> list[Word]:
    """Turn a sentence into evenly spaced words."""
    out = []
    for i, token in enumerate(text.split()):
        out.append(Word(start + i * step, start + (i + 1) * step, f" {token}"))
    return out


def texts(word_list: list[Word]) -> list[str]:
    return [w.text.strip() for w in word_list]


# -- normalisation ---------------------------------------------------------

def test_normalisation_ignores_punctuation_and_case():
    """Whisper changes punctuation between runs; that is not disagreement."""
    assert normalise_word(" Sistema.") == normalise_word("sistema")
    assert normalise_word("Ciao!") == normalise_word("ciao")
    assert normalise_word(" IL ") == "il"


def test_normalisation_keeps_distinct_words_distinct():
    assert normalise_word("sistema") != normalise_word("sistemi")


# -- agreement -------------------------------------------------------------

def test_nothing_is_confirmed_from_a_single_run():
    """One hypothesis is not evidence; two must agree."""
    buffer = HypothesisBuffer(agreement_runs=2)
    assert buffer.insert(sentence("consideriamo adesso il sistema")) == []


def test_agreeing_runs_confirm_the_common_prefix():
    buffer = HypothesisBuffer(agreement_runs=2)
    buffer.insert(sentence("consideriamo adesso il"))
    confirmed = buffer.insert(sentence("consideriamo adesso il sistema"))

    assert texts(confirmed) == ["consideriamo", "adesso", "il"]


def test_divergence_stops_confirmation_at_that_point():
    buffer = HypothesisBuffer(agreement_runs=2)
    buffer.insert(sentence("consideriamo adesso il gatto"))
    confirmed = buffer.insert(sentence("consideriamo adesso il sistema"))

    assert texts(confirmed) == ["consideriamo", "adesso", "il"]
    assert "gatto" not in " ".join(texts(confirmed))


def test_a_truncated_word_is_never_confirmed():
    """The cut-word failure: a word at the buffer edge must wait.

    Run 1 sees "sis" because the audio ended mid-word; run 2 sees "sistema".
    They disagree, so the truncation is never emitted.
    """
    buffer = HypothesisBuffer(agreement_runs=2)
    buffer.insert(sentence("il sis"))
    confirmed = buffer.insert(sentence("il sistema completo"))

    assert texts(confirmed) == ["il"]
    assert "sis" not in texts(confirmed)


def test_confirmed_text_is_never_emitted_twice():
    """The duplication failure: re-transcribing the same audio must not repeat."""
    buffer = HypothesisBuffer(agreement_runs=2)
    full = sentence("consideriamo adesso il sistema")

    buffer.insert(full)
    first = buffer.insert(full)
    second = buffer.insert(full)   # same hypothesis a third time
    third = buffer.insert(full)

    all_confirmed = texts(first) + texts(second) + texts(third)
    assert all_confirmed == ["consideriamo", "adesso", "il", "sistema"]
    assert len(all_confirmed) == len(set(all_confirmed))


def test_confirmation_advances_across_many_runs():
    buffer = HypothesisBuffer(agreement_runs=2)
    collected: list[str] = []
    growing = [
        "consideriamo adesso",
        "consideriamo adesso il",
        "consideriamo adesso il sistema",
        "consideriamo adesso il sistema fisico",
        "consideriamo adesso il sistema fisico completo",
    ]
    for text in growing:
        collected.extend(texts(buffer.insert(sentence(text))))
    collected.extend(texts(buffer.flush()))

    assert collected == ["consideriamo", "adesso", "il", "sistema", "fisico", "completo"]


def test_committed_boundary_moves_forward_only():
    buffer = HypothesisBuffer(agreement_runs=2)
    buffer.insert(sentence("uno due tre"))
    buffer.insert(sentence("uno due tre"))
    first = buffer.committed_until

    buffer.insert(sentence("uno due tre quattro"))
    assert buffer.committed_until >= first


def test_words_already_committed_are_filtered_out():
    buffer = HypothesisBuffer(agreement_runs=2)
    buffer.insert(sentence("uno due"))
    buffer.insert(sentence("uno due"))

    # A later run re-reports the same early words; none may be re-confirmed.
    confirmed = buffer.insert(sentence("uno due tre"))
    assert "uno" not in texts(confirmed)
    assert "due" not in texts(confirmed)


def test_three_way_agreement_is_stricter():
    buffer = HypothesisBuffer(agreement_runs=3)
    assert buffer.insert(sentence("uno due")) == []
    assert buffer.insert(sentence("uno due")) == []
    confirmed = buffer.insert(sentence("uno due tre"))
    assert texts(confirmed) == ["uno", "due"]


def test_agreement_of_one_confirms_immediately():
    buffer = HypothesisBuffer(agreement_runs=1)
    assert texts(buffer.insert(sentence("uno due"))) == ["uno", "due"]


def test_flush_confirms_everything_pending():
    """Recording stopped: nothing more can revise the tail."""
    buffer = HypothesisBuffer(agreement_runs=2)
    buffer.insert(sentence("uno due tre"))
    assert texts(buffer.flush()) == ["uno", "due", "tre"]
    assert buffer.pending == []


def test_flush_on_an_empty_buffer():
    assert HypothesisBuffer().flush() == []


def test_pending_exposes_the_provisional_text():
    buffer = HypothesisBuffer(agreement_runs=2)
    buffer.insert(sentence("uno due tre"))
    assert texts(buffer.pending) == ["uno", "due", "tre"]


def test_reset_clears_state():
    buffer = HypothesisBuffer(agreement_runs=2)
    buffer.insert(sentence("uno due"))
    buffer.insert(sentence("uno due"))
    buffer.reset()
    assert buffer.committed_until == 0.0
    assert buffer.pending == []


def test_empty_hypothesis_confirms_nothing():
    buffer = HypothesisBuffer(agreement_runs=2)
    buffer.insert(sentence("uno due"))
    assert buffer.insert([]) == []


def test_repetition_loop_is_not_confirmed():
    """A stuck model repeating a word must not have the repeats confirmed.

    Run 2 disagrees with run 1 from the second word on, so only the genuine
    first word survives.
    """
    buffer = HypothesisBuffer(agreement_runs=2)
    buffer.insert(sentence("grazie grazie grazie grazie"))
    confirmed = buffer.insert(sentence("grazie e adesso continuiamo"))
    assert texts(confirmed) == ["grazie"]


# -- settings --------------------------------------------------------------

def test_settings_are_clamped_to_sane_values():
    s = StreamingSettings(chunk_duration=0.01, agreement_runs=99).validate()
    assert s.chunk_duration >= 0.5
    assert s.agreement_runs <= 4


def test_buffer_cap_cannot_be_smaller_than_two_chunks():
    s = StreamingSettings(chunk_duration=10.0, max_buffer_duration=5.0).validate()
    assert s.max_buffer_duration >= 20.0


# -- update ----------------------------------------------------------------

def test_update_renders_confirmed_and_provisional_separately():
    update = StreamingUpdate(
        confirmed=sentence("testo confermato"),
        provisional=sentence("testo provvisorio", start=10.0),
    )
    assert update.confirmed_text == "testo confermato"
    assert update.provisional_text == "testo provvisorio"
    assert update.has_content


def test_empty_update_has_no_content():
    assert not StreamingUpdate().has_content


# -- grouping for export ---------------------------------------------------

def test_words_join_back_into_text():
    assert merge_words(sentence("uno due tre")) == "uno due tre"
    assert merge_words([]) == ""


def test_grouping_splits_on_a_long_pause():
    early = words(("uno", 0.0, 0.5), ("due", 0.5, 1.0))
    late = words(("tre", 5.0, 5.5), ("quattro", 5.5, 6.0))
    segments = words_to_segments(early + late, max_gap=1.0)

    assert len(segments) == 2
    assert segments[0].text == "uno due"
    assert segments[1].text == "tre quattro"


def test_grouping_splits_a_monologue_that_never_pauses():
    long_run = [Word(i * 0.4, (i + 1) * 0.4, f" parola{i}") for i in range(200)]
    segments = words_to_segments(long_run, max_duration=15.0)

    assert len(segments) > 1
    assert all(s.duration <= 20.0 for s in segments), "segments must stay readable"


def test_grouping_keeps_timings_and_words():
    segments = words_to_segments(sentence("uno due tre"))
    assert segments[0].start == 0.0
    assert segments[0].end == pytest.approx(1.5)
    assert len(segments[0].words) == 3


def test_grouping_of_nothing():
    assert words_to_segments([]) == []


def test_every_word_survives_grouping():
    """Grouping must never lose or duplicate a word."""
    source = sentence("alpha bravo charlie delta echo foxtrot")
    regrouped = [w for seg in words_to_segments(source) for w in seg.words]
    assert texts(regrouped) == texts(source)


# -- segments --------------------------------------------------------------

def test_segment_shifting_moves_words_too():
    segment = Segment(0.0, 1.0, "ciao", words=(Word(0.0, 1.0, " ciao"),))
    shifted = segment.shifted(10.0)
    assert shifted.start == 10.0
    assert shifted.words[0].start == 10.0
