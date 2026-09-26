"""Whether the speed measurement draws the right conclusions."""

from __future__ import annotations

import pytest

from app.transcription.calibration import (
    STREAMING_LADDER,
    StreamingCost,
    Verdict,
    cap_to_measurements,
    lighter_model,
    suggest_model,
)


def cost(value: float, model: str = "small", sources: int = 1) -> StreamingCost:
    return StreamingCost(
        model_key=model,
        device="cpu",
        compute_type="int8",
        cost=value,
        audio_seconds=20.0,
        runs=10,
    )


# -- verdicts --------------------------------------------------------------


def test_a_fast_result_is_comfortable():
    assert cost(0.2).verdict is Verdict.COMFORTABLE
    assert cost(0.2).headroom == pytest.approx(5.0)


def test_a_result_that_leaves_no_room_for_a_second_source_is_tight():
    # 0.55 alone is fine; recording PC and microphone together doubles it.
    assert cost(0.55).verdict is Verdict.TIGHT


def test_a_result_slower_than_the_audio_is_too_slow():
    assert cost(1.4).verdict is Verdict.TOO_SLOW
    assert cost(1.4).headroom == pytest.approx(1 / 1.4)


def test_both_sources_double_the_cost():
    measured = cost(0.4)
    assert measured.verdict is Verdict.COMFORTABLE
    assert measured.verdict_for(sources=2) is Verdict.TIGHT


def test_a_zero_cost_measurement_is_not_trusted():
    """A measurement with no inference in it means the sample never reached
    Whisper — a broken VAD or a silent sample, not an infinitely fast PC."""
    assert cost(0.0).verdict is Verdict.UNKNOWN
    assert not cost(0.0).is_usable
    assert StreamingCost("small", "cpu", "int8", 0.3, 20.0, runs=0).verdict is Verdict.UNKNOWN


# -- what to do about them -------------------------------------------------


def test_a_too_slow_model_is_stepped_down():
    assert suggest_model(cost(1.4, model="small")) == "base"
    assert suggest_model(cost(1.4, model="base")) == "tiny"


def test_the_smallest_model_has_nowhere_to_go():
    assert suggest_model(cost(1.4, model="tiny")) == "tiny"


def test_plenty_of_headroom_offers_a_better_model():
    assert suggest_model(cost(0.1, model="base")) == "small"


def test_a_comfortable_fit_is_left_alone():
    assert suggest_model(cost(0.45, model="small")) == "small"


def test_an_untrustworthy_measurement_changes_nothing():
    assert suggest_model(cost(0.0, model="small")) == "small"


def test_the_ladder_covers_the_catalogue_cheapest_first():
    from app.transcription import models

    assert set(STREAMING_LADDER) == {spec.key for spec in models.CATALOGUE}
    assert STREAMING_LADDER[0] == "tiny"


def test_lighter_model_walks_down_one_rung():
    assert lighter_model("medium") == "small"
    assert lighter_model("tiny") is None


# -- capping a preference by what was measured ------------------------------


def test_a_preference_with_no_measurements_is_left_alone():
    assert cap_to_measurements("small", lambda key: None) == "small"


def test_a_model_measured_too_slow_is_stepped_down():
    # Measured on this machine, CPU: small costs 1.03 per second of audio.
    measured = {"small": 1.03}
    assert cap_to_measurements("small", measured.get) == "base"


def test_the_descent_stops_at_the_first_unmeasured_rung():
    """An unmeasured model gets the benefit of the doubt.

    Assuming it is slow would be a guess, and guessing is what the measurement
    exists to replace — the runtime monitor covers the case where it is wrong.
    """
    measured = {"medium": 2.0, "small": 1.03}
    assert cap_to_measurements("medium", measured.get) == "base"


def test_the_descent_stops_at_the_bottom():
    measured = dict.fromkeys(STREAMING_LADDER, 9.0)
    assert cap_to_measurements("large-v3", measured.get) == "tiny"


def test_two_sources_can_cap_a_model_that_is_fine_alone():
    measured = {"base": 0.5}
    assert cap_to_measurements("base", measured.get, sources=1) == "base"
    assert cap_to_measurements("base", measured.get, sources=2) == "tiny"
