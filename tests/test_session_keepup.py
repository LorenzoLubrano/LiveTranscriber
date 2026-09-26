"""Whether the session reacts to a machine that cannot keep up.

Built on fakes rather than a real model and sound card: what is being tested is
the reaction, and a test that needed a GPU would never run on the machines this
code exists to protect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from app.audio.capture import CaptureStats
from app.sessions.session import (
    CHUNK_WIDEN_COOLDOWN_S,
    MAX_CHUNK_DURATION_S,
    SessionCallbacks,
    _SourcePipeline,
)
from app.sessions.transcript import Source, Transcript
from app.transcription.streaming import StreamingSettings


@dataclass
class FakeStream:
    sample_rate: int = 16000
    channels: int = 1
    level: float = 0.0
    stats: CaptureStats = field(default_factory=CaptureStats)


@dataclass
class FakeTranscriber:
    streaming: StreamingSettings = field(default_factory=StreamingSettings)
    stream_position: float = 0.0
    runs: int = 0
    skipped_silent_runs: int = 0
    total_inference_time: float = 0.0


@pytest.fixture
def pipeline():
    warnings: list[str] = []
    stream = FakeStream()
    pipe = _SourcePipeline(
        source=Source.PC,
        stream=stream,
        ring=None,
        transcriber=FakeTranscriber(),
        write_audio=lambda audio: None,
        callbacks=SessionCallbacks(on_warning=warnings.append),
        transcript=Transcript(),
    )
    pipe.warnings = warnings  # type: ignore[attr-defined]
    return pipe


def run_passes(pipe, cost: float, seconds: float = 2.0, count: int = 20) -> None:
    for _ in range(count):
        pipe.transcriber.stream_position += seconds
        pipe._check_keeping_up(cost * seconds)


# -- the warning -----------------------------------------------------------


def test_a_machine_that_keeps_up_is_left_in_peace(pipeline):
    run_passes(pipeline, cost=0.2)
    assert pipeline.warnings == []
    assert pipeline.transcriber.streaming.chunk_duration == 2.0


def test_a_struggling_machine_is_warned_once_and_told_what_changed(pipeline):
    run_passes(pipeline, cost=0.9)
    assert len(pipeline.warnings) == 1
    message = pipeline.warnings[0]
    assert "al limite" in message
    assert "4s" in message  # the widened interval, named in the message

    run_passes(pipeline, cost=0.9)
    assert len(pipeline.warnings) == 1


def test_an_overrun_says_the_audio_is_still_being_recorded(pipeline):
    """The one thing a user needs to know in that moment.

    Text arriving late is an annoyance; believing the recording is lost would
    make someone stop and start over for nothing.
    """
    run_passes(pipeline, cost=1.5)
    assert len(pipeline.warnings) == 1
    assert "continua a essere registrato" in pipeline.warnings[0]


# -- the mitigation --------------------------------------------------------


def test_the_interval_is_widened_to_halve_the_work(pipeline):
    run_passes(pipeline, cost=0.9)
    assert pipeline.transcriber.streaming.chunk_duration == 4.0


def test_the_interval_is_not_widened_twice_in_a_row(pipeline):
    run_passes(pipeline, cost=0.9)
    assert pipeline.transcriber.streaming.chunk_duration == 4.0
    # Still struggling, but within the cooldown: one bad stretch must not walk
    # the interval to the cap in a few seconds.
    run_passes(pipeline, cost=2.0, count=5)
    assert pipeline.transcriber.streaming.chunk_duration == 4.0


def test_the_interval_stops_at_the_cap(pipeline):
    for _ in range(6):
        pipeline.transcriber.stream_position += CHUNK_WIDEN_COOLDOWN_S + 1
        pipeline._widen_chunk()
    assert pipeline.transcriber.streaming.chunk_duration == MAX_CHUNK_DURATION_S


# -- lost audio ------------------------------------------------------------


def test_dropped_capture_frames_are_reported_once(pipeline):
    pipeline._check_overflow()
    assert pipeline.warnings == []

    pipeline.stream.stats.frames_dropped = 16000 * 3
    pipeline._check_overflow()
    assert len(pipeline.warnings) == 1
    assert "3s" in pipeline.warnings[0]
    assert "persi" in pipeline.warnings[0]

    pipeline.stream.stats.frames_dropped = 16000 * 9
    pipeline._check_overflow()
    assert len(pipeline.warnings) == 1


def test_the_resampler_is_built_for_the_stream_it_is_given(pipeline):
    """Guards the fake against drifting away from the real constructor."""
    assert pipeline.resampler.process(np.zeros(1600, dtype=np.float32)) is not None
