"""Streaming pipeline against a real model.

The unit tests in ``test_streaming.py`` prove the reconciliation algorithm with
synthetic hypotheses. These prove the whole pipeline on real inference: that
streaming output matches what a single transcription of the same audio would
produce, that silence stays silent, and that memory does not grow.

Skipped when no model is downloaded.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from app.audio.resampler import TARGET_SAMPLE_RATE, resample_array
from app.transcription import models
from app.transcription.engine import (
    TranscriptionEngine,
    TranscriptionSettings,
)
from app.transcription.hardware import Accelerator
from app.transcription.streaming import StreamingSettings, StreamingTranscriber

pytestmark = pytest.mark.hardware

SAMPLE_TEXT = (
    "Consideriamo adesso l'Hamiltoniana del sistema, "
    "dove il termine di interazione dipende dal tempo."
)


def _synthesize(text: str, destination: Path) -> bool:
    script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$v = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -eq 'it-IT' }} |
     Select-Object -First 1
if ($null -eq $v) {{ $s.Dispose(); exit 2 }}
$s.SelectVoice($v.VoiceInfo.Name)
$s.SetOutputToWaveFile('{destination}')
$s.Speak(@'
{text}
'@)
$s.SetOutputToNull()
$s.Dispose()
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=90,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and destination.exists()


def _load(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    return resample_array(np.frombuffer(raw, dtype=np.int16), rate, TARGET_SAMPLE_RATE, channels)


def _smallest_model() -> str:
    installed = models.installed_models()
    if not installed:
        pytest.skip("no Whisper model downloaded")
    return installed[0].key


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / "sample.wav"
        if not _synthesize(SAMPLE_TEXT, destination):
            pytest.skip("no Italian speech voice installed")
        return _load(destination)


@pytest.fixture(scope="module")
def engine():
    engine = TranscriptionEngine(_smallest_model(), accelerator=Accelerator.AUTO)
    engine.load()
    yield engine
    engine.unload()


def feed_all(transcriber: StreamingTranscriber, audio: np.ndarray, block: float = 0.25) -> None:
    """Push audio through in capture-sized blocks, as fast as possible."""
    step = int(block * TARGET_SAMPLE_RATE)
    for offset in range(0, audio.size, step):
        transcriber.feed(audio[offset : offset + step])
        if transcriber.should_run():
            transcriber.process()


def make(engine, **kwargs) -> StreamingTranscriber:
    return StreamingTranscriber(
        engine,
        TranscriptionSettings(language="it"),
        StreamingSettings(**kwargs),
    )


# -- correctness -----------------------------------------------------------

def test_streaming_produces_text(engine, speech):
    transcriber = make(engine)
    feed_all(transcriber, speech)
    transcriber.finish()

    text = transcriber.confirmed_text
    assert text, "streaming produced no text at all"
    assert "consideriamo" in text.lower()


def test_streaming_loses_nothing_against_one_shot(engine, speech):
    """The property that matters: windowing must not drop words.

    Compared by word overlap rather than exactly, because the two paths see
    different amounts of context and may punctuate differently.
    """
    transcriber = make(engine)
    feed_all(transcriber, speech)
    transcriber.finish()
    streamed = set(transcriber.confirmed_text.lower().split())

    one_shot = engine.transcribe(speech, TranscriptionSettings(language="it"))
    expected = set(one_shot.text.lower().split())

    missing = expected - streamed
    assert len(missing) <= len(expected) * 0.2, f"streaming lost: {missing}"


def test_no_duplicated_phrases(engine, speech):
    """Re-transcribing overlapping audio must not repeat text."""
    transcriber = make(engine, chunk_duration=1.5)
    feed_all(transcriber, speech)
    transcriber.finish()

    tokens = transcriber.confirmed_text.lower().split()
    for size in (3, 4, 5):
        for i in range(len(tokens) - size * 2 + 1):
            assert tokens[i : i + size] != tokens[i + size : i + size * 2], (
                f"duplicated run: {' '.join(tokens[i:i + size])}"
            )


def test_words_are_confirmed_in_order(engine, speech):
    transcriber = make(engine)
    feed_all(transcriber, speech)
    transcriber.finish()

    words = transcriber.confirmed_words
    starts = [w.start for w in words]
    assert starts == sorted(starts), "confirmed words are out of order"


def test_finish_flushes_the_tail(engine, speech):
    """The last sentence must not be lost when recording stops."""
    transcriber = make(engine, chunk_duration=10.0)  # too long to ever commit
    feed_all(transcriber, speech)
    before = transcriber.confirmed_text

    transcriber.finish()
    after = transcriber.confirmed_text

    assert len(after) > len(before), "finish() produced nothing"


# -- silence ---------------------------------------------------------------

def test_silence_produces_no_text_and_no_inference(engine):
    """Spec §5 and §31: never transcribe silence, never hallucinate on it."""
    transcriber = make(engine)
    feed_all(transcriber, np.zeros(int(12 * TARGET_SAMPLE_RATE), dtype=np.float32))
    transcriber.finish()

    assert transcriber.confirmed_text == ""
    assert transcriber.runs == 0, "Whisper was called on pure silence"
    assert transcriber.skipped_silent_runs > 0


def test_quiet_noise_does_not_become_words(engine):
    """Room tone must not be transcribed."""
    rng = np.random.default_rng(1234)
    noise = rng.normal(0, 0.002, int(8 * TARGET_SAMPLE_RATE)).astype(np.float32)

    transcriber = make(engine)
    feed_all(transcriber, noise)
    transcriber.finish()

    assert transcriber.confirmed_text == "", (
        f"hallucinated on noise: {transcriber.confirmed_text!r}"
    )


def test_speech_after_silence_is_still_captured(engine, speech):
    """A leading pause must not swallow the speech that follows it."""
    padded = np.concatenate(
        [np.zeros(int(6 * TARGET_SAMPLE_RATE), dtype=np.float32), speech]
    )
    transcriber = make(engine)
    feed_all(transcriber, padded)
    transcriber.finish()

    assert transcriber.confirmed_text, "speech after a silence was lost"


# -- memory ----------------------------------------------------------------

def test_buffer_stays_bounded_over_a_long_stream(engine, speech):
    """Spec §17: a multi-hour recording must not grow memory.

    Speech and silence are alternated so both the commit path and the
    silence-drop path are exercised.
    """
    silence = np.zeros(int(2 * TARGET_SAMPLE_RATE), dtype=np.float32)
    settings = StreamingSettings()
    transcriber = make(engine)

    peak = 0.0
    for _ in range(6):
        for block in (speech, silence):
            feed_all(transcriber, block)
            peak = max(peak, transcriber.buffer_duration)

    transcriber.finish()
    assert peak <= settings.max_buffer_duration + 5.0, (
        f"buffer grew to {peak:.1f}s"
    )
    assert transcriber.buffer_duration == 0.0, "finish() left audio behind"


def test_long_silence_does_not_accumulate(engine):
    """Ten minutes of silence must not fill memory."""
    transcriber = make(engine)
    block = np.zeros(int(10 * TARGET_SAMPLE_RATE), dtype=np.float32)
    for _ in range(60):
        feed_all(transcriber, block)

    assert transcriber.buffer_duration < 10.0, (
        f"silence accumulated to {transcriber.buffer_duration:.1f}s"
    )


def test_reset_clears_state(engine, speech):
    transcriber = make(engine)
    feed_all(transcriber, speech)
    transcriber.finish()
    assert transcriber.confirmed_text

    transcriber.reset()
    assert transcriber.confirmed_text == ""
    assert transcriber.buffer_duration == 0.0
