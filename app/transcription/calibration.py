"""Measuring what this particular PC can transcribe live.

The app has to choose a model before it knows anything about the machine it is
running on, and the honest answer is that nobody can predict it. Core count is a
poor proxy: two six-core laptops differ by a factor of two on the same model,
depending on clocks, thermal budget and whether AVX-512 is fused off. Published
real-time factors are worse than useless here, because they measure transcribing
a *file* — streaming re-transcribes a growing buffer, so it costs roughly five
times as much (see :mod:`app.sessions.keepup`).

So instead of a table of guesses, this measures. A short passage of synthesised
speech is pushed through the real streaming pipeline as fast as the machine will
go, and the result is one number: **seconds of inference per second of audio**.
Below 0.5 the machine is comfortable; at 0.85 it is about to start losing audio.

Windows' own speech synthesiser provides the sample, so nothing has to be
downloaded, bundled or licensed, and the measurement works offline.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from app.audio.resampler import TARGET_SAMPLE_RATE
from app.transcription.hardware import Accelerator

logger = logging.getLogger(__name__)

#: Cost at or below which one source runs comfortably, leaving room for a
#: second source (PC + microphone doubles the work) and for the rest of the PC.
COMFORTABLE_COST = 0.5

#: At or above this the machine is too slow for live use: latency climbs and the
#: capture ring buffer starts overwriting audio that was never saved.
TOO_SLOW_COST = 0.85

#: Ceiling for a model the app chooses *by itself*, which needs more margin than
#: one the user picked knowingly. A 20 s measurement is taken at whatever clock
#: speed the CPU happens to be boosting to; a lecture is an hour long, and a
#: laptop does not hold its boost clocks for an hour. Measured evidence for the
#: gap: `small` on this CPU read 0.65 on a cold machine and 0.96 across a hot
#: sequential batch, on the same code.
AUTO_CEILING = 0.7

#: Below this there is room for the next model up, which typically costs two to
#: three times as much.
STEP_UP_COST = 0.25

#: Seconds of audio to measure over. Long enough for the working buffer to reach
#: its live size, short enough that nobody minds waiting. Verified stable: the
#: same model measured 0.74 / 0.72 / 0.71 / 0.72 / 0.69 over 8, 12, 20, 30 and
#: 45 seconds, so 20 is not a special value — it is simply enough.
DEFAULT_TARGET_SECONDS = 20.0

#: Models from cheapest to most expensive **when streaming**, which is neither
#: their file size nor their parameter count. Measured here, cost per second of
#: audio (CPU int8 / GPU float16, medians of repeated runs):
#:
#:     tiny 0.18 / 0.07    base 0.55 / -       small 0.97 / 0.13
#:     turbo 1.38 / 0.20   medium 1.82 / 0.45  large-v3 1.81 / 0.36
#:
#: Turbo sits *below* medium, not above it: it is large-v3's encoder with a
#: four-layer decoder, and even though streaming re-runs the encoder on every
#: pass, the decoder saving still wins — 1.3x cheaper than medium on CPU and 2.2x
#: cheaper on the GPU, at close to large-v3 accuracy. This project assumed the
#: opposite until it measured.
#:
#: Medium and large-v3 come out within noise of each other (1.82 vs 1.81 on CPU),
#: so medium is placed first as the smaller download.
#:
#: Only used to pick a neighbour to *offer*. The measurement is the authority.
STREAMING_LADDER = ("tiny", "base", "small", "turbo", "medium", "large-v3")


class CalibrationError(RuntimeError):
    """Calibration could not run. Carries a message safe to show a user."""

    def __init__(self, user_message: str, technical: str = "") -> None:
        super().__init__(technical or user_message)
        self.user_message = user_message


class Verdict(StrEnum):
    """What a measurement means for live transcription."""

    COMFORTABLE = "comfortable"
    TIGHT = "tight"
    TOO_SLOW = "too_slow"
    #: The measurement cannot be trusted — no inference actually happened.
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return {
            Verdict.COMFORTABLE: "va bene",
            Verdict.TIGHT: "al limite",
            Verdict.TOO_SLOW: "troppo lento",
            Verdict.UNKNOWN: "non misurabile",
        }[self]


@dataclass(frozen=True)
class StreamingCost:
    """What one model costs this machine, live."""

    model_key: str
    device: str
    compute_type: str
    #: Seconds of inference per second of audio. 1.0 means "exactly real time",
    #: which in practice already means falling behind.
    cost: float
    audio_seconds: float
    runs: int

    @property
    def is_usable(self) -> bool:
        """False when nothing was actually inferred.

        A zero cost is not an infinitely fast PC: it means the sample never
        reached Whisper, because the VAD rejected it or synthesis produced
        silence. An earlier version of the soak test made exactly this mistake
        and reported a perfect result while measuring nothing.
        """
        return self.runs > 0 and self.cost > 0.0

    @property
    def headroom(self) -> float:
        """How many times faster than real time. 1.0 is the edge of the cliff."""
        return (1.0 / self.cost) if self.cost > 0 else 0.0

    def verdict_for(self, sources: int = 1) -> Verdict:
        """Verdict when ``sources`` streams are recorded at once.

        Recording the PC and the microphone together runs two engines, so the
        cost doubles. A model that is comfortable alone can be too slow for both.
        """
        if not self.is_usable:
            return Verdict.UNKNOWN
        effective = self.cost * max(1, sources)
        if effective >= TOO_SLOW_COST:
            return Verdict.TOO_SLOW
        if effective > COMFORTABLE_COST:
            return Verdict.TIGHT
        return Verdict.COMFORTABLE

    @property
    def verdict(self) -> Verdict:
        return self.verdict_for(1)

    @property
    def summary(self) -> str:
        """One line for the settings screen."""
        if not self.is_usable:
            return "Velocità non misurata."
        return (
            f"{self.model_key} su {self.device}: {self.headroom:.1f}× "
            f"più veloce del tempo reale ({self.verdict.label})"
        )


# --------------------------------------------------------------------------
# What to do about a measurement
# --------------------------------------------------------------------------


def lighter_model(model_key: str) -> str | None:
    """The next cheaper model, or None if there is none."""
    try:
        index = STREAMING_LADDER.index(model_key)
    except ValueError:
        return None
    return STREAMING_LADDER[index - 1] if index > 0 else None


def heavier_model(model_key: str) -> str | None:
    """The next more expensive model, or None if there is none."""
    try:
        index = STREAMING_LADDER.index(model_key)
    except ValueError:
        return None
    return STREAMING_LADDER[index + 1] if index + 1 < len(STREAMING_LADDER) else None


def cap_to_measurements(
    preferred: str,
    lookup: Callable[[str], float | None],
    sources: int = 1,
    ceiling: float = AUTO_CEILING,
) -> str:
    """Step ``preferred`` down while past measurements say it cannot keep up.

    ``lookup`` returns the measured cost for a model on this machine, or None if
    it was never measured. An unmeasured model stops the descent rather than
    being assumed slow: guessing is what this whole module exists to avoid.

    The ceiling is deliberately below :data:`TOO_SLOW_COST`. This picks a model
    on the user's behalf, so it should leave the margin that a 20 s measurement
    on a boosting laptop does not prove is there.
    """
    current = preferred
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        cost = lookup(current)
        if cost is None or cost * max(1, sources) < ceiling:
            return current
        lighter = lighter_model(current)
        if lighter is None:
            return current
        logger.info(
            "%s measured at %.2f per second of audio; stepping down to %s",
            current,
            cost,
            lighter,
        )
        current = lighter
    return current


def suggest_model(measured: StreamingCost, sources: int = 1) -> str:
    """The model to use, given what the measured one cost.

    One rung at a time, deliberately: each step changes the cost by a factor of
    two or three, and the next measurement confirms the new choice rather than
    this function predicting it.
    """
    verdict = measured.verdict_for(sources)
    if verdict is Verdict.UNKNOWN:
        return measured.model_key
    if verdict is Verdict.TOO_SLOW:
        return lighter_model(measured.model_key) or measured.model_key
    if measured.cost * max(1, sources) <= STEP_UP_COST:
        return heavier_model(measured.model_key) or measured.model_key
    return measured.model_key


# --------------------------------------------------------------------------
# The sample
# --------------------------------------------------------------------------

_SAMPLE_TEXT = (
    "Consideriamo adesso il caso in cui il termine di interazione dipende dal "
    "tempo. Nel limite adiabatico la variazione è sufficientemente lenta, "
    "quindi il sistema resta nello stato fondamentale. Questo vale finché il "
    "divario di energia non si chiude."
)

_SYNTH_SCRIPT = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$v = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -eq 'it-IT' }} |
     Select-Object -First 1
if ($v) {{ $s.SelectVoice($v.VoiceInfo.Name) }}
$s.SetOutputToWaveFile('{destination}')
$s.Speak(@'
{text}
'@)
$s.SetOutputToNull()
$s.Dispose()
"""


def _synthesise(destination: Path) -> None:
    """Speak the sample into a WAV with Windows' own synthesiser."""
    script = _SYNTH_SCRIPT.format(destination=destination, text=_SAMPLE_TEXT)
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CalibrationError(
            "Non è stato possibile generare la voce di prova su questo PC.",
            repr(exc),
        ) from exc

    if result.returncode != 0 or not destination.exists():
        raise CalibrationError(
            "La sintesi vocale di Windows non è disponibile, quindi la velocità "
            "non può essere misurata. Puoi comunque scegliere il modello a mano.",
            result.stderr.decode("utf-8", "replace")[:400],
        )


def _read_wav(path: Path) -> np.ndarray:
    from app.audio.resampler import resample_array

    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    return resample_array(samples, rate, TARGET_SAMPLE_RATE, channels)


def speech_sample(min_seconds: float = DEFAULT_TARGET_SECONDS) -> np.ndarray:
    """16 kHz mono speech, at least ``min_seconds`` long.

    Synthesised once and cached: the measurement is run again whenever the user
    changes model, and re-running the synthesiser each time is slower than the
    measurement itself.

    Pauses are kept between repetitions so the silence path and the commit path
    are exercised too — a sample of unbroken speech would measure a load the
    real pipeline never sees.
    """
    from app.utils.paths import cache_dir

    cached = cache_dir() / "speech-sample.wav"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "speech.wav"
            _synthesise(raw)
            audio = _read_wav(raw)
        pcm = np.clip(audio, -1.0, 1.0)
        with wave.open(str(cached), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(TARGET_SAMPLE_RATE)
            handle.writeframes((pcm * 32767.0).astype(np.int16).tobytes())
    else:
        audio = _read_wav(cached)

    if audio.size == 0:
        raise CalibrationError("La voce di prova è risultata vuota.")

    pause = np.zeros(int(1.2 * TARGET_SAMPLE_RATE), dtype=np.float32)
    one_loop = np.concatenate((audio, pause))
    repeats = max(1, int(np.ceil(min_seconds * TARGET_SAMPLE_RATE / one_loop.size)))
    return np.tile(one_loop, repeats)


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


def streaming_chunk(transcriber) -> float:
    """How much new audio the transcriber waits for between passes."""
    return transcriber.streaming.chunk_duration


def measure_streaming_cost(
    model_key: str,
    accelerator: Accelerator = Accelerator.AUTO,
    target_seconds: float = DEFAULT_TARGET_SECONDS,
    language: str = "it",
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> StreamingCost:
    """Run the real streaming pipeline flat out and report what it cost.

    Uses the same :class:`~app.transcription.streaming.StreamingTranscriber`,
    VAD and settings as a recording, with the pacing removed — a benchmark that
    measured anything else would not answer the question that matters.

    Blocking and slow (seconds to a minute). Never call it on the GUI thread.
    """
    from app.transcription.engine import TranscriptionEngine, TranscriptionSettings
    from app.transcription.streaming import StreamingSettings, StreamingTranscriber

    sample = speech_sample(target_seconds)
    engine = TranscriptionEngine(model_key, accelerator=accelerator)
    choice = engine.load()

    transcriber = StreamingTranscriber(
        engine,
        TranscriptionSettings(language=language),
        StreamingSettings(),
    )

    # A virtual clock, not a fixed feed rate. Audio keeps arriving while a pass
    # runs, so on a machine that is behind, each pass starts with more audio
    # waiting than the last — the buffer grows, passes get more expensive, and
    # the load diverges. Feeding a fixed 2 s between passes hides exactly that:
    # it measured 0.71 for `small` on this CPU where a real paced recording
    # measured 1.00. Advancing the clock by the time each pass actually took
    # reproduces the real dynamics and still costs no wall-clock waiting.
    total_samples = int(target_seconds * TARGET_SAMPLE_RATE)
    audio_clock = 0.0     # seconds of audio that have "arrived"
    fed = 0
    try:
        while fed < total_samples:
            if cancelled is not None and cancelled():
                break

            # Nothing runs before enough new audio exists, exactly as live.
            audio_clock = max(
                audio_clock, transcriber.stream_position + streaming_chunk(transcriber)
            )
            want = min(int(audio_clock * TARGET_SAMPLE_RATE), total_samples)
            if want > fed:
                chunk = sample[fed:want] if want <= sample.size else sample[fed:]
                transcriber.feed(chunk)
                fed += chunk.size

            started = time.monotonic()
            transcriber.process()
            # The audio that arrived while that pass was running.
            audio_clock += time.monotonic() - started

            if progress is not None:
                progress(fed / total_samples)
        transcriber.finish()
    finally:
        engine.unload()

    audio_seconds = fed / TARGET_SAMPLE_RATE
    cost = (
        transcriber.total_inference_time / audio_seconds if audio_seconds > 0 else 0.0
    )
    measured = StreamingCost(
        model_key=model_key,
        device=choice.device,
        compute_type=choice.compute_type,
        cost=cost,
        audio_seconds=audio_seconds,
        runs=transcriber.runs,
    )
    logger.info(
        "Calibration: %s on %s/%s cost %.2f per second of audio over %.1fs (%d runs)",
        model_key,
        choice.device,
        choice.compute_type,
        cost,
        audio_seconds,
        transcriber.runs,
    )
    return measured
