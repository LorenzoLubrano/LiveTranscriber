"""Voice activity detection.

Two jobs in the streaming pipeline:

1. **Keep silence away from Whisper.** Whisper hallucinates on silence — it will
   confidently produce "Sottotitoli e revisione a cura di QTSS" or a stray
   "Grazie" from nothing. Not calling it at all is the only reliable cure
   (spec §5, §31).
2. **Find commit points.** A pause in speech is the safe place to finalise
   text: no word straddles it, so anything before it can be confirmed and the
   audio buffer trimmed.

The model is Silero VAD, which **faster-whisper already bundles** as
``assets/silero_vad_v6.onnx`` (1.2 MB) and runs through onnxruntime. Reusing it
means no extra download and no extra dependency, and it works offline out of the
box. Measured cost: ~10 ms per second of audio, about 1% of a real-time budget.

Silero is a genuine speech classifier, not an energy gate: verified here, white
noise at the same amplitude as speech is correctly reported as *no speech*. That
is the property that makes it useful in a noisy room, and the reason the energy
fallback below is only a last resort for when onnxruntime cannot load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass
class VadSettings:
    """Tunables exposed in advanced settings (spec §16).

    Defaults are chosen for lecture and meeting audio: a short minimum speech
    duration so brief interjections are not lost, and a moderate silence
    duration so natural pauses between sentences become commit points without
    fragmenting a continuous speaker.
    """

    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 200
    max_speech_duration_s: float = 30.0

    def to_options(self):
        """Build a faster-whisper ``VadOptions``."""
        from faster_whisper.vad import VadOptions

        return VadOptions(
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            max_speech_duration_s=self.max_speech_duration_s,
        )


@dataclass(frozen=True)
class SpeechRegion:
    """A span of speech, in seconds relative to the audio passed in."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeechDetector:
    """Detects speech in 16 kHz mono float32 audio.

    Falls back to an energy gate if the Silero model cannot be loaded, so a
    broken onnxruntime degrades transcription quality rather than preventing
    the app from recording at all.
    """

    #: RMS below which the energy fallback calls a frame silent.
    _ENERGY_FLOOR = 0.005
    _ENERGY_FRAME_MS = 30

    def __init__(self, settings: VadSettings | None = None) -> None:
        self.settings = settings or VadSettings()
        self._silero_failed = False

    @property
    def using_silero(self) -> bool:
        return not self._silero_failed

    # -- detection --------------------------------------------------------

    def speech_regions(self, audio: np.ndarray) -> list[SpeechRegion]:
        """Speech spans within ``audio``, oldest first."""
        if audio.size == 0:
            return []

        if not self._silero_failed:
            try:
                from faster_whisper.vad import get_speech_timestamps

                stamps = get_speech_timestamps(
                    np.ascontiguousarray(audio, dtype=np.float32),
                    self.settings.to_options(),
                )
                return [
                    SpeechRegion(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE)
                    for s in stamps
                ]
            except Exception as exc:
                logger.warning(
                    "Silero VAD unavailable, falling back to an energy gate: %r", exc
                )
                self._silero_failed = True

        return self._energy_regions(audio)

    def has_speech(self, audio: np.ndarray) -> bool:
        """True if ``audio`` contains any speech worth transcribing."""
        return bool(self.speech_regions(audio))

    def speech_duration(self, audio: np.ndarray) -> float:
        """Total seconds of speech in ``audio``."""
        return sum(r.duration for r in self.speech_regions(audio))

    def trailing_silence(self, audio: np.ndarray) -> float:
        """Seconds of silence at the end of ``audio``.

        The streaming loop uses this to decide when it is safe to finalise
        text: a pause long enough means no word straddles the boundary.
        """
        if audio.size == 0:
            return 0.0
        total = audio.size / SAMPLE_RATE
        regions = self.speech_regions(audio)
        if not regions:
            return total
        return max(0.0, total - regions[-1].end)

    def leading_silence(self, audio: np.ndarray) -> float:
        """Seconds of silence before the first speech."""
        if audio.size == 0:
            return 0.0
        regions = self.speech_regions(audio)
        if not regions:
            return audio.size / SAMPLE_RATE
        return max(0.0, regions[0].start)

    # -- energy fallback --------------------------------------------------

    def _energy_regions(self, audio: np.ndarray) -> list[SpeechRegion]:
        """Crude RMS gate. Only used when Silero cannot run.

        This will call loud noise "speech"; that is accepted, because the
        alternative in a degraded environment is transcribing nothing.
        """
        frame = int(SAMPLE_RATE * self._ENERGY_FRAME_MS / 1000)
        if frame <= 0 or audio.size < frame:
            return []

        count = audio.size // frame
        frames = audio[: count * frame].reshape(count, frame)
        rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        loud = rms > self._ENERGY_FLOOR
        if not loud.any():
            return []

        min_speech = self.settings.min_speech_duration_ms / 1000
        max_gap = self.settings.min_silence_duration_ms / 1000
        pad = self.settings.speech_pad_ms / 1000
        total = audio.size / SAMPLE_RATE

        regions: list[SpeechRegion] = []
        start_idx: int | None = None
        silence_run = 0

        for i, is_loud in enumerate(loud):
            if is_loud:
                if start_idx is None:
                    start_idx = i
                silence_run = 0
            elif start_idx is not None:
                silence_run += 1
                if silence_run * frame / SAMPLE_RATE >= max_gap:
                    end_idx = i - silence_run + 1
                    regions.append((start_idx, end_idx))  # type: ignore[arg-type]
                    start_idx = None
                    silence_run = 0
        if start_idx is not None:
            regions.append((start_idx, len(loud)))  # type: ignore[arg-type]

        out: list[SpeechRegion] = []
        for start_i, end_i in regions:  # type: ignore[misc]
            start = max(0.0, start_i * frame / SAMPLE_RATE - pad)
            end = min(total, end_i * frame / SAMPLE_RATE + pad)
            if end - start >= min_speech:
                out.append(SpeechRegion(start, end))
        return out
