"""Pseudo-streaming transcription.

Whisper is not a streaming model. It transcribes a fixed window and re-decides
everything each time it runs, which produces the failures the spec lists
(§5): words cut between windows, phrases repeated, sentences duplicated, context
lost at boundaries, and text invented during silence.

The fix used here is **LocalAgreement-2**: transcribe a growing buffer
repeatedly, and only confirm text that two consecutive runs agreed on.

    run 1:  "Consideriamo adesso la Hamiltoniana del sis"
    run 2:  "Consideriamo adesso la Hamiltoniana del sistema. Nel limite"
                                                        ^
            agreement ends here -> everything before it is confirmed,
            the rest stays provisional and may still change

Why this works for each failure mode:

* **Cut words** — a word at the buffer edge has not been seen twice yet, so it
  is never confirmed while it is still truncated.
* **Repetitions and duplicates** — confirmed text is removed from the audio
  buffer, so it cannot be emitted a second time.
* **Lost context** — the buffer keeps unconfirmed audio *and* the last confirmed
  words are passed back as ``initial_prompt``, so the model still sees what came
  before.
* **Hallucinations on silence** — VAD gates the whole thing: Whisper is not
  called at all when there is no speech.

Buffer growth is bounded two ways: audio before the last confirmed word is
discarded, and a hard cap forces a commit if a speaker never pauses.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from app.transcription.engine import (
    SAMPLE_RATE,
    Segment,
    TranscriptionEngine,
    TranscriptionSettings,
    Word,
)
from app.transcription.vad import SpeechDetector, VadSettings

logger = logging.getLogger(__name__)

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def normalise_word(text: str) -> str:
    """Comparison form of a word: lowercase, no surrounding punctuation.

    Whisper often changes only punctuation or capitalisation between runs —
    "sistema" then "sistema." — and treating those as disagreement would stall
    confirmation forever.
    """
    return _PUNCTUATION.sub("", text.strip().lower())


@dataclass
class StreamingSettings:
    """Tunables for the real-time loop (spec §16)."""

    #: How much new audio to accumulate before running inference again.
    #: Lower means lower latency and more GPU/CPU work per second of audio.
    chunk_duration: float = 2.0

    #: Never let the working buffer exceed this. If a speaker does not pause,
    #: text is force-committed to keep latency and memory bounded.
    max_buffer_duration: float = 28.0

    #: Trailing silence that makes it safe to commit everything pending.
    silence_commit_duration: float = 0.8

    #: Runs that must agree before text is confirmed. 2 is the standard
    #: accuracy/latency trade-off; 3 is steadier but slower to commit.
    agreement_runs: int = 2

    #: Words of confirmed text fed back as context for the next run.
    context_words: int = 30

    #: Skip inference entirely when the buffer holds less speech than this.
    min_speech_duration: float = 0.3

    def validate(self) -> StreamingSettings:
        self.chunk_duration = max(0.5, min(self.chunk_duration, 30.0))
        self.max_buffer_duration = max(self.chunk_duration * 2, self.max_buffer_duration)
        self.agreement_runs = max(1, min(self.agreement_runs, 4))
        return self


@dataclass
class StreamingUpdate:
    """What changed after one inference run."""

    #: Newly confirmed words, on the session timeline. Never revised.
    confirmed: list[Word] = field(default_factory=list)
    #: Current best guess for what follows. Replaced on every update.
    provisional: list[Word] = field(default_factory=list)
    #: Seconds between the audio arriving and this text existing.
    latency: float = 0.0
    inference_time: float = 0.0
    audio_duration: float = 0.0
    language: str = ""

    @property
    def confirmed_text(self) -> str:
        return "".join(w.text for w in self.confirmed).strip()

    @property
    def provisional_text(self) -> str:
        return "".join(w.text for w in self.provisional).strip()

    @property
    def has_content(self) -> bool:
        return bool(self.confirmed or self.provisional)


class HypothesisBuffer:
    """Confirms text that successive transcriptions agree on.

    Holds the previous run's unconfirmed tail and compares it with the next
    run's output. The longest matching prefix is promoted to confirmed; the
    remainder is kept for the next comparison.
    """

    def __init__(self, agreement_runs: int = 2) -> None:
        self.agreement_runs = max(1, agreement_runs)
        #: Previous hypotheses' tails, newest last.
        self._history: list[list[Word]] = []
        #: Timeline position up to which text has been confirmed.
        self.committed_until: float = 0.0
        self._pending: list[Word] = []

    @property
    def pending(self) -> list[Word]:
        """Words not yet confirmed — the provisional text."""
        return list(self._pending)

    def insert(self, words: list[Word]) -> list[Word]:
        """Feed a new hypothesis. Returns the words newly confirmed."""
        # Anything at or before the commit point is already settled.
        fresh = [w for w in words if w.end > self.committed_until + 1e-6]

        self._history.append(fresh)
        if len(self._history) > self.agreement_runs:
            self._history.pop(0)

        if len(self._history) < self.agreement_runs:
            self._pending = fresh
            return []

        confirmed = self._common_prefix(self._history)
        if confirmed:
            self.committed_until = confirmed[-1].end
            # Drop the confirmed prefix from every retained hypothesis so the
            # next comparison starts from the new frontier.
            self._history = [
                [w for w in hyp if w.end > self.committed_until + 1e-6]
                for hyp in self._history
            ]

        self._pending = self._history[-1] if self._history else []
        return confirmed

    @staticmethod
    def _common_prefix(hypotheses: list[list[Word]]) -> list[Word]:
        """Longest leading run of words that every hypothesis agrees on."""
        if not hypotheses or any(not h for h in hypotheses):
            return []

        shortest = min(len(h) for h in hypotheses)
        prefix: list[Word] = []
        for i in range(shortest):
            forms = {normalise_word(h[i].text) for h in hypotheses}
            if len(forms) != 1 or not next(iter(forms)):
                break
            # Keep the most recent run's timing; it saw the most audio.
            prefix.append(hypotheses[-1][i])
        return prefix

    def flush(self) -> list[Word]:
        """Confirm everything still pending. Used when the stream ends."""
        remaining = self._pending
        if remaining:
            self.committed_until = remaining[-1].end
        self._pending = []
        self._history = []
        return remaining

    def reset(self, committed_until: float = 0.0) -> None:
        self._history = []
        self._pending = []
        self.committed_until = committed_until


class StreamingTranscriber:
    """Turns a continuous audio stream into confirmed and provisional text.

    Not thread-safe by itself; drive it from one worker thread per source.
    """

    def __init__(
        self,
        engine: TranscriptionEngine,
        settings: TranscriptionSettings | None = None,
        streaming: StreamingSettings | None = None,
        vad: VadSettings | None = None,
        source: str = "",
    ) -> None:
        self.engine = engine
        self.settings = settings or TranscriptionSettings()
        self.streaming = (streaming or StreamingSettings()).validate()
        self.detector = SpeechDetector(vad)
        self.source = source

        # Word timings are what make LocalAgreement possible.
        self.settings.word_timestamps = True
        # The buffer is already VAD-gated here; running Whisper's own VAD pass
        # as well would shift timestamps a second time for no benefit.
        self.settings.vad_filter = False

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0.0      # session time of _buffer[0]
        self._stream_position = 0.0   # session time of the newest sample fed
        self._last_run_position = 0.0

        self._hypotheses = HypothesisBuffer(self.streaming.agreement_runs)
        self._confirmed: list[Word] = []
        self._lock = threading.Lock()

        self.runs = 0
        self.skipped_silent_runs = 0
        self.total_inference_time = 0.0
        self._latencies: list[float] = []

    # -- state ------------------------------------------------------------

    @property
    def confirmed_words(self) -> list[Word]:
        with self._lock:
            return list(self._confirmed)

    @property
    def confirmed_text(self) -> str:
        return "".join(w.text for w in self.confirmed_words).strip()

    @property
    def buffer_duration(self) -> float:
        return self._buffer.size / SAMPLE_RATE

    @property
    def stream_position(self) -> float:
        """Session time of the newest sample fed, in seconds of audio.

        The denominator for any "is this machine keeping up" question: inference
        has to cost less than the audio arriving, and this is that audio.
        """
        return self._stream_position

    @property
    def average_latency(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

    # -- feeding ----------------------------------------------------------

    def feed(self, audio: np.ndarray) -> None:
        """Add 16 kHz mono float32 audio to the working buffer."""
        if audio.size == 0:
            return
        with self._lock:
            self._buffer = np.concatenate(
                (self._buffer, np.ascontiguousarray(audio, dtype=np.float32).reshape(-1))
            )
            self._stream_position += audio.size / SAMPLE_RATE

    def should_run(self) -> bool:
        """True when enough new audio has arrived to be worth transcribing."""
        pending = self._stream_position - self._last_run_position
        if pending >= self.streaming.chunk_duration:
            return True
        # A full buffer must be processed even if the chunk is not complete.
        return self.buffer_duration >= self.streaming.max_buffer_duration

    # -- the streaming step ----------------------------------------------

    def process(self) -> StreamingUpdate | None:
        """Run one inference pass. Returns None when there was nothing to do."""
        with self._lock:
            audio = self._buffer.copy()
            buffer_start = self._buffer_start
            stream_position = self._stream_position

        if audio.size == 0:
            return None

        started = time.monotonic()
        self._last_run_position = stream_position

        # Gate on speech: never ask Whisper to transcribe silence. The regions
        # are kept: they are also what tells invented text from real text below.
        regions = self.detector.speech_regions(audio)
        speech = sum(region.duration for region in regions)
        if speech < self.streaming.min_speech_duration:
            self.skipped_silent_runs += 1
            self._drop_silent_buffer(audio, buffer_start)
            return None

        result = self.engine.transcribe(
            audio, self.settings, offset=buffer_start
        )
        self.runs += 1
        self.total_inference_time += result.inference_time

        words = _words_of(result.segments)
        # Whisper invents text for stretches with no speech in them, and the
        # confirmed context fed back as a prompt tells it exactly what to
        # invent: the previous sentence, again. The VAD already knows where the
        # speech was, so anything outside it is dropped before it can be
        # confirmed.
        words = drop_hallucinated_words(words, regions, buffer_start)
        if not words:
            self._drop_silent_buffer(audio, buffer_start)
            return None

        newly_confirmed = self._hypotheses.insert(words)

        # A clear pause means nothing is mid-word: commit everything.
        trailing = self.detector.trailing_silence(audio)
        if trailing >= self.streaming.silence_commit_duration:
            newly_confirmed = newly_confirmed + self._hypotheses.flush()
        elif self.buffer_duration >= self.streaming.max_buffer_duration:
            # A speaker who never pauses must not grow the buffer forever.
            logger.debug("Buffer full (%.1fs); forcing a commit", self.buffer_duration)
            newly_confirmed = newly_confirmed + self._hypotheses.flush()

        if newly_confirmed:
            with self._lock:
                self._confirmed.extend(newly_confirmed)
            self._trim_buffer(self._hypotheses.committed_until)

        latency = stream_position + buffer_start - (
            newly_confirmed[-1].end if newly_confirmed else stream_position + buffer_start
        )
        elapsed = time.monotonic() - started
        self._latencies.append(elapsed)

        update = StreamingUpdate(
            confirmed=newly_confirmed,
            provisional=self._hypotheses.pending,
            latency=max(0.0, latency),
            inference_time=result.inference_time,
            audio_duration=audio.size / SAMPLE_RATE,
            language=result.language,
        )
        self._update_prompt()
        return update

    def finish(self) -> StreamingUpdate:
        """Transcribe whatever is left and confirm all of it.

        Called when recording stops, so the last sentence is not lost.
        """
        with self._lock:
            audio = self._buffer.copy()
            buffer_start = self._buffer_start

        confirmed: list[Word] = []
        regions = self.detector.speech_regions(audio) if audio.size else []
        speech = sum(region.duration for region in regions)
        if speech >= self.streaming.min_speech_duration:
            result = self.engine.transcribe(audio, self.settings, offset=buffer_start)
            self.runs += 1
            # The last pass is the most exposed to invented text: a recording
            # usually ends with someone stopping talking, so the tail of the
            # final buffer is silence with the whole transcript as its prompt.
            words = drop_hallucinated_words(
                _words_of(result.segments), regions, buffer_start
            )
            if words:
                # One last hypothesis, then accept everything still pending:
                # there is no more audio that could revise it.
                confirmed = self._hypotheses.insert(words)
                confirmed = confirmed + self._hypotheses.flush()

        if confirmed:
            with self._lock:
                self._confirmed.extend(confirmed)

        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)
            self._buffer_start = self._stream_position

        return StreamingUpdate(confirmed=confirmed, provisional=[])

    def reset(self) -> None:
        """Clear everything, keeping the stream position (used on resume)."""
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)
            self._buffer_start = self._stream_position
            self._confirmed = []
        self._hypotheses.reset()
        self.settings.initial_prompt = None

    # -- buffer management ------------------------------------------------

    def _trim_buffer(self, until: float) -> None:
        """Discard audio before ``until`` (session time). Bounds memory."""
        with self._lock:
            offset = until - self._buffer_start
            if offset <= 0:
                return
            samples = int(offset * SAMPLE_RATE)
            if samples <= 0 or samples > self._buffer.size:
                samples = min(max(samples, 0), self._buffer.size)
            self._buffer = self._buffer[samples:]
            self._buffer_start += samples / SAMPLE_RATE

    def _drop_silent_buffer(self, audio: np.ndarray, buffer_start: float) -> None:
        """Release silence so a quiet stretch cannot grow the buffer."""
        regions = self.detector.speech_regions(audio)
        if not regions:
            # Nothing but silence: keep a short tail in case speech is starting.
            keep = min(audio.size, int(self.streaming.chunk_duration * SAMPLE_RATE))
            with self._lock:
                dropped = self._buffer.size - keep
                if dropped > 0:
                    self._buffer = self._buffer[dropped:]
                    self._buffer_start += dropped / SAMPLE_RATE
            return
        # Drop audio before the first speech region.
        self._trim_buffer(buffer_start + max(0.0, regions[0].start - 0.2))

    def _update_prompt(self) -> None:
        """Feed recent confirmed words back as context for the next run.

        Whisper's own ``condition_on_previous_text`` is left off — it feeds the
        model its own unconfirmed output and is a known cause of repetition
        loops. Passing confirmed text explicitly gives the context without the
        feedback loop.
        """
        words = self.confirmed_words[-self.streaming.context_words :]
        if not words:
            self.settings.initial_prompt = None
            return
        self.settings.initial_prompt = "".join(w.text for w in words).strip() or None


def drop_hallucinated_words(
    words: list[Word], regions, offset: float
) -> list[Word]:
    """Keep only words that overlap a detected speech region.

    ``words`` carry session time; ``regions`` are relative to the buffer that
    was transcribed, so ``offset`` (the buffer's start) bridges the two.

    Overlap, not containment: Whisper's word timestamps are approximate and a
    real word often starts a fraction before the VAD calls it speech. What this
    catches is the other case entirely — text placed seconds away from any
    speech at all, which is invented.

    With no regions at all the words are returned untouched. A VAD that failed
    must not be able to delete a transcript.
    """
    if not regions or not words:
        return words

    spans = [(r.start + offset, r.end + offset) for r in regions]
    kept = [
        word
        for word in words
        if any(word.start < end and word.end > start for start, end in spans)
    ]
    if len(kept) != len(words):
        logger.debug(
            "Dropped %d word(s) outside the detected speech", len(words) - len(kept)
        )
    return kept


def _words_of(segments: list[Segment]) -> list[Word]:
    """Flatten segments to words, falling back to whole segments.

    If word timestamps are unavailable for some reason, a segment is treated as
    one long word so the pipeline degrades instead of producing nothing.
    """
    words: list[Word] = []
    for seg in segments:
        if seg.words:
            words.extend(seg.words)
        elif seg.text.strip():
            words.append(Word(seg.start, seg.end, f" {seg.text.strip()}", 1.0))
    return words


def merge_words(words: list[Word]) -> str:
    """Join words back into text."""
    return "".join(w.text for w in words).strip()


def words_to_segments(
    words: list[Word],
    source: str = "",
    max_gap: float = 1.0,
    max_duration: float = 15.0,
) -> list[Segment]:
    """Group words into readable segments, split at pauses.

    Used for export: SRT and VTT need lines of sensible length rather than one
    segment per word or one per multi-minute monologue.
    """
    if not words:
        return []

    segments: list[Segment] = []
    current: list[Word] = [words[0]]

    for previous, word in zip(words, words[1:], strict=False):
        too_far = word.start - previous.end > max_gap
        too_long = word.end - current[0].start > max_duration
        ends_sentence = previous.text.strip().endswith((".", "?", "!"))
        if too_far or too_long or (ends_sentence and word.end - current[0].start > 3.0):
            segments.append(_segment_of(current))
            current = []
        current.append(word)

    if current:
        segments.append(_segment_of(current))
    return segments


def _segment_of(words: list[Word]) -> Segment:
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text=merge_words(words),
        words=tuple(words),
    )


#: Callback signature used by the worker that drives a StreamingTranscriber.
UpdateCallback = Callable[[StreamingUpdate], None]
