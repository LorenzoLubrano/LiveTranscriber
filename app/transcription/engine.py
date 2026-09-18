"""Whisper inference engine.

Wraps ``faster_whisper.WhisperModel`` with the behaviour the application needs:

* a GPU failure degrades to CPU instead of crashing (spec §7, §18);
* the model is warmed up at load, because the first CUDA inference on Blackwell
  pays a one-off PTX JIT compilation that would otherwise land on the user's
  first sentence;
* it is loaded with ``local_files_only=True`` so a downloaded model works with
  no network, and a missing one fails immediately with a clear message rather
  than silently downloading gigabytes (spec §6);
* every error reaching a caller carries a plain-language message.

This module knows nothing about streaming, windows or the GUI. It transcribes a
NumPy array and returns segments. Milestone 4 builds the real-time behaviour on
top of it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from app.transcription.hardware import (
    Accelerator,
    AcceleratorChoice,
    select_accelerator,
)
from app.transcription.models import ModelError, get_spec, is_available, resolve_model_path
from app.utils.logging_setup import describe_segment

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

#: Marks "detect the language" in settings and in the UI.
AUTO_LANGUAGE = "auto"

#: Languages offered first in the picker; Whisper supports many more.
PRIORITY_LANGUAGES: tuple[tuple[str, str], ...] = (
    (AUTO_LANGUAGE, "Automatico"),
    ("it", "Italiano"),
    ("en", "English"),
    ("es", "Español"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("pt", "Português"),
    ("nl", "Nederlands"),
    ("ru", "Русский"),
    ("zh", "中文"),
    ("ja", "日本語"),
)


@dataclass
class TranscriptionSettings:
    """Inference parameters. Defaults are the "recommended" values of spec §16."""

    language: str = AUTO_LANGUAGE
    translate_to_english: bool = False

    beam_size: int = 5
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    initial_prompt: str | None = None

    #: faster-whisper's own Silero pass, applied inside a transcribe call.
    vad_filter: bool = True
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 200

    #: Drops segments that look like hallucinations over silence.
    hallucination_silence_threshold: float | None = 2.0

    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    word_timestamps: bool = False

    @property
    def task(self) -> str:
        return "translate" if self.translate_to_english else "transcribe"

    def whisper_language(self) -> str | None:
        """``None`` asks faster-whisper to detect the language."""
        return None if self.language == AUTO_LANGUAGE else self.language

    def vad_parameters(self) -> dict[str, float | int]:
        return {
            "threshold": self.vad_threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
        }


@dataclass(frozen=True)
class Word:
    """One word with its own timing.

    Word timings are what make reliable streaming possible: they give an exact,
    safe point to cut a hypothesis, so text can be confirmed without waiting for
    a whole segment to settle.
    """

    start: float
    end: float
    text: str
    probability: float = 1.0

    def shifted(self, offset: float) -> Word:
        return Word(self.start + offset, self.end + offset, self.text, self.probability)


@dataclass(frozen=True)
class Segment:
    """One transcribed span of audio."""

    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    words: tuple[Word, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    def shifted(self, offset: float) -> Segment:
        """Same segment placed on the session timeline."""
        return Segment(
            start=self.start + offset,
            end=self.end + offset,
            text=self.text,
            no_speech_prob=self.no_speech_prob,
            avg_logprob=self.avg_logprob,
            compression_ratio=self.compression_ratio,
            words=tuple(w.shifted(offset) for w in self.words),
        )


@dataclass
class TranscriptionResult:
    """Everything one inference call produced."""

    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    inference_time: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def real_time_factor(self) -> float:
        """Audio seconds processed per second of wall clock. >1 keeps up live."""
        if self.inference_time <= 0:
            return 0.0
        return self.duration / self.inference_time


class EngineError(RuntimeError):
    """Inference failure, with a message safe to show to a user."""

    def __init__(self, user_message: str, technical: str = "") -> None:
        super().__init__(technical or user_message)
        self.user_message = user_message
        self.technical = technical or user_message


class TranscriptionEngine:
    """A loaded Whisper model, safe to call from a worker thread.

    One engine owns one model. CTranslate2 releases the GIL during inference,
    but a single model must not be driven from two threads at once, so calls are
    serialised here. Two sources that need genuine parallelism get two engines.
    """

    def __init__(
        self,
        model_key: str,
        accelerator: Accelerator = Accelerator.AUTO,
        compute_type: str = "",
        cpu_threads: int = 0,
    ) -> None:
        self.model_key = model_key
        self.requested_accelerator = accelerator
        self._requested_compute_type = compute_type
        self._requested_cpu_threads = cpu_threads

        self._model = None
        self._choice: AcceleratorChoice | None = None
        self._lock = threading.Lock()
        self._load_warning = ""

    # -- state ------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def choice(self) -> AcceleratorChoice | None:
        """How the model actually ended up running."""
        return self._choice

    @property
    def load_warning(self) -> str:
        """Non-fatal problem the user should be told about, or ""."""
        return self._load_warning

    # -- loading ----------------------------------------------------------

    def load(self) -> AcceleratorChoice:
        """Load the model, falling back to CPU if the GPU path fails.

        Returns the configuration actually in use. Raises :class:`EngineError`
        only when even CPU could not be made to work.
        """
        if self._model is not None:
            assert self._choice is not None
            return self._choice

        spec = get_spec(self.model_key)
        if not is_available(self.model_key):
            raise ModelError(
                f"Il modello {spec.display_name} non è disponibile. "
                f"Scaricalo dalle impostazioni ({spec.size_label}).",
                f"model {self.model_key!r} not present on disk",
            )
        model_path = resolve_model_path(self.model_key)

        choice = select_accelerator(
            self.requested_accelerator,
            self._requested_compute_type,
            model_name=self.model_key,
            cpu_threads=self._requested_cpu_threads,
        )
        self._load_warning = choice.fallback_reason

        if choice.is_gpu:
            # Windows does not search site-packages for DLLs, so the pip-
            # installed cuBLAS/cuDNN must be registered before any CUDA work.
            from app.transcription.cuda_setup import ensure_cuda_libraries

            ensure_cuda_libraries()
            try:
                self._load_with(choice, model_path)
                return choice
            except Exception as exc:
                message = _friendly_gpu_error(exc)
                logger.warning("GPU load failed, falling back to CPU: %r", exc)
                self._load_warning = message
                choice = select_accelerator(
                    Accelerator.CPU,
                    "",  # a GPU compute type must not leak into the CPU attempt
                    model_name=self.model_key,
                    cpu_threads=self._requested_cpu_threads,
                )

        try:
            self._load_with(choice, model_path)
        except Exception as exc:
            raise EngineError(
                f"Impossibile caricare il modello {spec.display_name}.",
                f"CPU load failed: {exc!r}",
            ) from exc
        return choice

    def _load_with(self, choice: AcceleratorChoice, model_path: str) -> None:
        from faster_whisper import WhisperModel

        started = time.monotonic()
        model = WhisperModel(
            model_path,
            device=choice.device,
            device_index=choice.device_index,
            compute_type=choice.compute_type,
            cpu_threads=choice.cpu_threads,
            # Never reach the network at load time: the model is already on disk
            # and this is what makes the app work fully offline.
            local_files_only=True,
        )
        self._model = model
        self._choice = choice
        logger.info(
            "Loaded %s on %s in %.2fs", self.model_key, choice, time.monotonic() - started
        )
        self._warm_up()

    def _warm_up(self) -> None:
        """Run one tiny inference so the first real segment is not the slowest.

        On Blackwell, CTranslate2 reaches sm_120 through PTX JIT: the very first
        kernel launch compiles, which can take seconds. Paying that here keeps it
        out of the user's first sentence.

        The warm-up is also the moment a broken GPU install becomes visible.
        ``WhisperModel(device="cuda")`` constructs happily without touching
        cuBLAS, so a missing ``cublas64_12.dll`` only surfaces at the first real
        operation. Re-raising CUDA failures here turns that into a clean
        fallback to CPU at load time, instead of an error — or a very long stall
        — once the user has already pressed Start.
        """
        assert self._model is not None
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        started = time.monotonic()
        try:
            segments, _ = self._model.transcribe(silence, language="en", beam_size=1)
            list(segments)  # the generator is lazy; force the work
        except Exception as exc:
            if self._choice is not None and self._choice.is_gpu:
                logger.warning("GPU warm-up failed: %r", exc)
                self._model = None
                raise
            logger.warning("Warm-up failed (not fatal on CPU): %r", exc)
            return
        logger.info("Warm-up completed in %.2fs", time.monotonic() - started)

    def unload(self) -> None:
        """Release the model and its VRAM."""
        with self._lock:
            model, self._model = self._model, None
        if model is None:
            return
        del model
        import gc

        gc.collect()
        logger.info("Unloaded model %s", self.model_key)

    # -- inference --------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        settings: TranscriptionSettings | None = None,
        offset: float = 0.0,
    ) -> TranscriptionResult:
        """Transcribe 16 kHz mono float32 audio.

        ``offset`` places the returned segments on the session timeline.
        """
        if self._model is None:
            self.load()
        assert self._model is not None

        settings = settings or TranscriptionSettings()
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        duration = audio.size / SAMPLE_RATE

        if audio.size == 0:
            return TranscriptionResult()

        started = time.monotonic()
        with self._lock:
            try:
                raw_segments, info = self._model.transcribe(
                    audio,
                    language=settings.whisper_language(),
                    task=settings.task,
                    beam_size=settings.beam_size,
                    temperature=settings.temperature,
                    condition_on_previous_text=settings.condition_on_previous_text,
                    initial_prompt=settings.initial_prompt,
                    vad_filter=settings.vad_filter,
                    vad_parameters=settings.vad_parameters() if settings.vad_filter else None,
                    no_speech_threshold=settings.no_speech_threshold,
                    compression_ratio_threshold=settings.compression_ratio_threshold,
                    log_prob_threshold=settings.log_prob_threshold,
                    hallucination_silence_threshold=settings.hallucination_silence_threshold,
                    word_timestamps=settings.word_timestamps,
                )
                segments = _collect(raw_segments, offset)
            except Exception as exc:
                raise EngineError(_friendly_inference_error(exc), f"{exc!r}") from exc

        elapsed = time.monotonic() - started
        result = TranscriptionResult(
            segments=segments,
            language=getattr(info, "language", "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=duration,
            inference_time=elapsed,
        )
        logger.debug(
            "Transcribed %.2fs audio in %.2fs (RTF %.1fx), %d segments",
            duration,
            elapsed,
            result.real_time_factor,
            len(segments),
        )
        for seg in segments:
            logger.debug(describe_segment(seg.text, seg.start, seg.end))
        return result

    def __enter__(self) -> TranscriptionEngine:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()


def _collect(raw_segments: Iterable, offset: float) -> list[Segment]:
    """Materialise faster-whisper's lazy generator into our own segments.

    The generator performs the actual inference as it is consumed, so this is
    where the time is spent.
    """
    out: list[Segment] = []
    for seg in raw_segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        words = tuple(
            Word(
                start=float(w.start) + offset,
                end=float(w.end) + offset,
                # faster-whisper keeps the leading space that joins words.
                text=str(w.word),
                probability=float(getattr(w, "probability", 1.0) or 0.0),
            )
            for w in (getattr(seg, "words", None) or [])
        )
        out.append(
            Segment(
                start=float(seg.start) + offset,
                end=float(seg.end) + offset,
                text=text,
                no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                avg_logprob=float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                compression_ratio=float(getattr(seg, "compression_ratio", 0.0) or 0.0),
                words=words,
            )
        )
    return out


def _friendly_gpu_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "cuda_error_out_of_memory" in text:
        return (
            "Memoria GPU insufficiente. "
            "La trascrizione continuerà utilizzando la CPU."
        )
    if "cublas" in text or "cudnn" in text:
        return (
            "Le librerie CUDA non sono disponibili o non sono compatibili. "
            "La trascrizione continuerà utilizzando la CPU."
        )
    if "no cuda" in text or "no kernel image" in text or "driver" in text:
        return (
            "La GPU NVIDIA non è utilizzabile per la trascrizione. "
            "La trascrizione continuerà utilizzando la CPU."
        )
    return "Accelerazione GPU non disponibile. La trascrizione utilizzerà la CPU."


def _friendly_inference_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text:
        return (
            "Memoria insufficiente durante la trascrizione. "
            "Prova a selezionare un modello più piccolo."
        )
    if "cublas" in text or "cuda" in text:
        return "Errore della GPU durante la trascrizione."
    return "Errore durante la trascrizione dell'audio."
