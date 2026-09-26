"""The recording session: everything that happens between Start and Stop.

Owns one pipeline per audio source and ties the layers together::

    CaptureStream -> RingBuffer -> StreamResampler -> StreamingTranscriber
                          |                                   |
                          +-> MixRecorder (WAV)               +-> Transcript

Each source gets its own worker thread. The thread drains the ring buffer,
converts to 16 kHz mono, writes the WAV, and drives transcription. Audio capture
itself happens on PortAudio's own callback threads and is never blocked by
inference: a model too slow for the machine costs latency, never recorded audio.

This module contains no Qt. The GUI subscribes with plain callbacks, which keeps
the session testable from a script and keeps Qt out of the audio path.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path

import numpy as np

from app.audio.capture import CaptureStream, CaptureWatchdog
from app.audio.devices import AudioDevice, DeviceError
from app.audio.loopback import create_loopback_capture
from app.audio.microphone import create_microphone_capture
from app.audio.recorder import MixRecorder
from app.audio.resampler import TARGET_SAMPLE_RATE, StreamResampler
from app.export import export_all
from app.export.json_export import SessionRecord
from app.sessions.autosave import Autosave
from app.sessions.keepup import KeepUpMonitor, Level
from app.sessions.transcript import Source, Transcript, TranscriptStats
from app.transcription.engine import TranscriptionEngine, TranscriptionSettings
from app.transcription.hardware import Accelerator
from app.transcription.streaming import (
    StreamingSettings,
    StreamingTranscriber,
    StreamingUpdate,
)
from app.transcription.vad import VadSettings
from app.utils.paths import default_recordings_dir, unique_session_dir

logger = logging.getLogger(__name__)

#: Longest interval between inference passes the mitigation will grow to.
#: Beyond this, the wait for text becomes worse than the backlog it fixes.
MAX_CHUNK_DURATION_S = 8.0

#: Seconds of audio to wait before widening the interval again, so one bad
#: stretch does not walk the interval straight to the cap.
CHUNK_WIDEN_COOLDOWN_S = 45.0


class InputMode(StrEnum):
    """Which sources to record (spec §3)."""

    PC = "pc"
    MICROPHONE = "microphone"
    BOTH = "both"

    @property
    def label(self) -> str:
        return {
            InputMode.PC: "Audio PC",
            InputMode.MICROPHONE: "Microfono",
            InputMode.BOTH: "PC + Microfono",
        }[self]

    @property
    def sources(self) -> tuple[Source, ...]:
        if self is InputMode.PC:
            return (Source.PC,)
        if self is InputMode.MICROPHONE:
            return (Source.MIC,)
        return (Source.PC, Source.MIC)


class SessionState(StrEnum):
    IDLE = auto()
    STARTING = auto()
    RECORDING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()


@dataclass
class SessionConfig:
    """Everything chosen before pressing Start."""

    mode: InputMode = InputMode.PC
    loopback_device: AudioDevice | None = None
    microphone_device: AudioDevice | None = None

    model: str = "small"
    accelerator: Accelerator = Accelerator.AUTO
    language: str = "auto"
    translate_to_english: bool = False

    title: str = ""
    output_root: Path | None = None

    streaming: StreamingSettings = field(default_factory=StreamingSettings)
    vad: VadSettings = field(default_factory=VadSettings)

    def transcription_settings(self) -> TranscriptionSettings:
        return TranscriptionSettings(
            language=self.language,
            translate_to_english=self.translate_to_english,
        )


@dataclass
class SourceStats:
    """Live numbers for one source, polled by the UI."""

    level: float = 0.0
    dropped_frames: int = 0
    runs: int = 0
    skipped_silent: int = 0
    inference_time: float = 0.0


class SessionCallbacks:
    """Plain callbacks the GUI supplies. All are optional.

    Called from worker threads, so a Qt consumer must hop to the GUI thread —
    ``QMetaObject.invokeMethod`` or a queued signal — before touching widgets.
    """

    def __init__(
        self,
        on_update: Callable[[Source, StreamingUpdate], None] | None = None,
        on_state: Callable[[SessionState], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.on_update = on_update
        self.on_state = on_state
        self.on_error = on_error
        self.on_warning = on_warning

    def update(self, source: Source, update: StreamingUpdate) -> None:
        self._safe(self.on_update, source, update)

    def state(self, state: SessionState) -> None:
        self._safe(self.on_state, state)

    def error(self, message: str) -> None:
        logger.error("Session error surfaced: %s", message)
        self._safe(self.on_error, message)

    def warning(self, message: str) -> None:
        logger.warning("Session warning surfaced: %s", message)
        self._safe(self.on_warning, message)

    @staticmethod
    def _safe(fn, *args) -> None:
        if fn is None:
            return
        try:
            fn(*args)
        except Exception:
            logger.exception("Session callback raised")


class _SourcePipeline:
    """Capture, record and transcribe one source on its own thread."""

    def __init__(
        self,
        source: Source,
        stream: CaptureStream,
        ring,
        transcriber: StreamingTranscriber,
        write_audio: Callable[[np.ndarray], None],
        callbacks: SessionCallbacks,
        transcript: Transcript,
    ) -> None:
        self.source = source
        self.stream = stream
        self.ring = ring
        self.transcriber = transcriber
        self.write_audio = write_audio
        self.callbacks = callbacks
        self.transcript = transcript

        self.resampler = StreamResampler(stream.sample_rate, stream.channels)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()

        self.keepup = KeepUpMonitor()
        #: Stream position of the last interval widening. Starts before the
        #: beginning of time so the first one is never held back by the cooldown.
        self._widened_at = float("-inf")
        self._reported_overflow = False

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self.stream.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"pipeline-{self.source.value}", daemon=True
        )
        self._thread.start()

    def pause(self) -> None:
        self._paused.set()
        self.stream.pause()

    def resume(self) -> None:
        self.stream.resume()
        self._paused.clear()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=30)
        self.stream.stop()

    @property
    def stats(self) -> SourceStats:
        return SourceStats(
            level=self.stream.level,
            dropped_frames=self.stream.stats.frames_dropped,
            runs=self.transcriber.runs,
            skipped_silent=self.transcriber.skipped_silent_runs,
            inference_time=self.transcriber.total_inference_time,
        )

    # -- the worker -------------------------------------------------------

    def _run(self) -> None:
        logger.info("Pipeline started for %s", self.source.value)
        try:
            while not self._stop.is_set():
                self._drain()
                self._check_overflow()
                if self._paused.is_set():
                    time.sleep(0.05)
                    continue
                if self.transcriber.should_run():
                    self._transcribe()
                else:
                    time.sleep(0.05)

            # Stopping: take whatever is left, then flush the tail.
            self._drain(final=True)
            self._finish()
        except Exception:
            logger.exception("Pipeline for %s crashed", self.source.value)
            self.callbacks.error(
                "La trascrizione si è interrotta per un errore interno. "
                "La registrazione audio è stata salvata."
            )

    def _drain(self, final: bool = False) -> None:
        """Move audio from the ring buffer into the WAV and the transcriber."""
        raw = self.ring.read_all()
        if raw.size == 0 and not final:
            return

        audio = self.resampler.process(raw) if raw.size else np.zeros(0, np.float32)
        if final:
            tail = self.resampler.flush()
            if tail.size:
                audio = np.concatenate((audio, tail)) if audio.size else tail

        if audio.size == 0:
            return

        # Written to disk before transcription, so a crash during inference
        # cannot cost recorded audio.
        self.write_audio(audio)
        if not self._paused.is_set():
            self.transcriber.feed(audio)

    def _transcribe(self) -> None:
        started = time.monotonic()
        try:
            update = self.transcriber.process()
        except Exception as exc:
            logger.exception("Inference failed for %s", self.source.value)
            self.callbacks.warning(
                getattr(exc, "user_message", "Errore durante la trascrizione.")
            )
            return

        # The whole pass, not just the model's own timing: the VAD, the
        # hypothesis buffer and the copies are all time the ring buffer spends
        # filling up with nobody draining it.
        self._check_keeping_up(time.monotonic() - started)

        if update is not None and update.has_content:
            self._publish(update)

    # -- keeping up -------------------------------------------------------

    def _check_keeping_up(self, elapsed: float) -> None:
        """React when inference is slower than the audio arriving.

        Two things happen, in this order: the work per second of audio is
        reduced, and the user is told once. Reducing the work first means the
        message arrives alongside a machine that has already stopped getting
        worse, instead of one still sliding.

        The model is never changed here. Loading another model mid-recording
        would cost seconds of audio and silently change the quality of a
        transcript in progress — that is the user's decision to make.
        """
        self.keepup.record(elapsed, self.transcriber.stream_position)

        level = self.keepup.take_warning()
        if level is None:
            return

        widened = self._widen_chunk()
        if level is Level.OVERRUN:
            message = (
                "La trascrizione non riesce a stare al passo con l'audio su "
                "questo PC. L'audio continua a essere registrato, ma il testo "
                "arriva con molto ritardo: conviene fermarsi e scegliere un "
                "modello più leggero."
            )
        else:
            message = (
                "Questo PC è al limite con il modello scelto. Il testo può "
                "arrivare con qualche secondo di ritardo; con un modello più "
                "leggero la trascrizione resta al passo."
            )
        if widened:
            message += (
                f" L'intervallo tra le elaborazioni è stato portato a "
                f"{self.transcriber.streaming.chunk_duration:.0f}s per ridurre "
                "il carico."
            )
        self.callbacks.warning(message)

    def _widen_chunk(self) -> bool:
        """Transcribe less often, so each second of audio costs less.

        Streaming re-transcribes the working buffer on every pass, so the cost
        per second of audio is roughly ``buffer / chunk_duration`` model runs.
        Doubling the interval halves the work, and pays for it in latency —
        which is the right trade when the alternative is unbounded latency.
        """
        position = self.transcriber.stream_position
        if position - self._widened_at < CHUNK_WIDEN_COOLDOWN_S:
            return False

        streaming = self.transcriber.streaming
        if streaming.chunk_duration >= MAX_CHUNK_DURATION_S:
            return False

        before = streaming.chunk_duration
        streaming.chunk_duration = min(before * 2.0, MAX_CHUNK_DURATION_S)
        self._widened_at = position
        logger.info(
            "Widened the chunk interval for %s: %.1fs -> %.1fs",
            self.source.value,
            before,
            streaming.chunk_duration,
        )
        return True

    def _check_overflow(self) -> None:
        """Tell the user once if captured audio was actually dropped.

        The ring buffer overwrites its oldest frames rather than blocking the
        audio callback, so an overrun costs recorded audio and nothing else
        notices. Counting it without ever saying so would hide the one failure
        in this app that cannot be repaired afterwards.
        """
        if self._reported_overflow or self.stream.stats.frames_dropped == 0:
            return
        self._reported_overflow = True
        lost = self.stream.stats.frames_dropped / max(1, self.stream.sample_rate)
        self.callbacks.warning(
            f"Il PC non è riuscito a seguire l'audio in ingresso e circa "
            f"{lost:.0f}s di registrazione sono andati persi. Usa un modello "
            "più leggero per il resto della registrazione."
        )

    def _finish(self) -> None:
        try:
            update = self.transcriber.finish()
        except Exception:
            logger.exception("Final flush failed for %s", self.source.value)
            return
        if update.confirmed:
            self._publish(update)
        self.transcript.clear_provisional(self.source)

    def _publish(self, update: StreamingUpdate) -> None:
        for word_group in _group_confirmed(update):
            start, end, text = word_group
            self.transcript.add(self.source, start, end, text)

        if update.provisional:
            self.transcript.set_provisional(
                self.source,
                update.provisional[0].start,
                update.provisional[-1].end,
                update.provisional_text,
            )
        else:
            self.transcript.clear_provisional(self.source)

        self.callbacks.update(self.source, update)


def _group_confirmed(update: StreamingUpdate) -> list[tuple[float, float, str]]:
    """Confirmed words as one timed chunk, or nothing."""
    if not update.confirmed:
        return []
    return [(update.confirmed[0].start, update.confirmed[-1].end, update.confirmed_text)]


class RecordingSession:
    """One recording, from Start to Stop.

    Create, :meth:`start`, then :meth:`pause` / :meth:`resume` as needed, and
    :meth:`stop`. A session is not restartable; make a new one.
    """

    def __init__(
        self,
        config: SessionConfig,
        callbacks: SessionCallbacks | None = None,
        transcript: Transcript | None = None,
    ) -> None:
        self.config = config
        self.callbacks = callbacks if callbacks is not None else SessionCallbacks()
        # `transcript or Transcript()` would be wrong: Transcript defines
        # __bool__ as "has segments", so an empty one is falsy and the caller's
        # object would be silently replaced. The GUI passes its own transcript
        # in, and would then watch an object nothing ever writes to.
        self.transcript = transcript if transcript is not None else Transcript()

        self.directory: Path | None = None
        self.started_at: float = 0.0
        self._state = SessionState.IDLE
        self._pipelines: dict[Source, _SourcePipeline] = {}
        self._engines: list[TranscriptionEngine] = []
        self._recorder: MixRecorder | None = None
        self._autosave: Autosave | None = None
        self.record = SessionRecord()
        self.exports: dict[str, Path] = {}
        self._watchdog = CaptureWatchdog()
        self._paused_at: float = 0.0
        self._paused_total: float = 0.0
        self._lock = threading.RLock()

    # -- state ------------------------------------------------------------

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state in (SessionState.RECORDING, SessionState.PAUSED)

    @property
    def elapsed(self) -> float:
        """Seconds recorded, excluding time spent paused."""
        if not self.started_at:
            return 0.0
        if self._state is SessionState.PAUSED:
            return self._paused_at - self.started_at - self._paused_total
        if self._state in (SessionState.STOPPED, SessionState.IDLE):
            return self._final_elapsed
        return time.monotonic() - self.started_at - self._paused_total

    _final_elapsed: float = 0.0

    def stats(self) -> dict[Source, SourceStats]:
        return {source: p.stats for source, p in self._pipelines.items()}

    def _set_state(self, state: SessionState) -> None:
        self._state = state
        self.callbacks.state(state)

    # -- lifecycle --------------------------------------------------------

    def start(self) -> Path:
        """Open devices, load models and begin. Returns the session directory."""
        with self._lock:
            if self._state is not SessionState.IDLE:
                raise RuntimeError(f"cannot start a session in state {self._state}")
            self._set_state(SessionState.STARTING)

        try:
            self.directory = self._make_directory()
            captures = self._open_devices()
            self._recorder = self._make_recorder(captures)
            self._build_pipelines(captures)

            for pipeline in self._pipelines.values():
                pipeline.start()
                self._watchdog.add(pipeline.stream)
            self._watchdog.start()

            self.record = self._build_record()
            self._autosave = Autosave(
                self.directory,
                self.transcript,
                self.record,
                on_sync=self.sync_to_disk,
                on_error=self.callbacks.warning,
            )
            self._autosave.start()

            self.started_at = time.monotonic()
            self._set_state(SessionState.RECORDING)
            logger.info("Session started in %s", self.directory)
            return self.directory
        except Exception:
            self._teardown()
            self._set_state(SessionState.IDLE)
            raise

    def pause(self) -> None:
        with self._lock:
            if self._state is not SessionState.RECORDING:
                return
            self._paused_at = time.monotonic()
            for pipeline in self._pipelines.values():
                pipeline.pause()
            self._set_state(SessionState.PAUSED)

    def resume(self) -> None:
        with self._lock:
            if self._state is not SessionState.PAUSED:
                return
            self._paused_total += time.monotonic() - self._paused_at
            for pipeline in self._pipelines.values():
                pipeline.resume()
            self._set_state(SessionState.RECORDING)

    def stop(self) -> Path | None:
        """Finish the session, flushing audio and the last transcription."""
        with self._lock:
            if self._state in (SessionState.IDLE, SessionState.STOPPED):
                return self.directory
            self._final_elapsed = self.elapsed
            self._set_state(SessionState.STOPPING)

        self._teardown()
        self._write_exports()
        self._set_state(SessionState.STOPPED)
        logger.info(
            "Session stopped: %.1fs, %d segments",
            self._final_elapsed,
            len(self.transcript),
        )
        return self.directory

    def _teardown(self) -> None:
        self._watchdog.stop()
        for pipeline in self._pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                logger.exception("Pipeline shutdown failed")
        self._pipelines.clear()

        if self._recorder is not None:
            try:
                self._recorder.close()
            except Exception:
                logger.exception("Recorder shutdown failed")
            self._recorder = None

        for engine in self._engines:
            try:
                engine.unload()
            except Exception:
                logger.exception("Engine unload failed")
        self._engines.clear()

    # -- construction -----------------------------------------------------

    def _make_directory(self) -> Path:
        # Settings persist the folder as a string, so coerce rather than trust
        # the annotation.
        root = Path(self.config.output_root) if self.config.output_root else (
            default_recordings_dir()
        )
        directory = unique_session_dir(root, self.config.title)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _open_devices(self) -> dict[Source, tuple[CaptureStream, object]]:
        captures: dict[Source, tuple[CaptureStream, object]] = {}
        wanted = self.config.mode.sources

        try:
            if Source.PC in wanted:
                captures[Source.PC] = create_loopback_capture(
                    self.config.loopback_device, on_error=self.callbacks.error
                )
            if Source.MIC in wanted:
                captures[Source.MIC] = create_microphone_capture(
                    self.config.microphone_device, on_error=self.callbacks.error
                )
        except DeviceError:
            for stream, _ in captures.values():
                stream.stop()
            raise
        return captures

    def _make_recorder(self, captures: dict) -> MixRecorder:
        assert self.directory is not None
        both = len(captures) == 2
        return MixRecorder(
            self.directory / ("pc_audio.wav" if both else "audio.wav")
            if Source.PC in captures
            else None,
            self.directory / ("microphone.wav" if both else "audio.wav")
            if Source.MIC in captures
            else None,
            self.directory / "mixed.wav" if both else None,
            TARGET_SAMPLE_RATE,
        )

    def _build_pipelines(self, captures: dict) -> None:
        assert self._recorder is not None
        settings = self.config.transcription_settings()

        for source, (stream, ring) in captures.items():
            # One engine per source: a single CTranslate2 model must not be
            # driven from two threads at once, and PC and MIC transcribe
            # concurrently.
            engine = TranscriptionEngine(
                self.config.model, accelerator=self.config.accelerator
            )
            engine.load()
            if engine.load_warning:
                self.callbacks.warning(engine.load_warning)
            self._engines.append(engine)

            transcriber = StreamingTranscriber(
                engine,
                TranscriptionSettings(**vars(settings)),
                self.config.streaming,
                self.config.vad,
                source=source.value,
            )
            writer = (
                self._recorder.write_pc if source is Source.PC else self._recorder.write_mic
            )
            self._pipelines[source] = _SourcePipeline(
                source, stream, ring, transcriber, writer, self.callbacks, self.transcript
            )

    # -- output -----------------------------------------------------------

    def _build_record(self) -> SessionRecord:
        from datetime import datetime

        from app import __version__

        choice = self._engines[0].choice if self._engines else None
        return SessionRecord(
            title=self.config.title,
            directory=str(self.directory),
            started=datetime.now().isoformat(timespec="seconds"),
            mode=str(self.config.mode),
            model=self.config.model,
            language=self.config.language,
            device=choice.device if choice else "",
            compute_type=choice.compute_type if choice else "",
            loopback_device=(
                self.config.loopback_device.display_name
                if self.config.loopback_device
                else ""
            ),
            microphone_device=(
                self.config.microphone_device.display_name
                if self.config.microphone_device
                else ""
            ),
            audio_files={k: str(v) for k, v in self.audio_paths.items()},
            app_version=__version__,
        )

    def _write_exports(self) -> None:
        """Write TXT, SRT, VTT and JSON, then close the session record."""
        from datetime import datetime

        if self.directory is None:
            return

        self.record.finished = datetime.now().isoformat(timespec="seconds")
        self.record.duration = round(self._final_elapsed, 3)
        self.record.stats = TranscriptStats.of(self.transcript).to_dict()
        self.record.audio_files = {k: str(v) for k, v in self.audio_paths.items()}

        if len(self.transcript):
            try:
                self.exports = export_all(
                    self.transcript, self.directory, title=self.config.title
                )
                self.record.exports = {k: str(v) for k, v in self.exports.items()}
            except Exception:
                logger.exception("Export failed")
                self.callbacks.warning(
                    "Impossibile salvare alcuni formati della trascrizione. "
                    "L'audio e i dati sono comunque salvati."
                )

        # Marks the session complete, which is what stops recovery offering it.
        if self._autosave is not None:
            self._autosave.stop(completed=True)
            self._autosave = None

    @property
    def audio_paths(self) -> dict[str, Path]:
        return self._recorder.paths if self._recorder else {}

    def sync_to_disk(self) -> None:
        """Flush WAV headers. Called by autosave."""
        if self._recorder is not None:
            self._recorder.sync()

    def __enter__(self) -> RecordingSession:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
