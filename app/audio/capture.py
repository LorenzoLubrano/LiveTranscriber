"""Audio capture streams.

One :class:`CaptureStream` per source. The PortAudio callback does the absolute
minimum — wrap the incoming bytes in a NumPy view, push them into a bounded ring
buffer, update a scalar level — and returns. No allocation beyond the ring
buffer's own copy, no resampling, no disk I/O, no locks held across work, and it
cannot raise: an exception escaping a PortAudio callback kills the stream.

Everything expensive (resampling, VAD, inference, WAV writing) happens on
consumer threads that pull from the ring buffer.

WASAPI specifics that the code depends on:

* A loopback device is opened as an **input** stream on the loopback device's
  own index. PyAudioWPatch exposes those devices with ``isLoopbackDevice`` set.
* Shared mode only accepts the endpoint's own mix-format sample rate. Opening
  "Speakers (Realtek)" at 44100 when its mix format is 48000 fails, so the
  device's ``defaultSampleRate`` is always used and conversion happens later.
* When nothing is playing, a loopback stream still delivers buffers — they are
  digital silence. That is normal and must not be read as a dead device.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyaudiowpatch as pyaudio

from app.audio.devices import AudioDevice, DeviceError, get_pyaudio

logger = logging.getLogger(__name__)

#: Callback buffer size in frames. At 48 kHz this is ~21 ms — small enough to
#: keep capture latency low, large enough to avoid excessive callback overhead.
DEFAULT_FRAMES_PER_BUFFER = 1024

#: Seconds without a callback before a running stream is considered dead.
#: WASAPI delivers silence rather than nothing when idle, so a real gap this
#: long means the endpoint went away (unplugged, Bluetooth dropout, format change).
STALL_TIMEOUT_S = 3.0


@dataclass
class CaptureStats:
    """Counters for diagnostics and the UI. Read-only from outside."""

    frames_captured: int = 0
    frames_dropped: int = 0
    callback_count: int = 0
    input_overflows: int = 0
    last_callback_monotonic: float = 0.0
    peak_level: float = 0.0

    @property
    def dropped_ratio(self) -> float:
        total = self.frames_captured + self.frames_dropped
        return self.frames_dropped / total if total else 0.0


class CaptureStream:
    """A running capture from one device into a ring buffer.

    Parameters
    ----------
    device:
        The endpoint to capture. For system audio this must be the *loopback*
        device, not the render endpoint.
    ring:
        Destination buffer. Must have ``channels == device.capture_channels``.
    on_error:
        Called from a background thread when the device fails or stalls. Receives
        a user-facing message. Must not block.
    """

    def __init__(
        self,
        device: AudioDevice,
        ring,  # RingBuffer; untyped to avoid a circular import
        frames_per_buffer: int = DEFAULT_FRAMES_PER_BUFFER,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.device = device
        self.ring = ring
        self.frames_per_buffer = int(frames_per_buffer)
        self._on_error = on_error

        self.sample_rate = int(device.default_sample_rate)
        self.channels = device.capture_channels

        self._stream: pyaudio.Stream | None = None
        self._stats = CaptureStats()
        self._stats_lock = threading.Lock()
        self._paused = threading.Event()
        self._running = False
        self._error_reported = False

    # -- state ------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    @property
    def stats(self) -> CaptureStats:
        with self._stats_lock:
            return CaptureStats(**vars(self._stats))

    @property
    def level(self) -> float:
        """Peak level of the most recent callback, 0.0–1.0."""
        with self._stats_lock:
            return self._stats.peak_level

    def is_stalled(self) -> bool:
        """True if a running, unpaused stream has stopped receiving callbacks."""
        if not self._running or self._paused.is_set():
            return False
        with self._stats_lock:
            last = self._stats.last_callback_monotonic
        if last == 0.0:
            return False
        return (time.monotonic() - last) > STALL_TIMEOUT_S

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Open and start the stream. Raises :class:`DeviceError` on failure."""
        if self._running:
            return

        pa = get_pyaudio()
        try:
            self._stream = pa.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device.index,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self._callback,
                start=False,
            )
        except Exception as exc:
            raise DeviceError(
                self._friendly_open_error(exc),
                f"open failed for {self.device}: {exc!r}",
            ) from exc

        with self._stats_lock:
            self._stats.last_callback_monotonic = time.monotonic()

        try:
            self._stream.start_stream()
        except Exception as exc:
            self._close_stream()
            raise DeviceError(
                self._friendly_open_error(exc),
                f"start failed for {self.device}: {exc!r}",
            ) from exc

        self._running = True
        self._error_reported = False
        logger.info(
            "Capture started: %s (%d Hz, %d ch, %d frames/buffer)",
            self.device,
            self.sample_rate,
            self.channels,
            self.frames_per_buffer,
        )

    def pause(self) -> None:
        """Stop delivering audio to the ring buffer, keeping the stream open.

        The stream is left running and its frames discarded rather than stopped,
        because stopping and restarting a Bluetooth or USB endpoint frequently
        renegotiates the format and can fail outright.
        """
        self._paused.set()
        logger.info("Capture paused: %s", self.device.display_name)

    def resume(self) -> None:
        self._paused.clear()
        with self._stats_lock:
            self._stats.last_callback_monotonic = time.monotonic()
        logger.info("Capture resumed: %s", self.device.display_name)

    def stop(self) -> None:
        """Stop and release the stream. Safe to call more than once."""
        if not self._running and self._stream is None:
            return
        self._running = False
        self._close_stream()
        stats = self.stats
        logger.info(
            "Capture stopped: %s (%d frames, %d dropped, %d overflows)",
            self.device.display_name,
            stats.frames_captured,
            stats.frames_dropped,
            stats.input_overflows,
        )

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            if stream.is_active():
                stream.stop_stream()
        except Exception as exc:
            logger.debug("stop_stream failed: %r", exc)
        try:
            stream.close()
        except Exception as exc:
            logger.debug("close failed: %r", exc)

    def __enter__(self) -> CaptureStream:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- the audio callback ----------------------------------------------

    def _callback(self, in_data, frame_count, time_info, status):  # noqa: ANN001
        """PortAudio callback. Runs on a realtime-ish thread — keep it tiny."""
        try:
            if status:
                with self._stats_lock:
                    self._stats.input_overflows += 1

            if self._paused.is_set():
                with self._stats_lock:
                    self._stats.last_callback_monotonic = time.monotonic()
                    self._stats.peak_level = 0.0
                return (None, pyaudio.paContinue)

            # Zero-copy view over PortAudio's buffer; RingBuffer.write copies.
            samples = np.frombuffer(in_data, dtype=np.float32)
            dropped = self.ring.write(samples)

            level = float(np.abs(samples).max()) if samples.size else 0.0

            with self._stats_lock:
                self._stats.frames_captured += frame_count
                self._stats.frames_dropped += dropped
                self._stats.callback_count += 1
                self._stats.last_callback_monotonic = time.monotonic()
                self._stats.peak_level = level

        except Exception:  # pragma: no cover - must never escape
            logger.exception("Audio callback error (suppressed to keep stream alive)")

        return (None, pyaudio.paContinue)

    # -- error reporting --------------------------------------------------

    def report_stall(self) -> None:
        """Notify the error callback that the device stopped responding."""
        if self._error_reported:
            return
        self._error_reported = True
        msg = (
            f"Il dispositivo audio «{self.device.display_name}» "
            "non è più disponibile."
        )
        logger.error("Stream stalled: %s", self.device)
        if self._on_error:
            try:
                self._on_error(msg)
            except Exception:
                logger.exception("on_error handler raised")

    def _friendly_open_error(self, exc: Exception) -> str:
        text = str(exc).lower()
        name = self.device.display_name
        if "invalid sample rate" in text or "invalid number of channels" in text:
            return (
                f"Il formato audio del dispositivo «{name}» non è supportato. "
                "Prova a cambiare la qualità del dispositivo nelle impostazioni audio di Windows."
            )
        if "device unavailable" in text or "invalid device" in text:
            return f"Il dispositivo «{name}» non è disponibile."
        if "access" in text or "denied" in text:
            return (
                f"Accesso negato al dispositivo «{name}». "
                "Controlla le impostazioni sulla privacy del microfono in Windows."
            )
        return f"Impossibile avviare la registrazione da «{name}»."


class CaptureWatchdog:
    """Polls capture streams and reports the ones that went silent-dead.

    A dedicated thread rather than a Qt timer, so the audio layer stays usable
    from scripts and tests with no GUI.
    """

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self._streams: list[CaptureStream] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def add(self, stream: CaptureStream) -> None:
        with self._lock:
            self._streams.append(stream)

    def remove(self, stream: CaptureStream) -> None:
        with self._lock:
            if stream in self._streams:
                self._streams.remove(stream)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="CaptureWatchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            with self._lock:
                streams = list(self._streams)
            for stream in streams:
                if stream.is_stalled():
                    stream.report_stall()
