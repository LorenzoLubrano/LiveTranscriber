"""Crash-safe streaming WAV recording.

Python's :mod:`wave` module patches the RIFF and ``data`` chunk sizes only when
the file is closed. If the process dies — a power cut, a GPU driver fault, Task
Manager — the header still says zero bytes and a multi-hour recording reads as
an empty file. Since "don't lose the whole recording on a crash" is a hard
requirement (spec §11, §14), the writer here owns the header and rewrites the
two size fields every few seconds, then flushes. A crash costs at most the last
flush interval, and whatever was written before it opens normally in any player.

Audio is stored as **16 kHz mono 16-bit PCM**, the same stream that feeds
Whisper. One conversion path serves both recording and transcription, which
removes a whole class of drift bugs, keeps a four-hour session near 460 MB
across all three tracks, and stays fully intelligible for speech. (The rate is a
constructor argument, so a higher-fidelity option can be added later without
touching callers.)
"""

from __future__ import annotations

import contextlib
import logging
import struct
import threading
import time
from pathlib import Path

import numpy as np

from app.audio.resampler import TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)

_HEADER_SIZE = 44
_RIFF_SIZE_OFFSET = 4
_DATA_SIZE_OFFSET = 40

#: How often header sizes are rewritten and the file flushed to disk.
DEFAULT_FLUSH_INTERVAL_S = 5.0


def float32_to_int16(samples: np.ndarray) -> np.ndarray:
    """Convert float32 in [-1, 1] to int16, clipping rather than wrapping.

    Without the clip, a sample of 1.2 wraps to a large negative value and
    produces an audible crack instead of mild clipping.
    """
    if samples.size == 0:
        return np.zeros(0, dtype=np.int16)
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


class WavWriter:
    """Incremental 16-bit PCM WAV writer with a self-healing header."""

    def __init__(
        self,
        path: Path | str,
        sample_rate: int = TARGET_SAMPLE_RATE,
        channels: int = 1,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
    ) -> None:
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.flush_interval_s = float(flush_interval_s)

        self._lock = threading.Lock()
        self._file = None
        self._data_bytes = 0
        self._last_flush = 0.0
        self._closed = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open()

    # -- header -----------------------------------------------------------

    def _header(self, data_bytes: int) -> bytes:
        bits = 16
        byte_rate = self.sample_rate * self.channels * bits // 8
        block_align = self.channels * bits // 8
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_bytes,
            b"WAVE",
            b"fmt ",
            16,          # PCM fmt chunk size
            1,           # PCM
            self.channels,
            self.sample_rate,
            byte_rate,
            block_align,
            bits,
            b"data",
            data_bytes,
        )

    def _open(self) -> None:
        # Deliberately long-lived: the writer streams for the whole session
        # and owns the handle until close(). A context manager here would
        # defeat the purpose.
        self._file = open(self.path, "wb")  # noqa: SIM115
        self._file.write(self._header(0))
        self._file.flush()
        self._last_flush = time.monotonic()

    # -- writing ----------------------------------------------------------

    def write(self, samples: np.ndarray) -> int:
        """Append float32 (or int16) samples. Returns bytes written."""
        if self._closed or samples.size == 0:
            return 0

        pcm = samples if samples.dtype == np.int16 else float32_to_int16(samples)
        payload = pcm.tobytes()

        with self._lock:
            if self._file is None:
                return 0
            try:
                self._file.write(payload)
                self._data_bytes += len(payload)
                if (time.monotonic() - self._last_flush) >= self.flush_interval_s:
                    self._sync_locked()
            except OSError:
                logger.exception("WAV write failed: %s", self.path.name)
                return 0
        return len(payload)

    def _sync_locked(self) -> None:
        """Rewrite the size fields and flush. Caller holds the lock."""
        if self._file is None:
            return
        end = self._file.tell()
        self._file.seek(_RIFF_SIZE_OFFSET)
        self._file.write(struct.pack("<I", 36 + self._data_bytes))
        self._file.seek(_DATA_SIZE_OFFSET)
        self._file.write(struct.pack("<I", self._data_bytes))
        self._file.seek(end)
        self._file.flush()
        self._last_flush = time.monotonic()

    def sync(self) -> None:
        """Force a header update and flush."""
        with self._lock:
            if not self._closed:
                self._sync_locked()

    def close(self) -> None:
        """Finalise and close. Safe to call more than once."""
        with self._lock:
            if self._closed or self._file is None:
                self._closed = True
                return
            try:
                self._sync_locked()
            except OSError:
                logger.exception("Final WAV sync failed: %s", self.path.name)
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None
            self._closed = True
        logger.info(
            "WAV closed: %s (%.1f s, %.1f MB)",
            self.path.name,
            self.duration_seconds,
            self._data_bytes / 1_048_576,
        )

    # -- introspection ----------------------------------------------------

    @property
    def frames_written(self) -> int:
        return self._data_bytes // (2 * self.channels)

    @property
    def duration_seconds(self) -> float:
        return self.frames_written / self.sample_rate if self.sample_rate else 0.0

    @property
    def bytes_written(self) -> int:
        return self._data_bytes

    @property
    def is_closed(self) -> bool:
        return self._closed

    def __enter__(self) -> WavWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class MixRecorder:
    """Writes two sources plus a time-aligned mix of them.

    Alignment is by absolute sample index rather than arrival order: each source
    reports where its audio belongs on the session timeline, so if one stream
    drops frames the mix stays in sync instead of drifting for the rest of the
    recording. Audio is mixed once both sources have covered a region, or when a
    region falls far enough behind that waiting would stall the writer.
    """

    #: Mix a region once it is this far behind the leading source.
    _LAG_TOLERANCE_S = 2.0

    def __init__(
        self,
        pc_path: Path | str | None,
        mic_path: Path | str | None,
        mixed_path: Path | str | None,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self._lock = threading.Lock()

        self.pc = WavWriter(pc_path, sample_rate) if pc_path else None
        self.mic = WavWriter(mic_path, sample_rate) if mic_path else None
        self.mixed = (
            WavWriter(mixed_path, sample_rate)
            if mixed_path and self.pc and self.mic
            else None
        )

        # Pending audio per source, keyed by the absolute sample index it starts at.
        self._pending: dict[str, np.ndarray] = {"pc": np.zeros(0, np.float32),
                                                "mic": np.zeros(0, np.float32)}
        self._pending_start: dict[str, int] = {"pc": 0, "mic": 0}
        self._mixed_upto = 0

    def write_pc(self, samples: np.ndarray) -> None:
        if self.pc:
            self.pc.write(samples)
        self._offer("pc", samples)

    def write_mic(self, samples: np.ndarray) -> None:
        if self.mic:
            self.mic.write(samples)
        self._offer("mic", samples)

    def _offer(self, source: str, samples: np.ndarray) -> None:
        if self.mixed is None or samples.size == 0:
            return
        with self._lock:
            self._pending[source] = np.concatenate(
                (self._pending[source], np.asarray(samples, dtype=np.float32))
            )
            self._drain_locked()

    def _end(self, source: str) -> int:
        return self._pending_start[source] + self._pending[source].size

    def _drain_locked(self) -> None:
        """Emit every region both sources have covered."""
        assert self.mixed is not None
        pc_end, mic_end = self._end("pc"), self._end("mic")

        target = min(pc_end, mic_end)
        # If one source is far behind, stop waiting and mix what we have.
        lag = int(self._LAG_TOLERANCE_S * self.sample_rate)
        if max(pc_end, mic_end) - target > lag:
            target = max(pc_end, mic_end) - lag

        if target <= self._mixed_upto:
            return

        length = target - self._mixed_upto
        mix = np.zeros(length, dtype=np.float32)
        for source in ("pc", "mic"):
            mix += self._slice_locked(source, self._mixed_upto, target, length)

        # Both sources at full scale can exceed [-1, 1]; halve rather than clip
        # so a loud lecture plus a loud question stays undistorted.
        self.mixed.write(mix * 0.5)
        self._mixed_upto = target
        self._trim_locked()

    def _slice_locked(self, source: str, start: int, stop: int, length: int) -> np.ndarray:
        """Audio of ``source`` over [start, stop), zero-padded where missing."""
        buf = self._pending[source]
        base = self._pending_start[source]
        out = np.zeros(length, dtype=np.float32)
        lo = max(start, base)
        hi = min(stop, base + buf.size)
        if hi > lo:
            out[lo - start : hi - start] = buf[lo - base : hi - base]
        return out

    def _trim_locked(self) -> None:
        """Release audio that has already been mixed."""
        for source in ("pc", "mic"):
            base = self._pending_start[source]
            consumed = self._mixed_upto - base
            if consumed > 0:
                buf = self._pending[source]
                if consumed >= buf.size:
                    self._pending[source] = np.zeros(0, np.float32)
                    self._pending_start[source] = base + buf.size
                else:
                    self._pending[source] = buf[consumed:]
                    self._pending_start[source] = self._mixed_upto

    def sync(self) -> None:
        for writer in (self.pc, self.mic, self.mixed):
            if writer:
                writer.sync()

    def close(self) -> None:
        with self._lock:
            if self.mixed is not None:
                # Flush the tail: mix everything either source still holds.
                target = max(self._end("pc"), self._end("mic"))
                if target > self._mixed_upto:
                    length = target - self._mixed_upto
                    mix = np.zeros(length, dtype=np.float32)
                    for source in ("pc", "mic"):
                        mix += self._slice_locked(source, self._mixed_upto, target, length)
                    self.mixed.write(mix * 0.5)
                    self._mixed_upto = target
        for writer in (self.pc, self.mic, self.mixed):
            if writer:
                writer.close()

    @property
    def paths(self) -> dict[str, Path]:
        out: dict[str, Path] = {}
        if self.pc:
            out["pc"] = self.pc.path
        if self.mic:
            out["mic"] = self.mic.path
        if self.mixed:
            out["mixed"] = self.mixed.path
        return out

    def __enter__(self) -> MixRecorder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
