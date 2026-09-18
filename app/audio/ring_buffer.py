"""Bounded circular audio buffer.

The single most important property for a multi-hour recording: memory use is
fixed at construction and never grows, whatever the consumer does. If the
consumer stalls, the oldest audio is dropped and counted — the buffer never
expands and the producer never blocks.

The producer is a PortAudio callback thread, so :meth:`RingBuffer.write` must
not allocate, must not do I/O, and must not raise. It does none of the three.
"""

from __future__ import annotations

import threading

import numpy as np

DTYPE = np.float32


class RingBuffer:
    """Fixed-capacity single-producer / single-consumer float32 ring buffer.

    Capacity is in frames (samples per channel). ``channels`` > 1 stores
    interleaved frames and every operation works in whole frames, so a partial
    frame can never be read.
    """

    def __init__(self, capacity_frames: int, channels: int = 1) -> None:
        if capacity_frames <= 0:
            raise ValueError("capacity_frames must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self._capacity = int(capacity_frames)
        self._channels = int(channels)
        self._buf = np.zeros(self._capacity * self._channels, dtype=DTYPE)
        self._lock = threading.Lock()

        self._write_pos = 0   # next frame index to write (mod capacity)
        self._available = 0   # frames currently readable
        self._dropped = 0     # frames lost to overflow, cumulative
        self._written = 0     # frames accepted, cumulative

    # -- introspection ----------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def dropped_frames(self) -> int:
        """Frames discarded because the consumer fell behind."""
        with self._lock:
            return self._dropped

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._written

    def available(self) -> int:
        """Frames ready to read."""
        with self._lock:
            return self._available

    def fill_ratio(self) -> float:
        with self._lock:
            return self._available / self._capacity

    def __len__(self) -> int:
        return self.available()

    # -- producer side ----------------------------------------------------

    def write(self, data: np.ndarray) -> int:
        """Append frames, overwriting the oldest data when full.

        ``data`` is either shape ``(frames,)`` for mono or ``(frames, channels)``
        / a flat interleaved array otherwise. Returns the number of frames
        dropped by this call (0 in the normal case).

        Never raises and never blocks: this runs on the audio callback thread.
        """
        flat = np.ascontiguousarray(data, dtype=DTYPE).reshape(-1)
        if flat.size == 0:
            return 0

        frames = flat.size // self._channels
        if frames == 0:
            return 0
        flat = flat[: frames * self._channels]

        with self._lock:
            dropped = 0

            # A burst larger than the whole buffer: keep only its newest tail.
            if frames >= self._capacity:
                keep = self._capacity
                flat = flat[-keep * self._channels :]
                dropped = frames - keep + self._available
                frames = keep
                self._buf[:] = flat
                self._write_pos = 0
                self._available = self._capacity
                self._dropped += dropped
                self._written += frames
                return dropped

            start = self._write_pos
            end = start + frames
            if end <= self._capacity:
                self._buf[start * self._channels : end * self._channels] = flat
            else:
                split = self._capacity - start
                self._buf[start * self._channels :] = flat[: split * self._channels]
                self._buf[: (frames - split) * self._channels] = flat[split * self._channels :]

            self._write_pos = end % self._capacity

            overflow = self._available + frames - self._capacity
            if overflow > 0:
                dropped = overflow
                self._available = self._capacity
                self._dropped += dropped
            else:
                self._available += frames

            self._written += frames
            return dropped

    # -- consumer side ----------------------------------------------------

    def read(self, frames: int) -> np.ndarray:
        """Remove and return up to ``frames`` frames, oldest first.

        Returns shape ``(n,)`` for mono, ``(n, channels)`` otherwise. May return
        fewer frames than requested, including an empty array.
        """
        if frames <= 0:
            return self._empty()

        with self._lock:
            n = min(frames, self._available)
            if n == 0:
                return self._empty()
            out = self._peek_locked(n)
            self._available -= n
            return out

    def read_all(self) -> np.ndarray:
        """Remove and return everything currently available."""
        with self._lock:
            n = self._available
            if n == 0:
                return self._empty()
            out = self._peek_locked(n)
            self._available -= n
            return out

    def peek(self, frames: int) -> np.ndarray:
        """Return up to ``frames`` frames without consuming them."""
        if frames <= 0:
            return self._empty()
        with self._lock:
            n = min(frames, self._available)
            return self._peek_locked(n) if n else self._empty()

    def discard(self, frames: int) -> int:
        """Drop up to ``frames`` of the oldest frames. Returns how many went."""
        if frames <= 0:
            return 0
        with self._lock:
            n = min(frames, self._available)
            self._available -= n
            return n

    def clear(self) -> None:
        """Drop all buffered audio. Counters are preserved."""
        with self._lock:
            self._available = 0

    def reset(self) -> None:
        """Drop all audio and zero the statistics."""
        with self._lock:
            self._available = 0
            self._write_pos = 0
            self._dropped = 0
            self._written = 0
            self._buf.fill(0.0)

    # -- internals --------------------------------------------------------

    def _empty(self) -> np.ndarray:
        shape = (0,) if self._channels == 1 else (0, self._channels)
        return np.zeros(shape, dtype=DTYPE)

    def _peek_locked(self, n: int) -> np.ndarray:
        """Copy the ``n`` oldest frames. Caller must hold the lock."""
        read_pos = (self._write_pos - self._available) % self._capacity
        end = read_pos + n
        if end <= self._capacity:
            out = self._buf[read_pos * self._channels : end * self._channels].copy()
        else:
            split = self._capacity - read_pos
            out = np.concatenate(
                (
                    self._buf[read_pos * self._channels :],
                    self._buf[: (n - split) * self._channels],
                )
            )
        return out if self._channels == 1 else out.reshape(-1, self._channels)
