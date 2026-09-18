"""Format conversion: device audio to what Whisper expects.

Whisper always wants **16 kHz mono float32 in [-1, 1]**. Devices deliver
anything: 44.1 or 48 kHz, mono or stereo (a headset here reports 1 channel, the
mic array and both loopbacks report 2), int16 or float32.

Resampling uses ``soxr`` (libsoxr bindings) rather than FFmpeg or scipy:

* it has a genuine **streaming** API that carries filter state across calls, so
  chunk boundaries produce no clicks — ``scipy.signal.resample_poly`` restarts
  its filter on every call and would;
* VHQ quality is well above what speech recognition needs, at roughly a
  microsecond per millisecond of audio;
* it is a small self-contained wheel with no external binary, which keeps the
  PyInstaller bundle simple.
"""

from __future__ import annotations

import logging

import numpy as np
import soxr

logger = logging.getLogger(__name__)

#: Sample rate every Whisper model is trained on.
TARGET_SAMPLE_RATE = 16_000

DTYPE = np.float32


def to_float32(data: np.ndarray) -> np.ndarray:
    """Convert PCM samples of any common integer dtype to float32 in [-1, 1]."""
    if data.dtype == np.float32:
        return data
    if data.dtype == np.float64:
        return data.astype(np.float32)
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:  # unsigned 8-bit PCM is centred on 128
        return (data.astype(np.float32) - 128.0) / 128.0
    return data.astype(np.float32)


def to_mono(data: np.ndarray, channels: int) -> np.ndarray:
    """Downmix interleaved audio to mono by averaging channels.

    Averaging rather than taking channel 0 matters for real material: music and
    video routinely put dialogue in one channel, and hard-panned speech would be
    silent on a single-channel pick.
    """
    if channels <= 1:
        return np.ascontiguousarray(data, dtype=DTYPE).reshape(-1)

    flat = np.ascontiguousarray(data, dtype=DTYPE).reshape(-1)
    frames = flat.size // channels
    if frames == 0:
        return np.zeros(0, dtype=DTYPE)
    return flat[: frames * channels].reshape(frames, channels).mean(axis=1)


def peak_level(data: np.ndarray) -> float:
    """Peak absolute amplitude, for the level meter."""
    if data.size == 0:
        return 0.0
    return float(np.abs(data).max())


def rms_level(data: np.ndarray) -> float:
    """Root-mean-square amplitude, for silence checks."""
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))


def rms_dbfs(data: np.ndarray) -> float:
    """RMS expressed in dBFS. Digital silence returns -inf."""
    rms = rms_level(data)
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(rms))


class StreamResampler:
    """Stateful converter: device format in, 16 kHz mono float32 out.

    One instance per capture source, used from one thread. Filter state is kept
    between :meth:`process` calls, so feeding audio in chunks gives bit-identical
    output to feeding it all at once (within the resampler's own latency).
    """

    def __init__(
        self,
        source_rate: int,
        channels: int,
        target_rate: int = TARGET_SAMPLE_RATE,
        quality: str = "VHQ",
    ) -> None:
        if source_rate <= 0:
            raise ValueError("source_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self.channels = int(channels)
        self.quality = quality

        self._needs_resample = self.source_rate != self.target_rate
        self._stream: soxr.ResampleStream | None = None
        if self._needs_resample:
            self._stream = soxr.ResampleStream(
                self.source_rate,
                self.target_rate,
                num_channels=1,  # downmix happens before resampling
                dtype="float32",
                quality=quality,
            )
        logger.debug(
            "StreamResampler %d Hz x%d -> %d Hz (%s)",
            self.source_rate,
            self.channels,
            self.target_rate,
            "resampling" if self._needs_resample else "passthrough",
        )

    @property
    def ratio(self) -> float:
        return self.target_rate / self.source_rate

    def process(self, data: np.ndarray, last: bool = False) -> np.ndarray:
        """Convert one chunk of device audio to 16 kHz mono float32.

        ``last=True`` flushes the resampler's internal delay line; use it once,
        when the stream stops, to avoid losing the final few milliseconds.
        """
        if data.size == 0 and not last:
            return np.zeros(0, dtype=DTYPE)

        mono = to_mono(to_float32(data), self.channels)

        if not self._needs_resample:
            return np.ascontiguousarray(mono, dtype=DTYPE)

        assert self._stream is not None
        out = self._stream.resample_chunk(mono, last=last)
        return np.ascontiguousarray(out, dtype=DTYPE)

    def flush(self) -> np.ndarray:
        """Drain any audio still held in the resampler's delay line."""
        if not self._needs_resample:
            return np.zeros(0, dtype=DTYPE)
        return self.process(np.zeros(0, dtype=DTYPE), last=True)

    def reset(self) -> None:
        """Discard filter state, e.g. after a pause or a device change."""
        if self._needs_resample:
            self._stream = soxr.ResampleStream(
                self.source_rate,
                self.target_rate,
                num_channels=1,
                dtype="float32",
                quality=self.quality,
            )


def resample_array(
    data: np.ndarray,
    source_rate: int,
    target_rate: int = TARGET_SAMPLE_RATE,
    channels: int = 1,
    quality: str = "VHQ",
) -> np.ndarray:
    """One-shot conversion of a complete array. For files and tests."""
    mono = to_mono(to_float32(data), channels)
    if source_rate == target_rate:
        return np.ascontiguousarray(mono, dtype=DTYPE)
    out = soxr.resample(mono, source_rate, target_rate, quality=quality)
    return np.ascontiguousarray(out, dtype=DTYPE)
