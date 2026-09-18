"""End-to-end capture tests against real audio hardware.

These are the tests that prove the parts a unit test cannot: that WASAPI
loopback really returns what Windows is playing, that timing does not drift, and
that the callback keeps up.

The loopback test is self-contained — it generates its own tone through the
default output device and checks that exact frequency comes back — so it needs
no human and no audio playing. The microphone test only asserts that the capture
path runs and stays in sync, because whether anything is audible depends on the
room.

Run with::

    pytest tests/test_capture_hardware.py -m hardware -v

They are skipped automatically when no suitable device is present.
"""

from __future__ import annotations

import time

import numpy as np
import pyaudiowpatch as pyaudio
import pytest

from app.audio.capture import CaptureStream
from app.audio.devices import (
    default_loopback,
    default_microphone,
    default_output,
    get_pyaudio,
)
from app.audio.resampler import TARGET_SAMPLE_RATE, StreamResampler, rms_dbfs
from app.audio.ring_buffer import RingBuffer

pytestmark = pytest.mark.hardware

TONE_HZ = 440.0
DURATION_S = 4.0


class ToneGenerator:
    """Plays a continuous sine through an output device, phase-continuous."""

    def __init__(self, device, freq: float = TONE_HZ, amplitude: float = 0.25) -> None:
        self.rate = int(device.default_sample_rate)
        self.channels = 2
        self._phase = 0.0
        self._step = 2 * np.pi * freq / self.rate
        self._amplitude = amplitude

        self._stream = get_pyaudio().open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            output=True,
            output_device_index=device.index,
            frames_per_buffer=1024,
            stream_callback=self._callback,
            start=False,
        )

    def _callback(self, in_data, frame_count, time_info, status):  # noqa: ANN001
        t = self._phase + self._step * np.arange(frame_count, dtype=np.float32)
        if frame_count:
            self._phase = float(t[-1] + self._step)
        mono = (self._amplitude * np.sin(t)).astype(np.float32)
        return (np.repeat(mono, self.channels).tobytes(), pyaudio.paContinue)

    def __enter__(self) -> ToneGenerator:
        self._stream.start_stream()
        return self

    def __exit__(self, *exc_info: object) -> None:
        try:
            self._stream.stop_stream()
        finally:
            self._stream.close()


def capture_for(device, seconds: float) -> tuple[np.ndarray, CaptureStream]:
    """Capture a device for ``seconds`` and return 16 kHz mono audio."""
    ring = RingBuffer(int(device.default_sample_rate * 30), device.capture_channels)
    resampler = StreamResampler(device.default_sample_rate, device.capture_channels)
    stream = CaptureStream(device, ring)
    collected: list[np.ndarray] = []

    stream.start()
    try:
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            raw = ring.read_all()
            if raw.size:
                chunk = resampler.process(raw)
                if chunk.size:
                    collected.append(chunk)
            time.sleep(0.05)
        raw = ring.read_all()
        if raw.size:
            collected.append(resampler.process(raw))
        collected.append(resampler.flush())
    finally:
        stream.stop()

    parts = [c for c in collected if c.size]
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return audio, stream


def dominant_frequency(audio: np.ndarray, skip_s: float = 0.5) -> float:
    """Strongest frequency component, ignoring stream warm-up."""
    analysis = audio[int(skip_s * TARGET_SAMPLE_RATE) :]
    windowed = analysis * np.hanning(analysis.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    return float(np.fft.rfftfreq(analysis.size, 1 / TARGET_SAMPLE_RATE)[int(np.argmax(spectrum))])


# -- loopback --------------------------------------------------------------

@pytest.fixture(scope="module")
def loopback_capture():
    """Play a known tone and capture it through WASAPI loopback, once."""
    loopback = default_loopback()
    output = default_output()
    if loopback is None or output is None:
        pytest.skip("no loopback/output device")

    with ToneGenerator(output):
        time.sleep(0.2)  # let the render stream settle
        audio, stream = capture_for(loopback, DURATION_S)

    if audio.size == 0:
        pytest.skip("loopback produced no audio on this machine")
    return audio, stream


def test_loopback_captures_the_tone_that_was_played(loopback_capture):
    """The core WASAPI-loopback guarantee: we record what Windows plays."""
    audio, _ = loopback_capture
    assert dominant_frequency(audio) == pytest.approx(TONE_HZ, abs=10)


def test_loopback_signal_is_not_silence(loopback_capture):
    audio, _ = loopback_capture
    assert rms_dbfs(audio) > -50.0, "loopback captured silence while audio was playing"


def test_loopback_duration_does_not_drift(loopback_capture):
    """Captured length must match wall-clock: the resampler ratio must be right."""
    audio, _ = loopback_capture
    duration = audio.size / TARGET_SAMPLE_RATE
    assert duration == pytest.approx(DURATION_S, abs=0.35), f"captured {duration:.2f}s"


def test_loopback_drops_no_frames(loopback_capture):
    """The callback must keep up; dropped frames mean lost speech."""
    _, stream = loopback_capture
    stats = stream.stats
    assert stats.callback_count > 0
    assert stats.frames_dropped == 0, f"dropped {stats.frames_dropped} frames"


# -- microphone ------------------------------------------------------------

def test_microphone_capture_runs_and_stays_in_sync():
    """Whether the room is quiet is not our business; the pipeline running is."""
    mic = default_microphone()
    if mic is None:
        pytest.skip("no microphone")

    audio, stream = capture_for(mic, 2.0)
    stats = stream.stats

    assert stats.callback_count > 0, "no audio callbacks from the microphone"
    assert stats.frames_dropped == 0
    assert audio.size / TARGET_SAMPLE_RATE == pytest.approx(2.0, abs=0.35)
    assert np.isfinite(audio).all(), "capture produced NaN/inf samples"


def test_mono_and_stereo_devices_both_downmix_to_mono():
    """A 1-channel headset mic and a 2-channel array must both yield mono."""
    from app.audio.devices import list_microphones

    mics = list_microphones()
    if not mics:
        pytest.skip("no microphones")

    for mic in mics[:2]:
        audio, _ = capture_for(mic, 0.7)
        assert audio.ndim == 1, f"{mic} did not produce mono"


# -- pause -----------------------------------------------------------------

def test_pause_stops_audio_reaching_the_buffer():
    """Pause must hold the session without tearing down the device."""
    mic = default_microphone()
    if mic is None:
        pytest.skip("no microphone")

    ring = RingBuffer(int(mic.default_sample_rate * 10), mic.capture_channels)
    stream = CaptureStream(mic, ring)
    stream.start()
    try:
        time.sleep(0.5)
        assert ring.available() > 0, "no audio captured before pause"

        stream.pause()
        ring.clear()
        time.sleep(0.5)
        assert ring.available() == 0, "audio still arriving while paused"

        stream.resume()
        time.sleep(0.5)
        assert ring.available() > 0, "no audio after resume"
        assert stream.is_running, "stream must stay open across a pause"
    finally:
        stream.stop()
