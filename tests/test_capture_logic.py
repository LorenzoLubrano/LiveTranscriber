"""Capture logic that must not need real hardware to verify.

Device loss cannot be provoked on a test machine — you would have to physically
unplug something mid-run — so the detection logic is exercised directly against
a stream whose callback clock is controlled by the test.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.audio.capture import STALL_TIMEOUT_S, CaptureStats, CaptureStream, CaptureWatchdog
from app.audio.devices import DeviceKind
from app.audio.ring_buffer import RingBuffer
from tests.test_devices import make_device


@pytest.fixture
def stream():
    """A CaptureStream with no PortAudio stream behind it."""
    device = make_device("Fake Mic", DeviceKind.MICROPHONE, channels=2)
    return CaptureStream(device, RingBuffer(48000, 2))


# -- callback --------------------------------------------------------------

def test_callback_moves_audio_into_the_ring(stream):
    payload = np.ones(2048, dtype=np.float32).tobytes()
    stream._callback(payload, 1024, None, 0)

    assert stream.ring.available() == 1024
    assert stream.stats.frames_captured == 1024
    assert stream.stats.callback_count == 1
    assert stream.level == pytest.approx(1.0)


def test_callback_never_raises_on_malformed_input(stream):
    """An exception escaping a PortAudio callback kills the stream."""
    result = stream._callback(b"\x00\x01\x02", 999999, None, 0)
    assert result[1] == 0  # paContinue


def test_callback_counts_overflow_status(stream):
    stream._callback(np.zeros(2048, dtype=np.float32).tobytes(), 1024, None, 2)
    assert stream.stats.input_overflows == 1


def test_callback_reports_dropped_frames_from_a_full_ring():
    device = make_device("Fake", DeviceKind.MICROPHONE, channels=1)
    small = RingBuffer(100, 1)
    stream = CaptureStream(device, small)

    stream._callback(np.zeros(150, dtype=np.float32).tobytes(), 150, None, 0)
    assert stream.stats.frames_dropped == 50


# -- pause -----------------------------------------------------------------

def test_paused_callback_discards_audio(stream):
    stream.pause()
    stream._callback(np.ones(2048, dtype=np.float32).tobytes(), 1024, None, 0)

    assert stream.ring.available() == 0
    assert stream.level == 0.0
    assert stream.is_paused


def test_resume_lets_audio_through_again(stream):
    stream.pause()
    stream._callback(np.ones(2048, dtype=np.float32).tobytes(), 1024, None, 0)
    stream.resume()
    stream._callback(np.ones(2048, dtype=np.float32).tobytes(), 1024, None, 0)

    assert stream.ring.available() == 1024
    assert not stream.is_paused


# -- device loss -----------------------------------------------------------

def test_a_stream_that_is_not_running_is_never_stalled(stream):
    assert not stream.is_stalled()


def test_stall_detected_after_the_timeout(stream):
    stream._running = True
    stream._stats.last_callback_monotonic = time.monotonic() - (STALL_TIMEOUT_S + 1)
    assert stream.is_stalled()


def test_no_stall_while_callbacks_keep_arriving(stream):
    stream._running = True
    stream._stats.last_callback_monotonic = time.monotonic()
    assert not stream.is_stalled()


def test_paused_stream_is_not_reported_as_stalled(stream):
    """Pause stops callbacks from counting; that is not a dead device."""
    stream._running = True
    stream.pause()
    stream._stats.last_callback_monotonic = time.monotonic() - 3600
    assert not stream.is_stalled()


def test_stall_is_reported_once_with_a_user_facing_message():
    messages: list[str] = []
    device = make_device("Cuffie Bluetooth", DeviceKind.MICROPHONE)
    stream = CaptureStream(device, RingBuffer(1000, 2), on_error=messages.append)

    stream.report_stall()
    stream.report_stall()

    assert len(messages) == 1, "the user must not be told twice"
    assert "Cuffie Bluetooth" in messages[0]
    assert "Traceback" not in messages[0]
    assert "\n" not in messages[0]


def test_a_failing_error_handler_does_not_propagate():
    def explode(_: str) -> None:
        raise RuntimeError("handler is broken")

    stream = CaptureStream(make_device("X"), RingBuffer(1000, 2), on_error=explode)
    stream.report_stall()  # must not raise


# -- watchdog --------------------------------------------------------------

def test_watchdog_reports_a_stalled_stream():
    messages: list[str] = []
    stream = CaptureStream(make_device("Gone"), RingBuffer(1000, 2), on_error=messages.append)
    stream._running = True
    stream._stats.last_callback_monotonic = time.monotonic() - (STALL_TIMEOUT_S + 1)

    watchdog = CaptureWatchdog(interval_s=0.05)
    watchdog.add(stream)
    watchdog.start()
    try:
        deadline = time.monotonic() + 3.0
        while not messages and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        watchdog.stop()

    assert messages, "watchdog did not report a stalled stream"


def test_watchdog_leaves_healthy_streams_alone():
    messages: list[str] = []
    stream = CaptureStream(make_device("Fine"), RingBuffer(1000, 2), on_error=messages.append)
    stream._running = True
    stream._stats.last_callback_monotonic = time.monotonic()

    watchdog = CaptureWatchdog(interval_s=0.05)
    watchdog.add(stream)
    watchdog.start()
    time.sleep(0.4)
    watchdog.stop()

    assert not messages


def test_watchdog_stops_cleanly():
    watchdog = CaptureWatchdog(interval_s=0.05)
    watchdog.start()
    watchdog.stop()
    assert watchdog._thread is None


def test_removed_stream_is_no_longer_watched():
    stream = CaptureStream(make_device("Y"), RingBuffer(1000, 2))
    watchdog = CaptureWatchdog()
    watchdog.add(stream)
    watchdog.remove(stream)
    assert stream not in watchdog._streams


# -- error messages --------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_fragment",
    [
        ("Invalid sample rate", "formato audio"),
        ("Device unavailable", "non è disponibile"),
        ("Access denied", "privacy"),
        ("something entirely unexpected", "Impossibile avviare"),
    ],
)
def test_open_errors_are_translated_for_users(stream, raw, expected_fragment):
    """Spec §18: never show a Python traceback to a normal user."""
    message = stream._friendly_open_error(Exception(raw))
    assert expected_fragment in message
    assert "Traceback" not in message
    assert "Exception" not in message


def test_stats_snapshot_is_a_copy(stream):
    first = stream.stats
    stream._callback(np.zeros(2048, dtype=np.float32).tobytes(), 1024, None, 0)
    assert first.frames_captured == 0, "stats must be a snapshot, not a live view"
    assert stream.stats.frames_captured == 1024


def test_dropped_ratio_handles_a_fresh_stream():
    assert CaptureStats().dropped_ratio == 0.0
