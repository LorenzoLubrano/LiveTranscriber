"""Microphone capture via WASAPI.

Nothing here is loopback-specific; the difference from :mod:`app.audio.loopback`
is only which device is resolved. Channel counts vary in practice — the mic
array on this machine reports 2 channels while a headset mic reports 1 — so the
stream always opens with the device's own count and downmixing happens later.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.audio.capture import CaptureStream
from app.audio.devices import (
    AudioDevice,
    DeviceError,
    DeviceKind,
    default_microphone,
)
from app.audio.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_SECONDS = 30.0


def resolve_microphone_device(preferred: AudioDevice | None = None) -> AudioDevice:
    """Pick the microphone to record from, falling back to the system default."""
    if preferred is not None and preferred.kind is DeviceKind.MICROPHONE:
        return preferred

    device = default_microphone()
    if device is None:
        raise DeviceError(
            "Nessun microfono disponibile. Collega un microfono e riprova.",
            "no WASAPI input devices enumerated",
        )
    return device


def create_microphone_capture(
    device: AudioDevice | None = None,
    buffer_seconds: float = DEFAULT_BUFFER_SECONDS,
    on_error: Callable[[str], None] | None = None,
) -> tuple[CaptureStream, RingBuffer]:
    """Build a ready-to-start microphone capture and its ring buffer."""
    target = resolve_microphone_device(device)
    ring = RingBuffer(
        capacity_frames=int(target.default_sample_rate * buffer_seconds),
        channels=target.capture_channels,
    )
    stream = CaptureStream(target, ring, on_error=on_error)
    logger.info("Microphone capture prepared for %s", target)
    return stream, ring
