"""System-audio capture via WASAPI loopback.

Loopback taps the render endpoint itself, so it records exactly what Windows is
sending to the speakers — at full quality, with no room noise, and without the
microphone hearing anything. This is why the app never "listens to the speakers"
to capture PC audio.

Capturing from headphones works identically to speakers: the loopback device
mirrors whatever endpoint is selected, including Bluetooth.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.audio.capture import CaptureStream
from app.audio.devices import (
    AudioDevice,
    DeviceError,
    DeviceKind,
    default_loopback,
    list_loopbacks,
    loopback_for_output,
)
from app.audio.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)

#: Seconds of audio held in RAM per source before the oldest is dropped.
#: The consumer normally drains this within milliseconds; the depth exists to
#: absorb a slow Whisper window, not to store the recording.
DEFAULT_BUFFER_SECONDS = 30.0


def resolve_loopback_device(
    preferred: AudioDevice | None = None,
) -> AudioDevice:
    """Pick the loopback device to record from.

    Accepts either a loopback device or a render endpoint (in which case its
    loopback twin is looked up).
    """
    if preferred is not None:
        if preferred.kind is DeviceKind.LOOPBACK:
            return preferred
        if preferred.kind is DeviceKind.OUTPUT:
            twin = loopback_for_output(preferred)
            if twin is not None:
                return twin
            raise DeviceError(
                f"Non è possibile registrare l'audio da «{preferred.display_name}».",
                f"no loopback twin for output {preferred}",
            )

    device = default_loopback()
    if device is None:
        available = len(list_loopbacks())
        raise DeviceError(
            "Nessun dispositivo di riproduzione disponibile per la registrazione "
            "dell'audio del PC. Verifica che siano collegati altoparlanti o cuffie.",
            f"no WASAPI loopback devices (enumerated {available})",
        )
    return device


def create_loopback_capture(
    device: AudioDevice | None = None,
    buffer_seconds: float = DEFAULT_BUFFER_SECONDS,
    on_error: Callable[[str], None] | None = None,
) -> tuple[CaptureStream, RingBuffer]:
    """Build a ready-to-start loopback capture and its ring buffer."""
    target = resolve_loopback_device(device)
    ring = RingBuffer(
        capacity_frames=int(target.default_sample_rate * buffer_seconds),
        channels=target.capture_channels,
    )
    stream = CaptureStream(target, ring, on_error=on_error)
    logger.info("Loopback capture prepared for %s", target)
    return stream, ring
