"""Audio device enumeration.

Windows exposes the same physical hardware through several PortAudio host APIs.
Measured on the development machine:

======  ===  ==============================  ===========================
index   api  reported                        reality
======  ===  ==============================  ===========================
1       MME  ``Microphone Array (Realtek(R) Au``  name truncated at 31 chars,
                                             sample rate reported as 44100
7       DS   ``Microphone Array (Realtek(R) Audio)``  44100, resampled view
14      WAS  ``Microphone Array (Realtek(R) Audio)``  48000, the real endpoint
======  ===  ==============================  ===========================

MME truncates device names to 31 characters and both MME and DirectSound
advertise a fictional 44.1 kHz for hardware actually running at 48 kHz. WASAPI
reports the true shared-mode format and is the only host API that exposes
loopback devices. So WASAPI is the source of truth; the other host APIs are
offered only as a fallback if WASAPI is unavailable.

Devices are addressed by a stable :attr:`AudioDevice.key` rather than by
PortAudio index, because indices shift whenever a device is plugged, unplugged
or the default endpoint changes.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum

import pyaudiowpatch as pyaudio

logger = logging.getLogger(__name__)

_LOOPBACK_SUFFIX = re.compile(r"\s*\[Loopback\]\s*$", re.IGNORECASE)


class DeviceKind(StrEnum):
    """What a device can be used for in this application."""

    MICROPHONE = "microphone"
    LOOPBACK = "loopback"
    OUTPUT = "output"


@dataclass(frozen=True)
class AudioDevice:
    """A capturable (or selectable output) audio endpoint."""

    index: int
    name: str
    kind: DeviceKind
    host_api_index: int
    host_api_name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: int
    default_low_latency: float = 0.0
    is_default: bool = False
    is_wasapi: bool = True
    #: Set for loopback devices: the name of the render endpoint they mirror.
    render_endpoint: str = ""

    #: Stable identity, survives index reshuffling across sessions.
    key: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not self.key:
            object.__setattr__(self, "key", self.compute_key())

    def compute_key(self) -> str:
        channels = self.max_input_channels or self.max_output_channels
        return f"{self.host_api_name}|{self.kind.value}|{self.name}|{channels}"

    @property
    def display_name(self) -> str:
        """Plain endpoint name, without the ``[Loopback]`` marker.

        Deliberately carries no "(default)" decoration: this string is embedded
        in user-facing error messages, where «Speakers (default)» would read as
        part of the device's name.
        """
        return _LOOPBACK_SUFFIX.sub("", self.name).strip()

    @property
    def label(self) -> str:
        """Name for selection lists, marking the system default."""
        return f"{self.display_name} (predefinito)" if self.is_default else self.display_name

    @property
    def capture_channels(self) -> int:
        """Channel count to open the stream with."""
        return max(1, self.max_input_channels)

    def __str__(self) -> str:
        return f"{self.display_name} [{self.host_api_name} #{self.index} @{self.default_sample_rate}Hz]"


class DeviceError(RuntimeError):
    """Audio subsystem failure, with a message safe to show to a user."""

    def __init__(self, user_message: str, technical: str = "") -> None:
        super().__init__(technical or user_message)
        self.user_message = user_message
        self.technical = technical or user_message


# --------------------------------------------------------------------------
# PyAudio instance management
# --------------------------------------------------------------------------
# Creating and terminating PyAudio re-initialises COM and re-scans every
# endpoint; doing that per query is slow and occasionally races with Windows
# device-change notifications. A single guarded instance is reused instead.

_pa_lock = threading.RLock()
_pa_instance: pyaudio.PyAudio | None = None


def get_pyaudio() -> pyaudio.PyAudio:
    """Return the shared PyAudio instance, creating it on first use."""
    global _pa_instance
    with _pa_lock:
        if _pa_instance is None:
            try:
                _pa_instance = pyaudio.PyAudio()
            except Exception as exc:  # pragma: no cover - platform failure
                raise DeviceError(
                    "Impossibile inizializzare il sistema audio di Windows.",
                    f"PyAudio init failed: {exc!r}",
                ) from exc
        return _pa_instance


def terminate_pyaudio() -> None:
    """Release the shared PyAudio instance. Call on application shutdown."""
    global _pa_instance
    with _pa_lock:
        if _pa_instance is not None:
            try:
                _pa_instance.terminate()
            except Exception as exc:  # pragma: no cover
                logger.warning("PyAudio terminate failed: %r", exc)
            _pa_instance = None


def refresh() -> None:
    """Force a full re-scan of devices.

    PortAudio caches its device list at initialisation, so a device plugged in
    after startup is invisible until the library is re-initialised.
    """
    terminate_pyaudio()
    get_pyaudio()


@contextmanager
def _audio() -> Iterator[pyaudio.PyAudio]:
    with _pa_lock:
        yield get_pyaudio()


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------


def _host_api_name(pa: pyaudio.PyAudio, index: int) -> str:
    try:
        return str(pa.get_host_api_info_by_index(index)["name"])
    except Exception:
        return f"api{index}"


def _wasapi_info(pa: pyaudio.PyAudio) -> dict | None:
    try:
        return dict(pa.get_host_api_info_by_type(pyaudio.paWASAPI))
    except Exception as exc:
        logger.warning("WASAPI host API unavailable: %r", exc)
        return None


def _build(
    pa: pyaudio.PyAudio,
    info: dict,
    kind: DeviceKind,
    is_default: bool,
    wasapi_index: int | None,
) -> AudioDevice:
    name = str(info["name"])
    render = _LOOPBACK_SUFFIX.sub("", name).strip() if kind is DeviceKind.LOOPBACK else ""
    return AudioDevice(
        index=int(info["index"]),
        name=name,
        kind=kind,
        host_api_index=int(info["hostApi"]),
        host_api_name=_host_api_name(pa, int(info["hostApi"])),
        max_input_channels=int(info["maxInputChannels"]),
        max_output_channels=int(info["maxOutputChannels"]),
        default_sample_rate=int(round(float(info["defaultSampleRate"]))),
        default_low_latency=float(info.get("defaultLowInputLatency", 0.0) or 0.0),
        is_default=is_default,
        is_wasapi=int(info["hostApi"]) == wasapi_index,
        render_endpoint=render,
    )


def _dedupe(devices: list[AudioDevice]) -> list[AudioDevice]:
    """Drop duplicate endpoints, keeping the first (WASAPI-preferred) entry.

    Windows commonly reports the same endpoint more than once — for example an
    exclusive-mode and a shared-mode view of one device. Identity is the stable
    key; a later duplicate that happens to be the system default promotes the
    entry that was already kept.
    """
    seen: dict[str, AudioDevice] = {}
    order: list[str] = []
    for dev in devices:
        existing = seen.get(dev.key)
        if existing is None:
            seen[dev.key] = dev
            order.append(dev.key)
        elif dev.is_default and not existing.is_default:
            seen[dev.key] = existing_with_default(existing)
    return [seen[k] for k in order]


def existing_with_default(dev: AudioDevice) -> AudioDevice:
    """Copy of ``dev`` marked as the system default."""
    return AudioDevice(
        index=dev.index,
        name=dev.name,
        kind=dev.kind,
        host_api_index=dev.host_api_index,
        host_api_name=dev.host_api_name,
        max_input_channels=dev.max_input_channels,
        max_output_channels=dev.max_output_channels,
        default_sample_rate=dev.default_sample_rate,
        default_low_latency=dev.default_low_latency,
        is_default=True,
        is_wasapi=dev.is_wasapi,
        render_endpoint=dev.render_endpoint,
        key=dev.key,
    )


def list_microphones(allow_fallback: bool = True) -> list[AudioDevice]:
    """Capture devices suitable for microphone input, WASAPI first."""
    with _audio() as pa:
        wasapi = _wasapi_info(pa)
        wasapi_index = int(wasapi["index"]) if wasapi else None
        default_index = int(wasapi["defaultInputDevice"]) if wasapi else -1

        found: list[AudioDevice] = []
        if wasapi_index is not None:
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                except Exception:
                    continue
                if int(info["hostApi"]) != wasapi_index:
                    continue
                if int(info["maxInputChannels"]) < 1:
                    continue
                if info.get("isLoopbackDevice", False):
                    continue
                found.append(
                    _build(pa, dict(info), DeviceKind.MICROPHONE, i == default_index, wasapi_index)
                )

        if not found and allow_fallback:
            logger.warning("No WASAPI microphones found; falling back to other host APIs")
            try:
                default_index = int(pa.get_default_input_device_info()["index"])
            except Exception:
                default_index = -1
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                except Exception:
                    continue
                if int(info["maxInputChannels"]) < 1 or info.get("isLoopbackDevice", False):
                    continue
                found.append(
                    _build(pa, dict(info), DeviceKind.MICROPHONE, i == default_index, wasapi_index)
                )

        return _dedupe(found)


def list_loopbacks() -> list[AudioDevice]:
    """WASAPI loopback devices — what Windows is playing.

    These are the only correct way to capture system audio: they tap the render
    endpoint directly instead of listening to the speakers with a microphone.
    """
    with _audio() as pa:
        wasapi = _wasapi_info(pa)
        if wasapi is None:
            return []
        wasapi_index = int(wasapi["index"])

        default_render_name = ""
        try:
            default_render_name = _LOOPBACK_SUFFIX.sub(
                "", str(pa.get_default_wasapi_loopback()["name"])
            ).strip()
        except Exception as exc:
            logger.debug("No default WASAPI loopback: %r", exc)

        found: list[AudioDevice] = []
        try:
            infos = list(pa.get_loopback_device_info_generator())
        except Exception as exc:
            logger.warning("Loopback enumeration failed: %r", exc)
            infos = []

        for info in infos:
            info = dict(info)
            render = _LOOPBACK_SUFFIX.sub("", str(info["name"])).strip()
            found.append(
                _build(
                    pa,
                    info,
                    DeviceKind.LOOPBACK,
                    bool(default_render_name) and render == default_render_name,
                    wasapi_index,
                )
            )
        return _dedupe(found)


def list_outputs() -> list[AudioDevice]:
    """Render endpoints, for the "Output device" selector in the UI."""
    with _audio() as pa:
        wasapi = _wasapi_info(pa)
        if wasapi is None:
            return []
        wasapi_index = int(wasapi["index"])
        default_index = int(wasapi["defaultOutputDevice"])

        found: list[AudioDevice] = []
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            if int(info["hostApi"]) != wasapi_index:
                continue
            if int(info["maxOutputChannels"]) < 1:
                continue
            found.append(
                _build(pa, dict(info), DeviceKind.OUTPUT, i == default_index, wasapi_index)
            )
        return _dedupe(found)


def loopback_for_output(output: AudioDevice) -> AudioDevice | None:
    """Find the loopback device that mirrors a given render endpoint.

    The UI lets the user pick an *output* device ("Speakers (Realtek)"); capture
    has to happen on its loopback twin ("Speakers (Realtek) [Loopback]").
    """
    target = _LOOPBACK_SUFFIX.sub("", output.name).strip().casefold()
    candidates = list_loopbacks()
    for dev in candidates:
        if dev.render_endpoint.casefold() == target:
            return dev
    # Windows sometimes appends/omits endpoint decorations; fall back to a
    # prefix match before giving up entirely.
    for dev in candidates:
        if dev.render_endpoint.casefold().startswith(target[:20]):
            return dev
    return None


def find_by_key(key: str, kind: DeviceKind | None = None) -> AudioDevice | None:
    """Resolve a persisted device key back to a currently present device."""
    pools: list[list[AudioDevice]]
    if kind is DeviceKind.MICROPHONE:
        pools = [list_microphones()]
    elif kind is DeviceKind.LOOPBACK:
        pools = [list_loopbacks()]
    elif kind is DeviceKind.OUTPUT:
        pools = [list_outputs()]
    else:
        pools = [list_microphones(), list_loopbacks(), list_outputs()]

    for pool in pools:
        for dev in pool:
            if dev.key == key:
                return dev
    return None


def default_microphone() -> AudioDevice | None:
    mics = list_microphones()
    if not mics:
        return None
    return next((m for m in mics if m.is_default), mics[0])


def default_loopback() -> AudioDevice | None:
    loops = list_loopbacks()
    if not loops:
        return None
    return next((lb for lb in loops if lb.is_default), loops[0])


def default_output() -> AudioDevice | None:
    outs = list_outputs()
    if not outs:
        return None
    return next((o for o in outs if o.is_default), outs[0])


def device_is_present(device: AudioDevice) -> bool:
    """True if a device with the same stable key is still enumerated."""
    return find_by_key(device.key, device.kind) is not None


def describe_environment() -> str:
    """Multi-line summary of the audio environment, for logs and diagnostics."""
    lines: list[str] = []
    with _audio() as pa:
        for i in range(pa.get_host_api_count()):
            try:
                h = pa.get_host_api_info_by_index(i)
            except Exception:
                continue
            lines.append(f"host api [{h['index']}] {h['name']}: {h['deviceCount']} devices")
    lines.append(f"microphones: {len(list_microphones())}")
    lines.append(f"loopbacks:   {len(list_loopbacks())}")
    lines.append(f"outputs:     {len(list_outputs())}")
    return "\n".join(lines)
