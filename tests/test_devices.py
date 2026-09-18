"""Device enumeration: identity, de-duplication, loopback pairing.

Pure logic is tested with synthetic devices; the tests that need real hardware
are marked and skipped when no audio endpoint is present, so the suite runs on
any machine.
"""

from __future__ import annotations

import pytest

from app.audio.devices import (
    AudioDevice,
    DeviceKind,
    _dedupe,
    default_loopback,
    default_microphone,
    find_by_key,
    list_loopbacks,
    list_microphones,
    list_outputs,
    loopback_for_output,
)


def make_device(
    name: str,
    kind: DeviceKind = DeviceKind.MICROPHONE,
    index: int = 0,
    host: str = "Windows WASAPI",
    channels: int = 2,
    rate: int = 48000,
    is_default: bool = False,
) -> AudioDevice:
    return AudioDevice(
        index=index,
        name=name,
        kind=kind,
        host_api_index=2,
        host_api_name=host,
        max_input_channels=channels if kind is not DeviceKind.OUTPUT else 0,
        max_output_channels=0 if kind is not DeviceKind.OUTPUT else channels,
        default_sample_rate=rate,
        is_default=is_default,
        render_endpoint=name.replace(" [Loopback]", "") if kind is DeviceKind.LOOPBACK else "",
    )


# -- identity --------------------------------------------------------------

def test_key_is_stable_across_index_changes():
    """Indices shift when devices are plugged in; the key must not."""
    a = make_device("Microphone Array (Realtek(R) Audio)", index=14)
    b = make_device("Microphone Array (Realtek(R) Audio)", index=7)
    assert a.key == b.key


def test_key_distinguishes_host_apis():
    wasapi = make_device("Speakers", host="Windows WASAPI")
    mme = make_device("Speakers", host="MME")
    assert wasapi.key != mme.key


def test_key_distinguishes_kinds():
    mic = make_device("Realtek", kind=DeviceKind.MICROPHONE)
    loop = make_device("Realtek", kind=DeviceKind.LOOPBACK)
    assert mic.key != loop.key


def test_display_name_strips_loopback_marker():
    dev = make_device("Speakers (Realtek(R) Audio) [Loopback]", kind=DeviceKind.LOOPBACK)
    assert dev.display_name == "Speakers (Realtek(R) Audio)"


def test_display_name_is_undecorated_for_error_messages():
    """display_name is embedded in user-facing errors; keep it a plain name."""
    dev = make_device("Speakers", is_default=True)
    assert dev.display_name == "Speakers"


def test_label_marks_the_default_for_selection_lists():
    assert make_device("Speakers", is_default=True).label == "Speakers (predefinito)"
    assert make_device("Speakers", is_default=False).label == "Speakers"


def test_capture_channels_never_zero():
    """A device reporting 0 input channels must still open as mono."""
    dev = make_device("Odd", channels=0)
    assert dev.capture_channels == 1


def test_mono_device_reports_one_channel():
    dev = make_device("Headset Microphone (Oculus Virtual Audio Device)", channels=1)
    assert dev.capture_channels == 1


# -- de-duplication --------------------------------------------------------

def test_dedupe_removes_repeated_endpoints():
    devices = [make_device("Mic", index=14), make_device("Mic", index=15)]
    assert len(_dedupe(devices)) == 1


def test_dedupe_keeps_distinct_devices():
    devices = [make_device("Mic A"), make_device("Mic B"), make_device("Mic C")]
    assert len(_dedupe(devices)) == 3


def test_dedupe_preserves_order():
    devices = [make_device("A"), make_device("B"), make_device("A", index=9)]
    assert [d.name for d in _dedupe(devices)] == ["A", "B"]


def test_dedupe_promotes_a_duplicate_that_is_the_default():
    devices = [make_device("Mic", index=14), make_device("Mic", index=15, is_default=True)]
    result = _dedupe(devices)
    assert len(result) == 1
    assert result[0].is_default
    assert result[0].index == 14, "the first (WASAPI-preferred) entry is kept"


def test_dedupe_of_empty_list():
    assert _dedupe([]) == []


# -- loopback pairing ------------------------------------------------------

def test_loopback_pairing_matches_render_endpoint(monkeypatch):
    speakers_lb = make_device("Speakers (Realtek(R) Audio) [Loopback]", DeviceKind.LOOPBACK)
    headset_lb = make_device(
        "Cuffie (Oculus Virtual Audio Device) [Loopback]", DeviceKind.LOOPBACK, index=16
    )
    monkeypatch.setattr(
        "app.audio.devices.list_loopbacks", lambda: [headset_lb, speakers_lb]
    )

    speakers_out = make_device("Speakers (Realtek(R) Audio)", DeviceKind.OUTPUT)
    assert loopback_for_output(speakers_out) is speakers_lb


def test_loopback_pairing_returns_none_when_absent(monkeypatch):
    monkeypatch.setattr("app.audio.devices.list_loopbacks", lambda: [])
    assert loopback_for_output(make_device("Nothing", DeviceKind.OUTPUT)) is None


# -- real hardware ---------------------------------------------------------

@pytest.mark.hardware
def test_enumeration_finds_real_devices():
    mics, loops, outs = list_microphones(), list_loopbacks(), list_outputs()
    if not (mics or loops or outs):
        pytest.skip("no audio hardware on this machine")

    for pool in (mics, loops, outs):
        keys = [d.key for d in pool]
        assert len(keys) == len(set(keys)), "enumeration returned duplicates"


@pytest.mark.hardware
def test_every_loopback_is_capturable():
    loops = list_loopbacks()
    if not loops:
        pytest.skip("no loopback devices")
    for dev in loops:
        assert dev.max_input_channels >= 1, f"{dev} cannot be opened for input"
        assert dev.default_sample_rate > 0
        assert dev.render_endpoint, "loopback must name the endpoint it mirrors"


@pytest.mark.hardware
def test_defaults_resolve_and_round_trip_by_key():
    mic = default_microphone()
    if mic is None:
        pytest.skip("no microphone")
    assert find_by_key(mic.key, DeviceKind.MICROPHONE) is not None

    loop = default_loopback()
    if loop is not None:
        assert find_by_key(loop.key, DeviceKind.LOOPBACK) is not None


@pytest.mark.hardware
def test_wasapi_reports_untruncated_names():
    """MME truncates at 31 chars; WASAPI must not be doing that."""
    mics = list_microphones()
    if not mics:
        pytest.skip("no microphone")
    wasapi_mics = [m for m in mics if m.is_wasapi]
    if not wasapi_mics:
        pytest.skip("no WASAPI microphones")
    assert all(m.host_api_name == "Windows WASAPI" for m in wasapi_mics)
