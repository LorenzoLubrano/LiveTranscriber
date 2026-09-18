"""Format conversion: dtype handling, downmix, and streaming resampling."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.resampler import (
    TARGET_SAMPLE_RATE,
    StreamResampler,
    peak_level,
    resample_array,
    rms_dbfs,
    rms_level,
    to_float32,
    to_mono,
)


def sine(freq: float, seconds: float, rate: int, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# -- dtype conversion ------------------------------------------------------

def test_int16_scales_to_unit_range():
    out = to_float32(np.array([32767, -32768, 0], dtype=np.int16))
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(1.0, abs=1e-4)
    assert out[1] == pytest.approx(-1.0, abs=1e-4)
    assert out[2] == 0.0


def test_uint8_is_centred_on_128():
    out = to_float32(np.array([128, 255, 0], dtype=np.uint8))
    assert out[0] == pytest.approx(0.0)
    assert out[1] > 0.9
    assert out[2] == pytest.approx(-1.0)


def test_float32_passes_through_without_copy_cost():
    data = np.array([0.5, -0.5], dtype=np.float32)
    assert to_float32(data) is data


# -- downmix ---------------------------------------------------------------

def test_stereo_downmix_averages_channels():
    interleaved = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)  # L=1, R=0
    np.testing.assert_allclose(to_mono(interleaved, 2), [0.5, 0.5])


def test_hard_panned_speech_survives_downmix():
    """Averaging, not channel-0 selection: dialogue is often panned."""
    silent_left_loud_right = np.array([0.0, 0.8, 0.0, 0.8], dtype=np.float32)
    mono = to_mono(silent_left_loud_right, 2)
    assert mono.max() > 0.3, "right-channel-only audio must not become silence"


def test_mono_input_is_unchanged():
    data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    np.testing.assert_array_equal(to_mono(data, 1), data)


def test_downmix_drops_trailing_partial_frame():
    assert to_mono(np.array([1.0, 1.0, 1.0], dtype=np.float32), 2).size == 1


# -- levels ----------------------------------------------------------------

def test_level_helpers():
    data = np.array([0.5, -0.9, 0.2], dtype=np.float32)
    assert peak_level(data) == pytest.approx(0.9)
    assert rms_level(data) == pytest.approx(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    assert rms_dbfs(np.zeros(10, dtype=np.float32)) == float("-inf")
    assert rms_dbfs(np.ones(10, dtype=np.float32)) == pytest.approx(0.0, abs=1e-6)
    assert peak_level(np.zeros(0, dtype=np.float32)) == 0.0


# -- resampling ------------------------------------------------------------

@pytest.mark.parametrize("source_rate", [8000, 16000, 22050, 44100, 48000, 96000])
def test_output_length_matches_rate_ratio(source_rate):
    audio = sine(440, 1.0, source_rate)
    out = resample_array(audio, source_rate, TARGET_SAMPLE_RATE)
    assert out.dtype == np.float32
    assert abs(out.size - TARGET_SAMPLE_RATE) <= 32, f"got {out.size} samples"


def test_same_rate_is_passthrough():
    audio = sine(440, 0.1, 16000)
    np.testing.assert_array_equal(resample_array(audio, 16000, 16000), audio)


def test_resampling_preserves_tone_frequency():
    """A 1 kHz tone at 48 kHz must still be 1 kHz at 16 kHz."""
    out = resample_array(sine(1000, 0.5, 48000), 48000, 16000)
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(out.size, 1 / 16000)[int(np.argmax(spectrum))]
    assert peak_hz == pytest.approx(1000, abs=20)


def test_streaming_matches_one_shot():
    """Chunked processing must not differ from processing the whole array.

    This is the property that prevents clicks at chunk boundaries — it fails
    for any stateless resampler.
    """
    source_rate = 48000
    audio = sine(440, 1.0, source_rate)

    streamer = StreamResampler(source_rate, channels=1)
    chunks = [streamer.process(audio[i : i + 1024]) for i in range(0, audio.size, 1024)]
    chunks.append(streamer.flush())
    streamed = np.concatenate(chunks)

    one_shot = resample_array(audio, source_rate, TARGET_SAMPLE_RATE)

    n = min(streamed.size, one_shot.size)
    assert n > TARGET_SAMPLE_RATE * 0.9
    np.testing.assert_allclose(streamed[:n], one_shot[:n], atol=1e-4)


def test_streaming_has_no_discontinuity_at_chunk_edges():
    """A pure tone must stay smooth across the joins."""
    streamer = StreamResampler(48000, channels=1)
    audio = sine(200, 0.5, 48000)
    out = np.concatenate(
        [streamer.process(audio[i : i + 512]) for i in range(0, audio.size, 512)]
    )
    # Max sample-to-sample step of a 200 Hz tone at 16 kHz is small; a reset
    # filter would produce a visible jump.
    assert np.abs(np.diff(out)).max() < 0.05


def test_stream_resampler_handles_stereo_input():
    streamer = StreamResampler(48000, channels=2)
    stereo = np.repeat(sine(440, 0.1, 48000), 2)  # duplicate each sample L/R

    out = streamer.process(stereo)
    assert out.ndim == 1
    # A streaming resampler holds part of the signal in its delay line, so the
    # first chunk is deliberately short; flush() releases the remainder.
    assert out.size < 1600
    total = out.size + streamer.flush().size
    assert abs(total - 1600) <= 32, f"got {total} samples after flush"


def test_reset_clears_filter_state():
    streamer = StreamResampler(48000, channels=1)
    streamer.process(sine(440, 0.1, 48000))
    streamer.reset()
    out = streamer.process(sine(440, 0.1, 48000))
    assert out.size > 0


def test_empty_chunk_is_safe():
    streamer = StreamResampler(48000, channels=1)
    assert streamer.process(np.zeros(0, dtype=np.float32)).size == 0


@pytest.mark.parametrize("rate,channels", [(0, 1), (-1, 1), (48000, 0)])
def test_invalid_construction_rejected(rate, channels):
    with pytest.raises(ValueError):
        StreamResampler(rate, channels)


def test_silence_stays_silent():
    out = resample_array(np.zeros(48000, dtype=np.float32), 48000, 16000)
    assert np.abs(out).max() == pytest.approx(0.0, abs=1e-9)
