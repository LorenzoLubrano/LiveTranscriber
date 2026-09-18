"""WAV recording: valid headers, crash safety, and time-aligned mixing."""

from __future__ import annotations

import struct
import wave

import numpy as np
import pytest

from app.audio.recorder import MixRecorder, WavWriter, float32_to_int16


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        return {
            "channels": w.getnchannels(),
            "rate": w.getframerate(),
            "width": w.getsampwidth(),
            "frames": w.getnframes(),
            "data": np.frombuffer(frames, dtype=np.int16),
        }


# -- conversion ------------------------------------------------------------

def test_float_to_int16_clips_instead_of_wrapping():
    """Wrapping would turn a loud peak into a full-scale crack."""
    out = float32_to_int16(np.array([1.5, -1.5], dtype=np.float32))
    assert out[0] == 32767
    assert out[1] == -32767
    assert out.dtype == np.int16


def test_float_to_int16_scales_correctly():
    out = float32_to_int16(np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32))
    assert out[0] == 0
    assert out[1] == 32767
    assert out[3] == pytest.approx(16383, abs=2)


# -- writer ----------------------------------------------------------------

def test_written_file_is_a_valid_wav(tmp_path):
    path = tmp_path / "a.wav"
    with WavWriter(path, sample_rate=16000) as w:
        w.write(np.zeros(16000, dtype=np.float32))

    info = read_wav(path)
    assert info["rate"] == 16000
    assert info["channels"] == 1
    assert info["width"] == 2
    assert info["frames"] == 16000


def test_roundtrip_preserves_signal(tmp_path):
    path = tmp_path / "tone.wav"
    t = np.arange(8000, dtype=np.float32) / 16000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    with WavWriter(path, 16000) as w:
        w.write(tone)

    back = read_wav(path)["data"].astype(np.float32) / 32767.0
    np.testing.assert_allclose(back, tone, atol=1e-4)


def test_header_is_valid_before_close(tmp_path):
    """The crash-safety property: kill the process, keep the audio."""
    path = tmp_path / "crash.wav"
    writer = WavWriter(path, 16000, flush_interval_s=0.0)  # sync on every write
    writer.write(np.ones(16000, dtype=np.float32) * 0.25)

    # Deliberately do NOT close — simulate the process dying here.
    info = read_wav(path)
    assert info["frames"] == 16000, "audio written before a crash must be readable"
    assert np.abs(info["data"]).max() > 0

    writer.close()


def test_sync_updates_sizes_in_place(tmp_path):
    path = tmp_path / "s.wav"
    writer = WavWriter(path, 16000, flush_interval_s=999)
    writer.write(np.zeros(4000, dtype=np.float32))
    writer.sync()

    raw = path.read_bytes()
    riff_size = struct.unpack("<I", raw[4:8])[0]
    data_size = struct.unpack("<I", raw[40:44])[0]
    assert data_size == 8000
    assert riff_size == 8000 + 36
    writer.close()


def test_incremental_writes_accumulate(tmp_path):
    path = tmp_path / "inc.wav"
    with WavWriter(path, 16000) as w:
        for _ in range(10):
            w.write(np.zeros(1600, dtype=np.float32))
        assert w.frames_written == 16000
        assert w.duration_seconds == pytest.approx(1.0)
    assert read_wav(path)["frames"] == 16000


def test_close_is_idempotent(tmp_path):
    w = WavWriter(tmp_path / "x.wav", 16000)
    w.write(np.zeros(100, dtype=np.float32))
    w.close()
    w.close()
    assert w.is_closed


def test_write_after_close_is_ignored(tmp_path):
    path = tmp_path / "y.wav"
    w = WavWriter(path, 16000)
    w.write(np.zeros(100, dtype=np.float32))
    w.close()
    assert w.write(np.ones(100, dtype=np.float32)) == 0
    assert read_wav(path)["frames"] == 100


def test_empty_write_is_noop(tmp_path):
    with WavWriter(tmp_path / "e.wav", 16000) as w:
        assert w.write(np.zeros(0, dtype=np.float32)) == 0


def test_parent_directory_created(tmp_path):
    path = tmp_path / "deep" / "nested" / "z.wav"
    with WavWriter(path, 16000) as w:
        w.write(np.zeros(10, dtype=np.float32))
    assert path.exists()


# -- mixing ----------------------------------------------------------------

def test_mix_recorder_writes_three_tracks(tmp_path):
    rec = MixRecorder(tmp_path / "pc.wav", tmp_path / "mic.wav", tmp_path / "mix.wav", 16000)
    rec.write_pc(np.ones(1600, dtype=np.float32) * 0.5)
    rec.write_mic(np.ones(1600, dtype=np.float32) * 0.5)
    rec.close()

    for name in ("pc.wav", "mic.wav", "mix.wav"):
        assert (tmp_path / name).exists(), name
    assert read_wav(tmp_path / "mix.wav")["frames"] == 1600


def test_mixed_track_combines_both_sources(tmp_path):
    rec = MixRecorder(tmp_path / "pc.wav", tmp_path / "mic.wav", tmp_path / "mix.wav", 16000)
    rec.write_pc(np.full(1600, 0.4, dtype=np.float32))
    rec.write_mic(np.full(1600, 0.2, dtype=np.float32))
    rec.close()

    mixed = read_wav(tmp_path / "mix.wav")["data"].astype(np.float32) / 32767.0
    assert mixed.mean() == pytest.approx((0.4 + 0.2) * 0.5, abs=1e-3)


def test_mix_stays_aligned_when_one_source_lags(tmp_path):
    """Out-of-order arrival must not shift audio on the timeline."""
    rec = MixRecorder(tmp_path / "pc.wav", tmp_path / "mic.wav", tmp_path / "mix.wav", 16000)

    rec.write_pc(np.full(800, 0.5, dtype=np.float32))
    rec.write_pc(np.full(800, 0.5, dtype=np.float32))
    rec.write_mic(np.full(1600, 0.5, dtype=np.float32))  # arrives late, all at once
    rec.close()

    mixed = read_wav(tmp_path / "mix.wav")["data"].astype(np.float32) / 32767.0
    assert mixed.size == 1600
    # Every sample had both sources present, so all should be 0.5.
    np.testing.assert_allclose(mixed, np.full(1600, 0.5), atol=1e-3)


def test_mix_does_not_stall_when_one_source_is_silent(tmp_path):
    """A muted mic must not freeze the mixed track for the whole session."""
    rec = MixRecorder(tmp_path / "pc.wav", tmp_path / "mic.wav", tmp_path / "mix.wav", 16000)
    for _ in range(10):  # 10 s of PC audio, zero mic callbacks
        rec.write_pc(np.full(16000, 0.5, dtype=np.float32))
    rec.close()

    assert read_wav(tmp_path / "mix.wav")["frames"] == 160000


def test_mix_track_omitted_when_only_one_source(tmp_path):
    rec = MixRecorder(tmp_path / "pc.wav", None, tmp_path / "mix.wav", 16000)
    rec.write_pc(np.zeros(1600, dtype=np.float32))
    rec.close()

    assert rec.mixed is None
    assert not (tmp_path / "mix.wav").exists()
    assert set(rec.paths) == {"pc"}


def test_mix_memory_stays_bounded(tmp_path):
    """Pending buffers must be released as regions are mixed."""
    rec = MixRecorder(tmp_path / "pc.wav", tmp_path / "mic.wav", tmp_path / "mix.wav", 16000)
    chunk = np.zeros(1600, dtype=np.float32)
    for _ in range(500):  # ~50 s
        rec.write_pc(chunk)
        rec.write_mic(chunk)

    pending = rec._pending["pc"].size + rec._pending["mic"].size
    assert pending < 16000 * 5, f"pending audio grew to {pending} samples"
    rec.close()
