"""Model catalogue, settings and engine plumbing.

Everything here runs without a downloaded model and without a GPU. The tests
that need real weights live at the bottom behind the ``hardware`` marker.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.transcription import models
from app.transcription.cuda_setup import ensure_cuda_libraries, find_cuda_dirs
from app.transcription.engine import (
    AUTO_LANGUAGE,
    PRIORITY_LANGUAGES,
    Segment,
    TranscriptionResult,
    TranscriptionSettings,
    _collect,
    _friendly_gpu_error,
    _friendly_inference_error,
)
from app.transcription.models import ModelError

# -- catalogue -------------------------------------------------------------


def test_catalogue_covers_every_model_the_spec_lists():
    """Spec §6 names these six explicitly."""
    keys = {spec.key for spec in models.list_models()}
    assert {"tiny", "base", "small", "medium", "large-v3", "turbo"} <= keys


def test_exactly_one_model_is_recommended():
    recommended = [s for s in models.list_models() if s.recommended]
    assert len(recommended) == 1
    assert recommended[0].key == models.DEFAULT_MODEL


def test_every_model_has_a_real_repository():
    for spec in models.list_models():
        assert "/" in spec.repo_id, f"{spec.key} needs a full HF repo id"
        assert spec.approx_size_mb > 0
        assert spec.quality and spec.description


def test_models_are_listed_in_the_order_the_spec_shows_them():
    """Spec §6 fixes this order, which is by quality tier rather than by size.

    Turbo comes last despite being smaller than Large-v3, because it is the
    quality/speed compromise rather than a step up in accuracy.
    """
    keys = [spec.key for spec in models.list_models()]
    assert keys == ["tiny", "base", "small", "medium", "large-v3", "turbo"]


def test_size_labels_switch_to_gigabytes():
    assert models.get_spec("tiny").size_label == "75 MB"
    assert models.get_spec("large-v3").size_label.endswith("GB")


def test_unknown_model_raises_a_user_facing_error():
    with pytest.raises(ModelError) as info:
        models.get_spec("enormous")
    assert "enormous" in info.value.user_message
    assert "Traceback" not in info.value.user_message


def test_models_live_under_the_app_directory():
    directory = models.get_spec("small").directory
    assert directory.name == "small"
    assert "LiveTranscriber" in str(directory)


def test_missing_model_is_not_available(monkeypatch, tmp_path):
    monkeypatch.setattr("app.transcription.models.models_dir", lambda: tmp_path)
    assert not models.is_available("small")


def test_empty_directory_is_not_mistaken_for_a_download(monkeypatch, tmp_path):
    """An interrupted download leaves a directory but no weights."""
    monkeypatch.setattr("app.transcription.models.models_dir", lambda: tmp_path)
    (tmp_path / "small").mkdir()
    assert not models.is_available("small")


def test_truncated_weights_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr("app.transcription.models.models_dir", lambda: tmp_path)
    directory = tmp_path / "small"
    directory.mkdir()
    (directory / "model.bin").write_bytes(b"x" * 512)  # far too small
    assert not models.is_available("small")


def test_plausible_weights_count_as_available(monkeypatch, tmp_path):
    monkeypatch.setattr("app.transcription.models.models_dir", lambda: tmp_path)
    directory = tmp_path / "small"
    directory.mkdir()
    (directory / "model.bin").write_bytes(b"x" * (2 * 1024 * 1024))
    assert models.is_available("small")
    assert models.disk_usage_mb("small") == 2


def test_deleting_a_model_that_is_not_there(monkeypatch, tmp_path):
    monkeypatch.setattr("app.transcription.models.models_dir", lambda: tmp_path)
    assert models.delete_model("small") is False


@pytest.mark.parametrize(
    "exc,fragment",
    [
        (ConnectionError("Connection refused"), "Internet"),
        (OSError("No space left on device"), "Spazio"),
        (PermissionError("Access is denied"), "Permessi"),
        (RuntimeError("something odd"), "Impossibile scaricare"),
    ],
)
def test_download_errors_are_translated(exc, fragment):
    message = models._friendly_download_error(exc, models.get_spec("small"))
    assert fragment in message
    assert "Traceback" not in message


# -- settings --------------------------------------------------------------


def test_auto_language_maps_to_none():
    """faster-whisper detects the language when given None."""
    assert TranscriptionSettings(language=AUTO_LANGUAGE).whisper_language() is None
    assert TranscriptionSettings(language="it").whisper_language() == "it"


def test_translation_mode_switches_the_task():
    assert TranscriptionSettings().task == "transcribe"
    assert TranscriptionSettings(translate_to_english=True).task == "translate"


def test_priority_languages_lead_with_auto_then_italian_and_english():
    codes = [code for code, _ in PRIORITY_LANGUAGES]
    assert codes[0] == AUTO_LANGUAGE
    assert codes[1:3] == ["it", "en"]


def test_vad_parameters_match_the_backend_keys():
    """These names are what faster_whisper.vad.VadOptions accepts."""
    params = TranscriptionSettings().vad_parameters()
    assert set(params) == {
        "threshold",
        "min_speech_duration_ms",
        "min_silence_duration_ms",
        "speech_pad_ms",
    }


def test_defaults_are_tuned_for_streaming():
    settings = TranscriptionSettings()
    assert settings.temperature == 0.0, "sampling invites hallucination"
    assert settings.condition_on_previous_text is False, "prevents repetition loops"
    assert settings.vad_filter is True
    assert settings.hallucination_silence_threshold is not None


# -- segments --------------------------------------------------------------


def test_segment_duration():
    assert Segment(1.0, 3.5, "ciao").duration == pytest.approx(2.5)


def test_shifting_places_a_segment_on_the_session_timeline():
    shifted = Segment(1.0, 2.0, "ciao").shifted(100.0)
    assert (shifted.start, shifted.end) == (101.0, 102.0)
    assert shifted.text == "ciao"


def test_result_text_joins_segments():
    result = TranscriptionResult(
        segments=[Segment(0, 1, "Primo."), Segment(1, 2, " Secondo.")]
    )
    assert result.text == "Primo. Secondo."


def test_real_time_factor():
    result = TranscriptionResult(duration=10.0, inference_time=2.0)
    assert result.real_time_factor == pytest.approx(5.0)
    assert TranscriptionResult(duration=10.0).real_time_factor == 0.0


def test_collect_skips_blank_segments():
    class Fake:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    collected = _collect([Fake(0, 1, "ciao"), Fake(1, 2, "   "), Fake(2, 3, "")], offset=0.0)
    assert [s.text for s in collected] == ["ciao"]


def test_collect_applies_the_offset():
    class Fake:
        start, end, text = 1.0, 2.0, "ciao"

    assert _collect([Fake()], offset=60.0)[0].start == pytest.approx(61.0)


# -- error translation -----------------------------------------------------


@pytest.mark.parametrize(
    "raw,fragment",
    [
        ("CUDA failed with error out of memory", "Memoria GPU insufficiente"),
        ("Library cublas64_12.dll is not found", "librerie CUDA"),
        ("no kernel image is available for execution", "non è utilizzabile"),
        ("mystery failure", "GPU non disponibile"),
    ],
)
def test_gpu_errors_are_translated(raw, fragment):
    message = _friendly_gpu_error(RuntimeError(raw))
    assert fragment in message
    assert "Traceback" not in message and "RuntimeError" not in message


def test_gpu_errors_always_say_what_happens_next():
    """The user needs to know transcription continues, not that it died."""
    for raw in ("out of memory", "cublas missing", "no cuda driver", "???"):
        assert "CPU" in _friendly_gpu_error(RuntimeError(raw))


@pytest.mark.parametrize(
    "raw,fragment",
    [
        ("CUDA out of memory", "modello più piccolo"),
        ("cublas error", "GPU"),
        ("unexpected", "Errore durante la trascrizione"),
    ],
)
def test_inference_errors_are_translated(raw, fragment):
    assert fragment in _friendly_inference_error(RuntimeError(raw))


# -- CUDA setup ------------------------------------------------------------


def test_cuda_directory_registration_is_idempotent():
    first = ensure_cuda_libraries()
    second = ensure_cuda_libraries()
    assert first == second


def test_discovered_cuda_directories_contain_dlls():
    for path in find_cuda_dirs():
        assert path.is_dir()
        assert any(path.glob("*.dll")), f"{path} was registered but holds no DLLs"


# -- real inference --------------------------------------------------------


@pytest.mark.hardware
def test_engine_transcribes_real_audio():
    """End to end with actual weights, if any model is downloaded."""
    from app.transcription.engine import TranscriptionEngine
    from app.transcription.hardware import Accelerator

    installed = models.installed_models()
    if not installed:
        pytest.skip("no Whisper model downloaded")

    engine = TranscriptionEngine(installed[0].key, accelerator=Accelerator.AUTO)
    try:
        choice = engine.load()
        assert engine.is_loaded
        assert choice.device in ("cpu", "cuda")

        # Silence must not invent words (spec §5: no hallucination on silence).
        result = engine.transcribe(np.zeros(16000 * 2, dtype=np.float32))
        assert result.text.strip() == "", f"hallucinated on silence: {result.text!r}"
    finally:
        engine.unload()
    assert not engine.is_loaded


@pytest.mark.hardware
def test_empty_audio_returns_an_empty_result():
    from app.transcription.engine import TranscriptionEngine

    installed = models.installed_models()
    if not installed:
        pytest.skip("no Whisper model downloaded")

    engine = TranscriptionEngine(installed[0].key)
    try:
        result = engine.transcribe(np.zeros(0, dtype=np.float32))
        assert result.segments == []
    finally:
        engine.unload()
