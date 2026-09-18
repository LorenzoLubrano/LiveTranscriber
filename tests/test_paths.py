"""Application paths: safe names and never overwriting an existing recording."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.utils.paths import (
    app_data_dir,
    default_recordings_dir,
    logs_dir,
    models_dir,
    sanitize_name,
    unique_session_dir,
)


def test_directories_are_nested_under_the_app_root():
    root = app_data_dir()
    assert models_dir().parent == root
    assert logs_dir().parent == root
    assert root.name == "LiveTranscriber"


def test_recordings_default_is_outside_appdata():
    """Recordings are user content and must be easy to find."""
    assert "LiveTranscriber" in default_recordings_dir().parts


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Lezione QCED", "Lezione QCED"),
        ("Fisica: meccanica/quantistica", "Fisica_ meccanica_quantistica"),
        ("a<b>c|d?e*f", "a_b_c_d_e_f"),
        ("   spaced   out   ", "spaced out"),
        ("trailing...", "trailing"),
    ],
)
def test_sanitize_name(raw, expected):
    assert sanitize_name(raw) == expected


def test_sanitize_name_rejects_windows_reserved_names():
    assert sanitize_name("CON") == "_CON"
    assert sanitize_name("aux.txt") == "_aux.txt"


def test_sanitize_name_falls_back_when_everything_is_stripped():
    assert sanitize_name("///") == "Session"
    assert sanitize_name("") == "Session"


def test_sanitize_name_is_length_capped():
    assert len(sanitize_name("x" * 500)) <= 80


def test_session_dir_includes_date_and_title(tmp_path):
    when = datetime(2026, 9, 18, 14, 30)
    path = unique_session_dir(tmp_path, "Lezione QCED", when)
    assert path.name == "2026-09-18_14-30_Lezione QCED"


def test_session_dir_without_title(tmp_path):
    when = datetime(2026, 9, 18, 14, 30)
    assert unique_session_dir(tmp_path, "", when).name == "2026-09-18_14-30"


def test_session_dir_never_collides_with_existing(tmp_path):
    """Spec §14: never overwrite a previous recording."""
    when = datetime(2026, 9, 18, 14, 30)

    first = unique_session_dir(tmp_path, "Lezione", when)
    first.mkdir(parents=True)
    (first / "audio.wav").write_bytes(b"precious")

    second = unique_session_dir(tmp_path, "Lezione", when)
    assert second != first
    assert second.name.endswith("_2")
    second.mkdir()

    third = unique_session_dir(tmp_path, "Lezione", when)
    assert third.name.endswith("_3")

    assert (first / "audio.wav").read_bytes() == b"precious"


def test_session_dir_is_not_created_by_the_helper(tmp_path):
    path = unique_session_dir(tmp_path, "X", datetime(2026, 1, 1))
    assert not path.exists()
