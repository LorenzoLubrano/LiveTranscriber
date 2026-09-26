"""The first-run choice: does it offer models this machine can actually run?"""

from __future__ import annotations

import pytest

from app.config.settings import AppSettings
from app.transcription.hardware import AcceleratorChoice
from app.transcription.models import Quality
from app.ui.first_run import FirstRunDialog


@pytest.fixture
def cpu_machine(monkeypatch):
    # Patched where it is used: first_run imports the name directly, so
    # patching the hardware module would leave the dialog on the real GPU.
    monkeypatch.setattr(
        "app.ui.first_run.select_accelerator",
        lambda *a, **k: AcceleratorChoice(
            device="cpu", compute_type="int8", cpu_threads=4
        ),
    )
    monkeypatch.setattr("app.transcription.hardware.detect_gpus", lambda: [])
    return AppSettings()


def test_it_offers_every_preset(qtbot, cpu_machine):
    dialog = FirstRunDialog(cpu_machine)
    qtbot.addWidget(dialog)
    labels = {b.text() for b in dialog.group.buttons()}
    assert labels == {q.label for q in Quality}


def test_balanced_is_preselected(qtbot, cpu_machine):
    dialog = FirstRunDialog(cpu_machine)
    qtbot.addWidget(dialog)
    checked = dialog.group.checkedButton()
    assert Quality(checked.property("quality")) is Quality.BALANCED


def test_a_cpu_machine_is_not_offered_gpu_sized_models(qtbot, cpu_machine):
    dialog = FirstRunDialog(cpu_machine)
    qtbot.addWidget(dialog)
    offered = {b.property("model") for b in dialog.group.buttons()}
    assert offered.isdisjoint({"medium", "large-v3", "turbo"})


def test_measurements_narrow_what_is_offered(qtbot, cpu_machine):
    """The whole point: a measured machine is offered what it can keep up with.

    Without this the presets are a guess from a single bit of hardware
    information, which is what made `small` the "best quality" answer on a CPU
    that needs 0.97 seconds of compute per second of audio.
    """
    plain = FirstRunDialog(cpu_machine)
    qtbot.addWidget(plain)
    before = {
        Quality(b.property("quality")): b.property("model")
        for b in plain.group.buttons()
    }
    assert before[Quality.BEST] == "small"

    cpu_machine.remember_measurement("small", "cpu", "int8", 0.97)
    cpu_machine.remember_measurement("base", "cpu", "int8", 0.55)
    measured = FirstRunDialog(cpu_machine)
    qtbot.addWidget(measured)
    after = {
        Quality(b.property("quality")): b.property("model")
        for b in measured.group.buttons()
    }
    assert after[Quality.BEST] == "base"
    assert after[Quality.BALANCED] == "base"


def test_the_privacy_statement_is_present(qtbot, cpu_machine):
    """Spec §20: the wording is required, not decorative."""
    from PySide6.QtWidgets import QLabel

    dialog = FirstRunDialog(cpu_machine)
    qtbot.addWidget(dialog)
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "localmente sul dispositivo" in text
    assert "Nessun audio viene inviato a server esterni" in text


def test_it_says_where_transcription_will_run(qtbot, cpu_machine):
    from PySide6.QtWidgets import QLabel

    dialog = FirstRunDialog(cpu_machine)
    qtbot.addWidget(dialog)
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "CPU (4 thread)" in text


def test_choosing_later_leaves_no_model_chosen(qtbot, cpu_machine):
    dialog = FirstRunDialog(cpu_machine)
    qtbot.addWidget(dialog)
    dialog.reject()
    assert dialog.chosen_model == ""
    assert dialog.chosen_quality is None
