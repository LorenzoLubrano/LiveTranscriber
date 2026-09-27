"""Interface behaviour.

Driven through pytest-qt against real widgets. The transcript view gets the most
attention because it is where the product's distinctive idea lives — text that
arrives provisional and settles into confirmed — and because incremental editing
of a live QTextDocument is easy to get subtly wrong.
"""

from __future__ import annotations

import pytest

from app.sessions.transcript import Source
from app.ui.theme import DARK, LIGHT, ThemeMode, palette_for, stylesheet
from app.ui.widgets.level_meter import LevelMeter, RecordingDot
from app.ui.widgets.transcript_view import TranscriptView

pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture
def view(qtbot):
    widget = TranscriptView(DARK)
    qtbot.addWidget(widget)
    return widget


# -- theme -----------------------------------------------------------------

def test_both_palettes_define_every_colour():
    for palette in (DARK, LIGHT):
        for name, value in vars(palette).items():
            assert isinstance(value, str) and value.startswith("#"), name


def test_theme_selection():
    assert palette_for(ThemeMode.DARK, False) is DARK
    assert palette_for(ThemeMode.LIGHT, True) is LIGHT
    assert palette_for(ThemeMode.SYSTEM, True) is DARK
    assert palette_for(ThemeMode.SYSTEM, False) is LIGHT


def test_stylesheet_renders_without_placeholders():
    for palette in (DARK, LIGHT):
        css = stylesheet(palette)
        assert "{p." not in css, "an f-string placeholder survived"
        assert palette.window in css


def test_labels_do_not_paint_their_own_surface():
    """Otherwise a band of window colour is drawn across every panel."""
    assert "QLabel {" in stylesheet(DARK)
    assert "background: transparent" in stylesheet(DARK)


def test_dark_and_light_differ_in_contrast_direction():
    assert DARK.window < DARK.ink, "dark theme needs light text on a dark ground"
    assert LIGHT.window > LIGHT.ink


# -- transcript: writing ---------------------------------------------------

def test_confirmed_text_appears(view):
    view.append_confirmed(Source.PC, 0.0, "Consideriamo il sistema.")
    assert "Consideriamo il sistema." in view.toPlainText()


def test_blank_text_is_ignored(view):
    view.append_confirmed(Source.PC, 0.0, "   ")
    assert view.toPlainText() == ""


def test_speaker_tag_appears_once_per_run(view):
    """One speaker reads as prose, not as a tag on every fragment."""
    view.append_confirmed(Source.PC, 0.0, "Prima parte,")
    view.append_confirmed(Source.PC, 2.0, "seconda parte,")
    view.append_confirmed(Source.PC, 4.0, "terza parte.")

    assert view.toPlainText().count("PC") == 1


def test_tag_reappears_when_the_speaker_changes(view):
    view.append_confirmed(Source.PC, 0.0, "Dal computer.")
    view.append_confirmed(Source.MIC, 2.0, "Dal microfono.")
    view.append_confirmed(Source.PC, 4.0, "Di nuovo dal computer.")

    text = view.toPlainText()
    assert text.count("PC") == 2
    assert text.count("MIC") == 1


def test_consecutive_chunks_are_separated_by_exactly_one_space(view):
    view.append_confirmed(Source.PC, 0.0, "Prima parte,")
    view.append_confirmed(Source.PC, 2.0, "seconda parte.")
    assert "parte, seconda" in view.toPlainText()
    assert "  " not in view.toPlainText()


# -- transcript: provisional ----------------------------------------------

def test_provisional_text_is_shown(view):
    view.set_provisional(Source.PC, "testo in arrivo")
    assert "testo in arrivo" in view.toPlainText()


def test_provisional_is_replaced_not_accumulated(view):
    view.set_provisional(Source.PC, "primo tentativo")
    view.set_provisional(Source.PC, "secondo tentativo")

    text = view.toPlainText()
    assert "primo tentativo" not in text
    assert text.count("secondo tentativo") == 1


def test_confirming_replaces_the_provisional_text(view):
    view.set_provisional(Source.PC, "Consideriamo adesso la")
    view.append_confirmed(Source.PC, 0.0, "Consideriamo adesso la Hamiltoniana.")

    text = view.toPlainText()
    assert text.count("Consideriamo") == 1, "provisional text was left behind"
    assert "Hamiltoniana" in text


def test_the_settle_cycle_leaves_clean_text(view):
    """The real loop: provisional, confirmed, provisional, confirmed...

    This is where stray spaces accumulated: the anchor used to sit after the
    separating space, so clearing provisional text left the space behind and the
    next confirmed chunk added another.
    """
    pairs = [
        ("Consideriamo adesso la", "Consideriamo adesso la miltoniana del sistema,"),
        ("dove il termine", "dove il termine di interazione"),
        ("dipende dal", "dipende dal tempo."),
    ]
    for provisional, confirmed in pairs:
        view.set_provisional(Source.PC, provisional)
        view.append_confirmed(Source.PC, 0.0, confirmed)

    text = view.toPlainText()
    assert "  " not in text, f"stray spacing: {text!r}"
    assert not text.split("\n")[-1].startswith(" "), "paragraph starts with a space"
    assert text.endswith("dipende dal tempo.")


def test_clearing_provisional_removes_it(view):
    view.append_confirmed(Source.PC, 0.0, "Confermato.")
    view.set_provisional(Source.PC, "provvisorio")
    view.clear_provisional()

    assert "provvisorio" not in view.toPlainText()
    assert "Confermato." in view.toPlainText()


def test_empty_provisional_clears_rather_than_inserting(view):
    view.set_provisional(Source.PC, "qualcosa")
    view.set_provisional(Source.PC, "")
    assert "qualcosa" not in view.toPlainText()


# -- transcript: options ---------------------------------------------------

def test_timestamps_can_be_switched_on(view):
    view.set_show_timestamps(True)
    view.append_confirmed(Source.PC, 3675.0, "Testo.")
    assert "01:01:15" in view.toPlainText()


def test_timestamps_are_absent_by_default(view):
    view.append_confirmed(Source.PC, 3675.0, "Testo.")
    assert "01:01:15" not in view.toPlainText()


def test_rebuild_reproduces_a_transcript(view):
    from app.sessions.transcript import Transcript

    transcript = Transcript()
    transcript.add(Source.PC, 0.0, 2.0, "Primo.")
    transcript.add(Source.MIC, 2.0, 4.0, "Secondo.")

    view.rebuild(transcript.segments)
    text = view.toPlainText()
    assert "Primo." in text
    assert "Secondo." in text


def test_rebuild_clears_what_was_there(view):
    view.append_confirmed(Source.PC, 0.0, "Vecchio testo.")
    view.rebuild([])
    assert view.toPlainText() == ""


def test_transcript_is_read_only_but_selectable(view):
    from PySide6.QtCore import Qt

    assert view.isReadOnly()
    flags = view.textInteractionFlags()
    assert flags & Qt.TextInteractionFlag.TextSelectableByMouse


def test_autoscroll_toggle(view):
    view.set_autoscroll(False)
    assert not view.autoscroll
    view.set_autoscroll(True)
    assert view.autoscroll


# -- meter -----------------------------------------------------------------

def test_meter_clamps_out_of_range_levels(qtbot):
    meter = LevelMeter(DARK)
    qtbot.addWidget(meter)

    meter.set_level(2.5)
    assert meter.level == 1.0
    meter.set_level(-1.0)
    assert meter.level == 0.0


def test_meter_colours_follow_the_hardware_convention(qtbot):
    """Green, then amber near full scale, then red at clipping."""
    meter = LevelMeter(DARK, segments=10)
    qtbot.addWidget(meter)

    assert meter._segment_colour(0).name() == DARK.meter_low.lower()
    assert meter._segment_colour(9).name() == DARK.meter_high.lower()


def test_inactive_meter_shows_no_level(qtbot):
    meter = LevelMeter(DARK)
    qtbot.addWidget(meter)

    meter.set_active(True)
    meter.set_level(0.8)
    meter.set_active(False)
    assert meter.level == 0.0


def test_recording_dot_animates_only_while_active(qtbot):
    dot = RecordingDot(DARK)
    qtbot.addWidget(dot)

    assert not dot._timer.isActive()
    dot.set_active(True)
    assert dot._timer.isActive()
    dot.set_active(False)
    assert not dot._timer.isActive()


# -- main window -----------------------------------------------------------

def test_window_opens_and_closes(qtbot):
    from app.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "LiveTranscriber"
    assert window.session is None


def test_window_starts_idle_with_setup_visible(qtbot):
    from app.sessions.session import SessionState
    from app.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window._update_state(SessionState.IDLE)

    assert window.setup_panel.isVisibleTo(window)
    assert not window.summary_bar.isVisibleTo(window)
    assert window.start_button.isVisibleTo(window)


def test_recording_collapses_setup_and_shows_transport(qtbot):
    """The shape change: setup gives way to the transcript while recording."""
    from app.sessions.session import SessionState
    from app.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window._update_state(SessionState.RECORDING)

    assert not window.setup_panel.isVisibleTo(window)
    assert window.summary_bar.isVisibleTo(window)
    assert not window.start_button.isVisibleTo(window)
    assert window.stop_button.isVisibleTo(window)
    assert window.pause_button.text() == "Pausa"


def test_paused_offers_resume(qtbot):
    from app.sessions.session import SessionState
    from app.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window._update_state(SessionState.PAUSED)

    assert window.pause_button.text() == "Riprendi"
    assert window.stop_button.isVisibleTo(window)


def test_device_rows_follow_the_selected_source(qtbot):
    from app.sessions.session import InputMode
    from app.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    window.mode_buttons[InputMode.MICROPHONE].setChecked(True)
    window._sync_device_rows()
    assert window.current_mode() is InputMode.MICROPHONE

    window.mode_buttons[InputMode.BOTH].setChecked(True)
    window._sync_device_rows()
    assert window.current_mode() is InputMode.BOTH


def test_model_list_offers_every_catalogue_entry(qtbot):
    from app.transcription import models
    from app.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    listed = {window.model_combo.itemData(i) for i in range(window.model_combo.count())}
    assert listed == {spec.key for spec in models.list_models()}


def test_language_list_leads_with_automatic(qtbot):
    from app.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    assert window.language_combo.itemData(0) == "auto"
    assert window.language_combo.itemData(1) == "it"


# -- hardware-aware default ------------------------------------------------

def test_default_model_is_lighter_without_a_gpu(qtbot, monkeypatch):
    """Streaming costs ~5x a one-shot run, so CPU cannot take the GPU default.

    Measured: small streaming leaves 7.7x headroom on GPU and 1.4x on this CPU,
    with latency spikes to 9.6s.
    """
    from app.transcription.hardware import AcceleratorChoice
    from app.transcription.models import get_spec
    from app.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    monkeypatch.setattr(
        "app.transcription.hardware.select_accelerator",
        lambda *a, **k: AcceleratorChoice(device="cpu", compute_type="int8"),
    )
    cpu_default = window._default_model()

    monkeypatch.setattr(
        "app.transcription.hardware.select_accelerator",
        lambda *a, **k: AcceleratorChoice(device="cuda", compute_type="float16"),
    )
    gpu_default = window._default_model()

    assert get_spec(cpu_default).approx_size_mb <= get_spec(gpu_default).approx_size_mb


def test_default_prefers_a_model_already_on_disk(qtbot, monkeypatch):
    """Offering to download on first launch is worse than a slightly smaller model."""
    from app.transcription import models
    from app.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    monkeypatch.setattr(models, "is_available", lambda key: key == "tiny")
    monkeypatch.setattr(models, "installed_models", lambda: [models.get_spec("tiny")])

    assert window._default_model() == "tiny"


def test_accelerator_chip_never_promises_more_than_the_build_has(qtbot, monkeypatch):
    """A CPU-only build must not display "GPU" just because a card is present."""
    from app.transcription.hardware import AcceleratorChoice
    from app.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    monkeypatch.setattr(
        "app.transcription.hardware.select_accelerator",
        lambda *a, **k: AcceleratorChoice(device="cpu", compute_type="int8"),
    )
    window._refresh_accelerator_chip()
    assert window.accel_chip.text() == "CPU"


def test_accelerator_chip_names_the_thread_count_on_a_cpu_machine(qtbot, monkeypatch):
    """On a PC without a GPU, the thread count is what explains the speed."""
    from app.transcription.hardware import AcceleratorChoice
    from app.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    monkeypatch.setattr(
        "app.transcription.hardware.select_accelerator",
        lambda *a, **k: AcceleratorChoice(
            device="cpu", compute_type="int8", cpu_threads=4
        ),
    )
    window._refresh_accelerator_chip()
    assert window.accel_chip.text() == "CPU · 4 thread"


def test_settings_survive_a_restart(qtbot, tmp_path, monkeypatch):
    """The settings dialog used to be decorative: nothing was ever stored."""
    from app.config.settings import AppSettings
    from app.transcription.hardware import Accelerator
    from app.ui.main_window import MainWindow

    settings = AppSettings()
    settings.accelerator = str(Accelerator.CPU)
    settings.show_timestamps = True
    settings.output_folder = str(tmp_path)

    window = MainWindow(settings)
    qtbot.addWidget(window)

    assert window.timestamps_button.isChecked()
    assert window.app_settings.accelerator_choice is Accelerator.CPU

    saved: list[bool] = []
    monkeypatch.setattr(type(settings), "save", lambda self: saved.append(True))
    window._persist_settings()
    assert saved == [True]
    assert settings.show_timestamps is True


def test_a_measured_cpu_gets_a_model_it_can_keep_up_with(qtbot, monkeypatch):
    """End to end: a stored measurement overrides the hardware guess.

    The preset for a CPU machine is `base`; a machine measured at 0.9 for `base`
    must not be handed it, because that is the case the whole calibration exists
    to catch.
    """
    from app.config.settings import AppSettings
    from app.transcription.hardware import Accelerator, AcceleratorChoice
    from app.ui.main_window import MainWindow

    settings = AppSettings()
    settings.accelerator = str(Accelerator.CPU)
    settings.remember_measurement("base", "cpu", "int8", 0.9)

    monkeypatch.setattr(
        "app.transcription.hardware.select_accelerator",
        lambda *a, **k: AcceleratorChoice(
            device="cpu", compute_type="int8", cpu_threads=4
        ),
    )
    # Every model on disk, stated rather than inherited: this used to depend on
    # which ones happened to be downloaded here, and broke the day some were
    # deleted to free space.
    monkeypatch.setattr("app.transcription.models.is_available", lambda key: True)

    window = MainWindow(settings)
    qtbot.addWidget(window)
    assert window._default_model() == "tiny"

    # Without the measurement, the same machine gets the preset.
    settings.speed_measurements = ""
    assert window._default_model() == "base"


def test_changing_theme_redraws_the_transcript(qtbot):
    """Text keeps the colour it was inserted with, so a theme change must redraw.

    Without this the transcript stayed dark-grey on a dark page after switching
    to the dark theme: still there, effectively unreadable.
    """
    from app.sessions.transcript import Source
    from app.ui.main_window import MainWindow
    from app.ui.theme import DARK, LIGHT, ThemeMode

    window = MainWindow()
    qtbot.addWidget(window)

    window.set_theme(ThemeMode.LIGHT)
    window.transcript.add(Source.PC, 0.0, 2.0, "Una frase di prova.")
    window.transcript_view.rebuild(window.transcript.segments)

    def ink_colours() -> set[str]:
        document = window.transcript_view.document()
        found = set()
        block = document.begin()
        while block.isValid():
            for fragment in block.begin():
                if fragment.fragment().isValid():
                    colour = fragment.fragment().charFormat().foreground().color()
                    found.add(colour.name().lower())
            block = block.next()
        return found

    light_inks = ink_colours()
    assert light_inks

    window.set_theme(ThemeMode.DARK)
    dark_inks = ink_colours()

    assert dark_inks != light_inks
    assert LIGHT.ink.lower() not in dark_inks
    assert DARK.ink.lower() in dark_inks


# -- the NVIDIA support pack -----------------------------------------------


def _availability(monkeypatch, usable: bool, can_install: bool):
    from app.transcription.hardware import GpuAvailability

    monkeypatch.setattr(
        "app.transcription.hardware.gpu_availability",
        lambda: GpuAvailability(usable, "motivo", can_install),
    )


def test_the_gpu_button_appears_only_when_it_would_do_something(qtbot, monkeypatch):
    """Three machines, three different right answers.

    A PC with no NVIDIA card has nothing to install; one that already has the
    libraries has nothing to gain; only the third case is a button.
    """
    from app.ui.main_window import MainWindow
    from app.ui.settings_window import SettingsDialog

    window = MainWindow()
    qtbot.addWidget(window)

    _availability(monkeypatch, usable=False, can_install=False)   # no card
    dialog = SettingsDialog(window)
    qtbot.addWidget(dialog)
    assert not dialog.cuda_button.isVisible()

    _availability(monkeypatch, usable=True, can_install=False)    # already able
    dialog = SettingsDialog(window)
    qtbot.addWidget(dialog)
    assert not dialog.cuda_button.isVisible()

    _availability(monkeypatch, usable=False, can_install=True)    # card, no libs
    dialog = SettingsDialog(window)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.cuda_button.isVisible()
    assert dialog.accelerator_hint.text() == "motivo"


# -- the theme, seen through Qt --------------------------------------------


def test_palette_for_accepts_what_qt_gives_back():
    """Qt stores a StrEnum as its string, so this gets "dark", not the member.

    `palette_for` compared with `is`, which a plain string never satisfies, so
    it silently fell through to "follow Windows" and returned the light palette
    on a light-configured PC.
    """
    from app.ui.theme import DARK, LIGHT, ThemeMode, palette_for

    assert palette_for("dark", system_is_dark=False) is DARK
    assert palette_for("light", system_is_dark=True) is LIGHT
    assert palette_for(ThemeMode.DARK, system_is_dark=False) is DARK
    # Anything unrecognisable still follows the system rather than raising.
    assert palette_for("nonsense", system_is_dark=True) is DARK


def test_opening_the_settings_does_not_change_the_theme(qtbot):
    """Reported: opening settings turned the window light, closing it fixed it.

    Loading the stored value into the combo fires currentIndexChanged, which
    applied the theme from `currentData()` — a string — and landed on the wrong
    palette until `apply_settings()` put the real one back on close.
    """
    from app.config.settings import AppSettings
    from app.ui.main_window import MainWindow
    from app.ui.settings_window import SettingsDialog
    from app.ui.theme import DARK, ThemeMode

    settings = AppSettings()
    settings.theme = str(ThemeMode.DARK)
    settings.first_run_done = True

    window = MainWindow(settings)
    qtbot.addWidget(window)
    assert window._palette is DARK

    dialog = SettingsDialog(window)
    qtbot.addWidget(dialog)
    assert window._palette is DARK, "the theme changed just by opening settings"

    dialog.reject()
    assert window._palette is DARK
