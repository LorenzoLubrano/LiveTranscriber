"""The main window.

The window **changes shape** between two states, because the two states have
different jobs:

* **Idle** — setting up. Source, devices, language and model are what matter, so
  they get the space.
* **Recording** — watching. Those controls are now fixed for the session and
  only take attention away from the transcript, so they collapse into a single
  summary line and the transcript takes everything else.

Threading: the session runs entirely on worker threads and reports through
plain callbacks. Those callbacks arrive on the wrong thread for Qt, so each one
does nothing but emit a signal; the queued connection hands the work to the GUI
thread. Nothing that blocks — inference, device enumeration, model loading —
ever runs here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.audio import devices as audio_devices
from app.audio.devices import AudioDevice, DeviceError
from app.sessions.session import (
    InputMode,
    RecordingSession,
    SessionCallbacks,
    SessionConfig,
    SessionState,
)
from app.sessions.transcript import Source, Transcript, format_timestamp
from app.transcription import models
from app.transcription.streaming import StreamingUpdate
from app.ui.theme import (
    Palette,
    ThemeMode,
    palette_for,
    stylesheet,
    system_prefers_dark,
)
from app.ui.widgets.level_meter import LabelledMeter, RecordingDot
from app.ui.widgets.transcript_view import TranscriptView

logger = logging.getLogger(__name__)


class SessionBridge(QObject):
    """Moves session callbacks from worker threads onto the GUI thread."""

    updated = Signal(object, object)   # Source, StreamingUpdate
    state_changed = Signal(object)     # SessionState
    failed = Signal(str)
    warned = Signal(str)

    def callbacks(self) -> SessionCallbacks:
        return SessionCallbacks(
            on_update=lambda source, update: self.updated.emit(source, update),
            on_state=lambda state: self.state_changed.emit(state),
            on_error=self.failed.emit,
            on_warning=self.warned.emit,
        )


class MainWindow(QMainWindow):
    """LiveTranscriber's main window."""

    def __init__(self, settings=None) -> None:
        super().__init__()
        if settings is None:
            from app.config.settings import AppSettings

            settings = AppSettings.load()
        self.app_settings = settings
        self.transcript = Transcript()
        self.session: RecordingSession | None = None
        self.bridge = SessionBridge()

        self._theme_mode = ThemeMode.SYSTEM
        self._palette: Palette = palette_for(self._theme_mode, system_prefers_dark())
        self._loopbacks: list[AudioDevice] = []
        self._microphones: list[AudioDevice] = []

        self.setWindowTitle("LiveTranscriber")
        self.setMinimumSize(560, 520)
        self.resize(820, 780)

        self._build()
        self._connect()
        self._apply_theme()
        self.refresh_devices()
        self._restore_settings()
        self._update_state(SessionState.IDLE)

        # One timer drives the clock, the meters and the status chips.
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(100)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start()

        self.start_button.setFocus()

        # Deferred so the window is painted before a modal appears over it;
        # asking about a crash against a grey rectangle looks like a second crash.
        QTimer.singleShot(400, self.offer_recovery)

    # ------------------------------------------------------------------ UI

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        root.addLayout(self._build_titlebar())
        root.addWidget(self._build_setup_panel())
        root.addWidget(self._build_summary_bar())
        root.addWidget(self._build_status_strip())

        self.transcript_view = TranscriptView(self._palette)
        root.addWidget(self.transcript_view, 1)

        root.addWidget(self._build_search_bar())
        root.addLayout(self._build_transport())

    def _build_titlebar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        title = QLabel("LiveTranscriber")
        title.setObjectName("SectionTitle")

        self.accel_chip = QLabel()
        self.accel_chip.setObjectName("StatusChip")
        self.accel_chip.setToolTip(
            "Dove viene eseguita la trascrizione. Con una GPU NVIDIA è più veloce."
        )

        self.privacy_chip = QLabel("Tutto in locale")
        self.privacy_chip.setObjectName("StatusChip")
        self.privacy_chip.setToolTip(
            "L'audio e le trascrizioni restano sul tuo computer.\n"
            "Nessun audio viene inviato a server esterni."
        )

        self.settings_button = QPushButton("Impostazioni")
        self.settings_button.setObjectName("QuietButton")
        self.settings_button.setToolTip("Impostazioni avanzate (Ctrl+,)")

        row.addWidget(title)
        row.addStretch(1)
        row.addWidget(self.accel_chip)
        row.addWidget(self.privacy_chip)
        row.addWidget(self.settings_button)
        return row

    def _build_setup_panel(self) -> QWidget:
        self.setup_panel = QFrame()
        self.setup_panel.setObjectName("InstrumentPanel")

        outer = QVBoxLayout(self.setup_panel)
        outer.setContentsMargins(18, 16, 18, 18)
        outer.setSpacing(14)

        # -- source
        source_row = QHBoxLayout()
        source_row.setSpacing(20)
        source_label = QLabel("Cosa vuoi registrare")
        source_label.setObjectName("FieldLabel")

        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[InputMode, QRadioButton] = {}
        for mode in (InputMode.PC, InputMode.MICROPHONE, InputMode.BOTH):
            button = QRadioButton(mode.label)
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            source_row.addWidget(button)
        self.mode_buttons[InputMode.PC].setChecked(True)
        self.mode_buttons[InputMode.PC].setToolTip(
            "Registra l'audio riprodotto dal computer, non il microfono."
        )
        self.mode_buttons[InputMode.BOTH].setToolTip(
            "Registra entrambi e li tiene separati nel testo."
        )
        source_row.addStretch(1)

        outer.addWidget(source_label)
        outer.addLayout(source_row)

        # -- devices
        self.loopback_combo = QComboBox()
        self.loopback_combo.setToolTip("Il dispositivo da cui esce l'audio del computer.")
        self.mic_combo = QComboBox()
        self.mic_combo.setToolTip("Il microfono da registrare.")

        self.loopback_row = self._field("Audio dal computer", self.loopback_combo)
        self.mic_row = self._field("Microfono", self.mic_combo)
        outer.addLayout(self.loopback_row)
        outer.addLayout(self.mic_row)

        # -- language and model
        pair = QHBoxLayout()
        pair.setSpacing(14)
        self.language_combo = QComboBox()
        self.model_combo = QComboBox()
        self.model_combo.setToolTip(
            "Modelli più grandi sono più precisi e più lenti."
        )
        pair.addLayout(self._field("Lingua", self.language_combo), 1)
        pair.addLayout(self._field("Qualità", self.model_combo), 1)
        outer.addLayout(pair)

        # -- title
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Senza titolo")
        self.title_edit.setMaxLength(80)
        self.title_edit.setToolTip("Diventa il nome della cartella della registrazione.")
        outer.addLayout(self._field("Nome (facoltativo)", self.title_edit))

        self.model_hint = QLabel()
        self.model_hint.setObjectName("Hint")
        self.model_hint.setWordWrap(True)
        outer.addWidget(self.model_hint)

        return self.setup_panel

    def _build_summary_bar(self) -> QWidget:
        """One line replacing the setup panel while recording."""
        self.summary_bar = QFrame()
        self.summary_bar.setObjectName("SummaryBar")
        self.summary_bar.hide()

        row = QHBoxLayout(self.summary_bar)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(10)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("FieldLabel")
        row.addWidget(self.summary_label)
        row.addStretch(1)
        return self.summary_bar

    def _build_status_strip(self) -> QWidget:
        strip = QWidget()
        row = QHBoxLayout(strip)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(14)

        self.record_dot = RecordingDot(self._palette)
        self.state_label = QLabel("Pronto")
        self.state_label.setObjectName("FieldLabel")

        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setObjectName("ElapsedTime")
        self.elapsed_label.setProperty("recording", "false")

        meters = QWidget()
        meters_layout = QVBoxLayout(meters)
        meters_layout.setContentsMargins(0, 0, 0, 0)
        meters_layout.setSpacing(5)
        self.pc_meter = LabelledMeter("PC", self._palette.tag_pc, self._palette)
        self.mic_meter = LabelledMeter("MIC", self._palette.tag_mic, self._palette)
        meters_layout.addWidget(self.pc_meter)
        meters_layout.addWidget(self.mic_meter)
        # An Expanding meter inside a Preferred parent collapses to a few
        # pixels, which is exactly the widget that must never be hard to see.
        meters.setMinimumWidth(210)
        meters.setMaximumWidth(280)
        meters.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        row.addWidget(self.record_dot)
        row.addWidget(self.state_label)
        row.addStretch(1)
        row.addWidget(meters)
        row.addSpacing(8)
        row.addWidget(self.elapsed_label)
        return strip

    def _build_search_bar(self) -> QWidget:
        self.search_bar = QWidget()
        self.search_bar.hide()

        row = QHBoxLayout(self.search_bar)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Cerca nel testo")
        self.search_result = QLabel()
        self.search_result.setObjectName("Hint")
        close = QPushButton("Chiudi")
        close.setObjectName("QuietButton")
        close.clicked.connect(self.hide_search)

        row.addWidget(self.search_edit, 1)
        row.addWidget(self.search_result)
        row.addWidget(close)
        return self.search_bar

    def _build_transport(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self.start_button = QPushButton("Avvia registrazione")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.setMinimumWidth(210)

        self.pause_button = QPushButton("Pausa")
        self.stop_button = QPushButton("Termina")
        self.export_button = QPushButton("Esporta")
        self.export_button.setObjectName("QuietButton")
        self.export_button.setToolTip("Salva la trascrizione in un altro formato")

        self.folder_button = QPushButton("Apri cartella")
        self.folder_button.setObjectName("QuietButton")

        self.timestamps_button = QPushButton("Orari")
        self.timestamps_button.setObjectName("QuietButton")
        self.timestamps_button.setCheckable(True)
        self.timestamps_button.setToolTip("Mostra l'orario di ogni intervento")

        self.autoscroll_button = QPushButton("Segui")
        self.autoscroll_button.setObjectName("QuietButton")
        self.autoscroll_button.setCheckable(True)
        self.autoscroll_button.setChecked(True)
        self.autoscroll_button.setToolTip("Scorri automaticamente al testo più recente")

        row.addWidget(self.start_button)
        row.addWidget(self.pause_button)
        row.addWidget(self.stop_button)
        row.addStretch(1)
        row.addWidget(self.timestamps_button)
        row.addWidget(self.autoscroll_button)
        row.addWidget(self.export_button)
        row.addWidget(self.folder_button)
        return row

    @staticmethod
    def _field(label: str, widget: QWidget) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(5)
        caption = QLabel(label)
        caption.setObjectName("FieldLabel")
        box.addWidget(caption)
        box.addWidget(widget)
        return box

    # ------------------------------------------------------------- wiring

    def _connect(self) -> None:
        self.start_button.clicked.connect(self.start_recording)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_recording)
        self.folder_button.clicked.connect(self.open_folder)
        self.export_button.clicked.connect(self.export_transcript)
        self.settings_button.clicked.connect(self.open_settings)

        self.mode_group.buttonClicked.connect(lambda _: self._sync_device_rows())
        self.model_combo.currentIndexChanged.connect(self._sync_model_hint)

        self.timestamps_button.toggled.connect(self._on_timestamps_toggled)
        self.autoscroll_button.toggled.connect(self.transcript_view.set_autoscroll)

        self.transcript_view.search_requested.connect(self.show_search)
        self.search_edit.textChanged.connect(self._on_search)
        self.search_edit.returnPressed.connect(self._on_search)

        self.bridge.updated.connect(self._on_update)
        self.bridge.state_changed.connect(self._update_state)
        self.bridge.failed.connect(self._on_error)
        self.bridge.warned.connect(self._on_warning)

        QShortcut(QKeySequence("Ctrl+,"), self, self.open_settings)
        QShortcut(QKeySequence("Ctrl+R"), self, self._toggle_recording)
        QShortcut(QKeySequence("Escape"), self, self.hide_search)

        refresh = QAction("Aggiorna dispositivi", self)
        refresh.setShortcut(QKeySequence("F5"))
        refresh.triggered.connect(self.refresh_devices)
        self.addAction(refresh)

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet(self._palette))
        self.transcript_view.set_palette(self._palette)
        self.record_dot.set_palette(self._palette)
        self.pc_meter.set_palette(self._palette)
        self.mic_meter.set_palette(self._palette)

        self._refresh_accelerator_chip()

    def _refresh_accelerator_chip(self) -> None:
        """Show where transcription will really run.

        Asking detect_gpus() alone is not enough: a CPU-only build has no CUDA
        runtime, and nvidia-smi still reports the card. select_accelerator
        resolves both facts, so the chip never promises acceleration the build
        cannot deliver.
        """
        from app.transcription.hardware import describe_choice, select_accelerator

        try:
            choice = select_accelerator(self.app_settings.accelerator_choice)
        except Exception:
            logger.exception("Could not resolve the accelerator")
            self.accel_chip.setText("CPU")
            return

        if choice.is_gpu and choice.gpu is not None:
            self.accel_chip.setText(f"GPU {choice.gpu.short_name}")
        else:
            # Thread count on the chip, not just "CPU": on the machines without
            # a GPU it is the number that explains the speed they are getting.
            self.accel_chip.setText(
                f"CPU · {choice.cpu_threads} thread" if choice.cpu_threads else "CPU"
            )
        self.accel_chip.setToolTip(describe_choice(choice))

    def set_theme(self, mode: ThemeMode) -> None:
        self._theme_mode = mode
        self._palette = palette_for(mode, system_prefers_dark())
        self._apply_theme()

    # ------------------------------------------------------------ devices

    @Slot()
    def refresh_devices(self) -> None:
        """Re-enumerate audio devices. Cheap enough to run on the GUI thread."""
        try:
            audio_devices.refresh()
            self._loopbacks = audio_devices.list_loopbacks()
            self._microphones = audio_devices.list_microphones()
        except DeviceError as exc:
            self._on_error(exc.user_message)
            return

        self._fill_combo(self.loopback_combo, self._loopbacks)
        self._fill_combo(self.mic_combo, self._microphones)
        self._fill_languages()
        self._fill_models()
        self._sync_device_rows()

    @staticmethod
    def _fill_combo(combo: QComboBox, items: list[AudioDevice]) -> None:
        previous = combo.currentData()
        combo.clear()
        for device in items:
            combo.addItem(device.label, device)
        if not items:
            combo.addItem("Nessun dispositivo disponibile", None)
            combo.setEnabled(False)
        else:
            combo.setEnabled(True)
            index = combo.findData(previous)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                default = next((i for i, d in enumerate(items) if d.is_default), 0)
                combo.setCurrentIndex(default)

    def _fill_languages(self) -> None:
        from app.transcription.engine import PRIORITY_LANGUAGES

        if self.language_combo.count():
            return
        for code, label in PRIORITY_LANGUAGES:
            self.language_combo.addItem(label, code)

    def _fill_models(self) -> None:
        previous = self.model_combo.currentData()
        self.model_combo.clear()
        for spec in models.list_models():
            installed = models.is_available(spec.key)
            suffix = "" if installed else f"  — da scaricare ({spec.size_label})"
            self.model_combo.addItem(f"{spec.display_name}  ·  {spec.quality}{suffix}",
                                     spec.key)

        index = self.model_combo.findData(previous or self._default_model())
        self.model_combo.setCurrentIndex(max(0, index))
        self._sync_model_hint()

    def select_model(self, model_key: str) -> None:
        """Switch the model shown on the main screen (called from settings)."""
        index = self.model_combo.findData(model_key)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

    def apply_settings(self) -> None:
        """Re-read the stored settings after the settings dialog changed them."""
        self.set_theme(self.app_settings.theme_mode)
        self.timestamps_button.setChecked(self.app_settings.show_timestamps)
        self.autoscroll_button.setChecked(self.app_settings.autoscroll)
        self._refresh_accelerator_chip()

    # --------------------------------------------------------- persistence

    def _restore_settings(self) -> None:
        """Put back what the user chose last time.

        Runs after the device and model lists are filled, because the choices
        are restored by matching against what is actually available now: a
        microphone that has been unplugged must fall back to the default rather
        than leave the app pointing at nothing.
        """
        settings = self.app_settings
        self.set_theme(settings.theme_mode)

        try:
            button = self.mode_buttons[InputMode(settings.input_mode)]
        except (KeyError, ValueError):
            button = None
        if button is not None:
            button.setChecked(True)
            self._sync_device_rows()

        _select_device(self.loopback_combo, settings.loopback_device_key)
        _select_device(self.mic_combo, settings.microphone_device_key)

        index = self.language_combo.findData(settings.language)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)

        # Only a model that is on disk: restoring a choice that would now need a
        # download would turn "open the app" into "start a download".
        if models.is_available(settings.model):
            index = self.model_combo.findData(settings.model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

        self.timestamps_button.setChecked(settings.show_timestamps)
        self.autoscroll_button.setChecked(settings.autoscroll)

    def _persist_settings(self) -> None:
        """Remember the current choices. Never raises: closing must not fail."""
        settings = self.app_settings
        try:
            settings.theme = str(self._theme_mode)
            settings.input_mode = str(self.current_mode())
            loopback = self.loopback_combo.currentData()
            microphone = self.mic_combo.currentData()
            settings.loopback_device_key = loopback.key if loopback else ""
            settings.microphone_device_key = microphone.key if microphone else ""
            settings.language = self.language_combo.currentData() or "auto"
            settings.model = self.model_combo.currentData() or settings.model
            settings.show_timestamps = self.timestamps_button.isChecked()
            settings.autoscroll = self.autoscroll_button.isChecked()
            settings.save()
        except Exception:
            logger.exception("Could not save settings")

    def _default_model(self) -> str:
        """Pick a model this machine can actually keep up with.

        Streaming costs roughly five times a one-shot transcription, so the
        right default differs sharply with hardware: measured here, `small`
        streaming leaves 7.7x of headroom on this GPU but only 1.4x on the CPU,
        with latency spikes to 9.6s. Defaulting to the same model everywhere
        would make the app feel broken on exactly the machines that need the
        most help.

        A model already on disk wins over a better one that would have to be
        downloaded first.

        Measurements taken on this machine override the guess entirely. The
        hardware rule can only say "GPU or not", and that is not enough: on the
        development CPU, `base` costs 0.81 of real time and `small` costs 1.03 —
        one is usable and the other cannot keep up at all, on the same PC.
        """
        from app.transcription.calibration import cap_to_measurements
        from app.transcription.hardware import select_accelerator
        from app.transcription.models import Quality, recommend_model

        try:
            choice = select_accelerator(self.app_settings.accelerator_choice)
            has_gpu = choice.is_gpu
        except Exception:
            choice, has_gpu = None, False

        preferred = recommend_model(Quality.BALANCED, has_gpu).key
        if choice is not None:
            preferred = cap_to_measurements(
                preferred,
                lambda key: self.app_settings.measured_cost(
                    key, choice.device, choice.compute_type
                ),
                sources=2 if self.current_mode() is InputMode.BOTH else 1,
            )
        if models.is_available(preferred):
            return preferred

        # Fall back to the largest installed model this machine can sustain.
        ceiling = models.get_spec(preferred).approx_size_mb
        candidates = [
            s for s in models.installed_models() if s.approx_size_mb <= ceiling
        ]
        if candidates:
            return max(candidates, key=lambda s: s.approx_size_mb).key
        return preferred

    def _sync_model_hint(self) -> None:
        key = self.model_combo.currentData()
        if not key:
            return
        spec = models.get_spec(key)
        if models.is_available(key):
            self.model_hint.setText(spec.description)
        else:
            self.model_hint.setText(
                f"{spec.description}  Il modello deve essere scaricato "
                f"({spec.size_label}); serve la connessione una sola volta."
            )

    def _sync_device_rows(self) -> None:
        mode = self.current_mode()
        wants_pc = mode in (InputMode.PC, InputMode.BOTH)
        wants_mic = mode in (InputMode.MICROPHONE, InputMode.BOTH)
        _set_row_visible(self.loopback_row, wants_pc)
        _set_row_visible(self.mic_row, wants_mic)

    def current_mode(self) -> InputMode:
        for mode, button in self.mode_buttons.items():
            if button.isChecked():
                return mode
        return InputMode.PC

    # ---------------------------------------------------------- recording

    @Slot()
    def _toggle_recording(self) -> None:
        if self.session and self.session.is_active:
            self.stop_recording()
        else:
            self.start_recording()

    @Slot()
    def start_recording(self) -> None:
        if self.session and self.session.is_active:
            return

        model_key = self.model_combo.currentData() or models.DEFAULT_MODEL
        if not models.is_available(model_key):
            self._offer_download(model_key)
            return

        config = SessionConfig(
            mode=self.current_mode(),
            loopback_device=self.loopback_combo.currentData(),
            microphone_device=self.mic_combo.currentData(),
            model=model_key,
            accelerator=self.app_settings.accelerator_choice,
            language=self.language_combo.currentData() or "auto",
            title=self.title_edit.text().strip(),
            output_root=self.app_settings.output_path,
            streaming=self.app_settings.streaming_settings(),
            vad=self.app_settings.vad_settings(),
        )

        self.transcript.clear()
        self.transcript_view.clear()
        self.transcript_view.set_placeholder()

        self.session = RecordingSession(config, self.bridge.callbacks(), self.transcript)
        self.start_button.setEnabled(False)
        self.state_label.setText("Avvio…")
        QApplication.processEvents()

        try:
            self.session.start()
        except DeviceError as exc:
            self.session = None
            self._on_error(exc.user_message)
            self._update_state(SessionState.IDLE)
        except Exception as exc:
            self.session = None
            self._on_error(getattr(exc, "user_message", "Impossibile avviare la registrazione."))
            logger.exception("Session start failed")
            self._update_state(SessionState.IDLE)

    @Slot()
    def toggle_pause(self) -> None:
        if self.session is None:
            return
        if self.session.state is SessionState.RECORDING:
            self.session.pause()
        elif self.session.state is SessionState.PAUSED:
            self.session.resume()

    @Slot()
    def stop_recording(self) -> None:
        if self.session is None:
            return
        self.stop_button.setEnabled(False)
        self.state_label.setText("Salvataggio…")
        QApplication.processEvents()

        directory = self.session.stop()
        self.session = None
        if directory:
            self.state_label.setText(f"Salvato in {directory.name}")

    def _offer_download(self, model_key: str) -> None:
        spec = models.get_spec(model_key)
        answer = QMessageBox.question(
            self,
            "Scaricare il modello?",
            f"Il modello {spec.display_name} deve essere scaricato.\n\n"
            f"Dimensione: {spec.size_label}\n\n"
            "Serve la connessione a Internet una sola volta. "
            "Dopo il download l'app funziona senza connessione.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._download_model(model_key)

    def _download_model(self, model_key: str) -> None:
        from app.ui.download_dialog import ModelDownloadDialog

        dialog = ModelDownloadDialog(model_key, self)
        if dialog.exec() == ModelDownloadDialog.DialogCode.Accepted:
            self._fill_models()
            index = self.model_combo.findData(model_key)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            self._offer_speed_check(model_key)

    def _offer_speed_check(self, model_key: str) -> None:
        """Offer to measure this PC, once, after the first model arrives.

        Only on the CPU path, and only once: a GPU has so much headroom that
        measuring answers a question nobody has, while on a CPU the answer
        decides whether the app is usable at all. Asked rather than done, because
        it takes the better part of a minute on exactly those machines.
        """
        from app.transcription.hardware import select_accelerator

        if self.app_settings.first_run_done:
            return
        try:
            choice = select_accelerator(self.app_settings.accelerator_choice)
        except Exception:
            return
        if choice.is_gpu:
            self.app_settings.first_run_done = True
            return

        answer = QMessageBox.question(
            self,
            "Prova di velocità",
            "Vuoi misurare quanto margine ha questo PC con il modello scelto?\n\n"
            "Serve meno di un minuto, funziona senza connessione e non registra "
            "nulla. Senza la misura l'app non può sapere se questo PC riesce a "
            "trascrivere dal vivo.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        self.app_settings.first_run_done = True
        if answer != QMessageBox.StandardButton.Yes:
            return

        from app.transcription.calibration import suggest_model
        from app.ui.speed_dialog import SpeedTestDialog

        speed = SpeedTestDialog(
            model_key, accelerator=self.app_settings.accelerator_choice, parent=self
        )
        speed.exec()
        measured = speed.result
        if measured is None or not measured.is_usable:
            return

        self.app_settings.remember_measurement(
            measured.model_key, measured.device, measured.compute_type, measured.cost
        )
        suggested = suggest_model(measured)
        if suggested != model_key and models.is_available(suggested):
            self.select_model(suggested)

    # --------------------------------------------------------- transcript

    @Slot(object, object)
    def _on_update(self, source: Source, update: StreamingUpdate) -> None:
        if update.confirmed:
            self.transcript_view.append_confirmed(
                source, update.confirmed[0].start, update.confirmed_text
            )
        if update.provisional:
            self.transcript_view.set_provisional(source, update.provisional_text)
        else:
            self.transcript_view.clear_provisional()

    def _on_timestamps_toggled(self, enabled: bool) -> None:
        self.transcript_view.set_show_timestamps(enabled)
        self.transcript_view.rebuild(self.transcript.segments)

    # ------------------------------------------------------------- search

    @Slot()
    def show_search(self) -> None:
        self.search_bar.show()
        self.search_edit.setFocus()
        self.search_edit.selectAll()

    @Slot()
    def hide_search(self) -> None:
        self.search_bar.hide()
        self.search_edit.clear()
        self.transcript_view.setFocus()

    def _on_search(self) -> None:
        query = self.search_edit.text()
        if not query.strip():
            self.search_result.clear()
            return
        hits = self.transcript.search(query)
        self.search_result.setText(
            "nessun risultato" if not hits else f"{len(hits)} risultati"
        )
        if hits:
            self.transcript_view.find(query)

    # -------------------------------------------------------------- state

    @Slot(object)
    def _update_state(self, state: SessionState) -> None:
        recording = state is SessionState.RECORDING
        paused = state is SessionState.PAUSED
        active = recording or paused

        self.setup_panel.setVisible(not active)
        self.summary_bar.setVisible(active)
        if active:
            self.summary_label.setText(self._summary_text())

        self.start_button.setVisible(not active)
        self.start_button.setEnabled(not active)
        self.pause_button.setVisible(active)
        self.stop_button.setVisible(active)
        self.stop_button.setEnabled(active)
        self.pause_button.setText("Riprendi" if paused else "Pausa")

        self.record_dot.set_active(recording)
        self.elapsed_label.setProperty("recording", "true" if active else "false")
        self.elapsed_label.style().unpolish(self.elapsed_label)
        self.elapsed_label.style().polish(self.elapsed_label)

        mode = self.current_mode()
        self.pc_meter.setVisible(mode in (InputMode.PC, InputMode.BOTH))
        self.mic_meter.setVisible(mode in (InputMode.MICROPHONE, InputMode.BOTH))
        self.pc_meter.set_active(recording)
        self.mic_meter.set_active(recording)

        self.state_label.setText(
            {
                SessionState.IDLE: "Pronto",
                SessionState.STARTING: "Avvio…",
                SessionState.RECORDING: "Registrazione in corso",
                SessionState.PAUSED: "In pausa",
                SessionState.STOPPING: "Salvataggio…",
                SessionState.STOPPED: "Registrazione salvata",
            }.get(state, "")
        )

    def _summary_text(self) -> str:
        mode = self.current_mode()
        model = self.model_combo.currentData() or ""
        language = self.language_combo.currentText()
        parts = [mode.label]
        if model:
            parts.append(models.get_spec(model).display_name)
        parts.append(language)
        return "   ·   ".join(parts)

    def _tick(self) -> None:
        if self.session is None:
            return
        self.elapsed_label.setText(
            format_timestamp(self.session.elapsed, always_hours=True)
        )
        for source, stats in self.session.stats().items():
            meter = self.pc_meter if source is Source.PC else self.mic_meter
            meter.set_level(stats.level)

    # ------------------------------------------------------------- errors

    @Slot(str)
    def _on_error(self, message: str) -> None:
        QMessageBox.warning(self, "LiveTranscriber", message)

    @Slot(str)
    def _on_warning(self, message: str) -> None:
        self.state_label.setText(message)

    # -------------------------------------------------------------- misc

    @Slot()
    def open_folder(self) -> None:
        import os

        from app.utils.paths import default_recordings_dir

        target: Path = (
            self.session.directory
            if self.session and self.session.directory
            else default_recordings_dir()
        )
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(target)  # noqa: S606 - opening a folder the app owns

    @Slot()
    def export_transcript(self) -> None:
        """Save the transcript in a chosen format."""
        from app.export import get_format, list_formats

        if not len(self.transcript):
            QMessageBox.information(
                self, "LiveTranscriber", "Non c'è ancora testo da esportare."
            )
            return

        filters = ";;".join(fmt.filter_string for fmt in list_formats())
        suggested = self._last_directory() / "transcript.txt"
        chosen, selected_filter = QFileDialog.getSaveFileName(
            self, "Esporta la trascrizione", str(suggested), filters
        )
        if not chosen:
            return

        fmt = next(
            (f for f in list_formats() if f.filter_string == selected_filter),
            None,
        ) or get_format(Path(chosen).suffix.lstrip(".").lower() or "txt")

        path = Path(chosen)
        if path.suffix.lower() != fmt.extension:
            path = path.with_suffix(fmt.extension)

        try:
            fmt.write(self.transcript, path, self.title_edit.text().strip())
        except Exception as exc:
            self._on_error(
                getattr(exc, "user_message", "Impossibile salvare il file.")
            )
            return
        self.state_label.setText(f"Esportato in {path.name}")

    def _last_directory(self) -> Path:
        from app.utils.paths import default_recordings_dir

        if self.session is not None and self.session.directory:
            return self.session.directory
        return default_recordings_dir()

    @Slot()
    def offer_recovery(self) -> None:
        """Ask about an interrupted recording, if there is one (spec §14)."""
        from app.sessions.recovery import find_incomplete

        try:
            sessions = find_incomplete()
        except Exception:
            logger.exception("Could not scan for interrupted sessions")
            return
        if not sessions:
            return

        from app.ui.recovery_dialog import RecoveryDialog

        dialog = RecoveryDialog(sessions, self)
        if dialog.exec() != RecoveryDialog.DialogCode.Accepted:
            return
        if dialog.recovered is None or not len(dialog.recovered):
            self.state_label.setText("Registrazione recuperata (solo audio)")
            return

        self.transcript.load(dialog.recovered.to_dicts())
        self.transcript_view.rebuild(self.transcript.segments)
        name = dialog.recovered_session.directory.name if dialog.recovered_session else ""
        self.state_label.setText(f"Recuperata: {name}")

    @Slot()
    def open_settings(self) -> None:
        from app.ui.settings_window import SettingsDialog

        dialog = SettingsDialog(self)
        dialog.exec()

    def choose_output_folder(self) -> Path | None:
        from app.utils.paths import default_recordings_dir

        chosen = QFileDialog.getExistingDirectory(
            self, "Cartella delle registrazioni", str(default_recordings_dir())
        )
        return Path(chosen) if chosen else None

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        """Spec §10: never lose a recording to a stray window close."""
        if self.session is not None and self.session.is_active:
            answer = QMessageBox.question(
                self,
                "Registrazione in corso",
                "È in corso una registrazione.\n\nVuoi interromperla e salvare?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer != QMessageBox.StandardButton.Save:
                event.ignore()
                return
            self.stop_recording()

        self._tick_timer.stop()
        self._persist_settings()
        audio_devices.terminate_pyaudio()
        event.accept()


def _select_device(combo: QComboBox, key: str) -> None:
    """Select the device with this key, if it is still present.

    Keys rather than indices: PortAudio renumbers devices whenever anything is
    plugged in, so an index would quietly select a different device after a
    reboot.
    """
    if not key:
        return
    for index in range(combo.count()):
        device = combo.itemData(index)
        if device is not None and getattr(device, "key", "") == key:
            combo.setCurrentIndex(index)
            return


def _set_row_visible(layout: QVBoxLayout, visible: bool) -> None:
    for i in range(layout.count()):
        item = layout.itemAt(i)
        widget = item.widget() if item else None
        if widget is not None:
            widget.setVisible(visible)
