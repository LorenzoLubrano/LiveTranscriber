"""Settings, in two tiers.

The first tab holds the handful of things a normal user changes: appearance,
where recordings go, which model. Everything with a number in it lives behind
the second tab, because exposing `beam_size` and `int8_float16` on the main
screen makes the app look like it is for someone else (spec §9, §25).

The advanced tab always offers a way back to the recommended values, so a user
who has experimented is never stranded.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.transcription import models
from app.transcription.hardware import Accelerator, describe_hardware
from app.ui.theme import ThemeMode
from app.utils.paths import app_data_dir, logs_dir

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.main_window = parent

        self.setWindowTitle("Impostazioni")
        self.setModal(True)
        self.setMinimumSize(560, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(14)

        tabs = QTabWidget()
        tabs.addTab(self._general_tab(), "Generale")
        tabs.addTab(self._advanced_tab(), "Avanzate")
        tabs.addTab(self._about_tab(), "Informazioni")
        layout.addWidget(tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._load_values()

    # -- reading and writing the stored settings --------------------------

    def _load_values(self) -> None:
        """Show what is actually stored.

        Without this the dialog is decorative: it opens on its defaults whatever
        the user chose last time, and every change is lost on close.
        """
        settings = self._settings()
        if settings is None:
            return

        index = self.theme_combo.findData(settings.theme_mode)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)

        self.folder_edit.setText(str(settings.output_path))
        self.timestamps_check.setChecked(settings.show_timestamps)
        self.autoscroll_check.setChecked(settings.autoscroll)

        index = self.accelerator_combo.findData(settings.accelerator_choice)
        model = self.accelerator_combo.model()
        item = model.item(index) if index >= 0 and hasattr(model, "item") else None
        # A stored GPU preference on a PC that cannot honour it stays stored, but
        # is not shown as the active choice — the app is running on the CPU.
        if index >= 0 and (item is None or item.isEnabled()):
            self.accelerator_combo.setCurrentIndex(index)

        self.chunk_spin.setValue(settings.chunk_duration)
        self.buffer_spin.setValue(settings.max_buffer_duration)
        self.agreement_spin.setValue(settings.agreement_runs)
        self.vad_check.setChecked(settings.vad_enabled)
        self.vad_spin.setValue(settings.vad_threshold)
        self.silence_spin.setValue(settings.min_silence_duration_ms)
        self.speech_spin.setValue(settings.min_speech_duration_ms)
        self.beam_spin.setValue(settings.beam_size)
        self.threads_spin.setValue(settings.cpu_threads)

    def _store_values(self) -> None:
        """Save the choices. Never raises: closing a dialog must not fail."""
        settings = self._settings()
        if settings is None:
            return
        try:
            settings.theme = str(self.theme_combo.currentData())
            settings.output_folder = self.folder_edit.text().strip()
            settings.show_timestamps = self.timestamps_check.isChecked()
            settings.autoscroll = self.autoscroll_check.isChecked()
            settings.accelerator = str(self.accelerator_combo.currentData())
            settings.chunk_duration = self.chunk_spin.value()
            settings.max_buffer_duration = self.buffer_spin.value()
            settings.agreement_runs = self.agreement_spin.value()
            settings.vad_enabled = self.vad_check.isChecked()
            settings.vad_threshold = self.vad_spin.value()
            settings.min_silence_duration_ms = self.silence_spin.value()
            settings.min_speech_duration_ms = self.speech_spin.value()
            settings.beam_size = self.beam_spin.value()
            settings.cpu_threads = self.threads_spin.value()
            settings.save()
        except Exception:
            logger.exception("Could not save settings")

        if hasattr(self.main_window, "apply_settings"):
            self.main_window.apply_settings()

    def accept(self) -> None:
        self._store_values()
        super().accept()

    def reject(self) -> None:
        # Close is the only button, so treat it as "keep what I changed": every
        # control here either applies live or is harmless, and silently
        # discarding a deliberate change is the more surprising behaviour.
        self._store_values()
        super().reject()

    # -- general ----------------------------------------------------------

    def _general_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(16, 18, 16, 16)
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.theme_combo = QComboBox()
        for mode in ThemeMode:
            self.theme_combo.addItem(mode.label, mode)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        form.addRow("Aspetto", self.theme_combo)

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        browse = QPushButton("Cambia")
        browse.clicked.connect(self._choose_folder)
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse)
        folder_widget = QWidget()
        folder_widget.setLayout(folder_row)
        form.addRow("Registrazioni", folder_widget)

        self.timestamps_check = QCheckBox("Mostra gli orari nel testo")
        self.autoscroll_check = QCheckBox("Segui il testo più recente")
        self.autoscroll_check.setChecked(True)
        form.addRow("", self.timestamps_check)
        form.addRow("", self.autoscroll_check)

        self.accelerator_combo = QComboBox()
        self.accelerator_combo.addItem("Automatico", Accelerator.AUTO)
        self.accelerator_combo.addItem("Solo CPU", Accelerator.CPU)
        self.accelerator_combo.addItem("GPU NVIDIA", Accelerator.GPU)
        self._restrict_accelerator_choices()

        accelerator_row = QHBoxLayout()
        accelerator_row.addWidget(self.accelerator_combo, 1)
        self.cuda_button = QPushButton("Attiva la GPU…")
        self.cuda_button.clicked.connect(self._install_cuda_pack)
        accelerator_row.addWidget(self.cuda_button)
        accelerator_widget = QWidget()
        accelerator_widget.setLayout(accelerator_row)
        form.addRow("Elaborazione", accelerator_widget)

        self.accelerator_hint = QLabel()
        self.accelerator_hint.setObjectName("Hint")
        self.accelerator_hint.setWordWrap(True)
        form.addRow("", self.accelerator_hint)
        self._sync_accelerator_hint()

        form.addRow(QLabel(""))
        self.models_label = QLabel(self._models_summary())
        self.models_label.setObjectName("Hint")
        self.models_label.setWordWrap(True)
        form.addRow("Modelli", self.models_label)

        speed_row = QHBoxLayout()
        self.speed_button = QPushButton("Misura la velocità")
        self.speed_button.clicked.connect(self._measure_speed)
        self.speed_label = QLabel(self._speed_summary())
        self.speed_label.setObjectName("Hint")
        self.speed_label.setWordWrap(True)
        speed_row.addWidget(self.speed_button)
        speed_row.addWidget(self.speed_label, 1)
        speed_widget = QWidget()
        speed_widget.setLayout(speed_row)
        form.addRow("Velocità", speed_widget)

        return page

    def _sync_accelerator_hint(self) -> None:
        """Say where the work will run, and offer the fix when there is one."""
        from app.transcription.hardware import gpu_availability

        try:
            availability = gpu_availability()
        except Exception:
            self.cuda_button.setVisible(False)
            return

        self.accelerator_hint.setText(availability.reason)
        # The button exists only when pressing it would change something: a PC
        # with no NVIDIA card has nothing to install, and one that already has
        # the libraries has nothing to gain.
        self.cuda_button.setVisible(availability.can_install)

    @Slot()
    def _install_cuda_pack(self) -> None:
        from app.transcription.hardware import detect_gpus
        from app.ui.cuda_dialog import CudaPackDialog

        try:
            name = detect_gpus()[0].short_name
        except Exception:
            name = ""

        dialog = CudaPackDialog(name, self)
        dialog.exec()
        if not dialog.installed:
            return

        self._restrict_accelerator_choices()
        self._sync_accelerator_hint()
        if hasattr(self.main_window, "apply_settings"):
            self.main_window.apply_settings()

    def _speed_summary(self) -> str:
        """What has been measured on this machine, if anything.

        Shown because it is the one number that explains the app's behaviour on
        a PC without a GPU, and because "measured here" is a very different
        claim from "should be about".
        """
        settings = self._settings()
        if settings is None:
            return "Non misurata."

        from app.transcription.calibration import StreamingCost
        from app.transcription.hardware import select_accelerator

        try:
            choice = select_accelerator(settings.accelerator_choice)
            cost = settings.measured_cost(
                settings.model, choice.device, choice.compute_type
            )
        except Exception:
            return "Non misurata."
        if not cost:
            return "Non ancora misurata su questo PC."

        measured = StreamingCost(
            model_key=settings.model,
            device=choice.device,
            compute_type=choice.compute_type,
            cost=cost,
            audio_seconds=0.0,
            runs=1,
        )
        return measured.summary

    def _settings(self):
        return getattr(self.main_window, "app_settings", None)

    @Slot()
    def _measure_speed(self) -> None:
        """Measure the configured model on this machine, and store the result."""
        from app.ui.speed_dialog import SpeedTestDialog

        settings = self._settings()
        if settings is None:
            return

        if not models.is_available(settings.model):
            self.speed_label.setText(
                "Scarica prima un modello: la misura usa quello configurato."
            )
            return

        dialog = SpeedTestDialog(
            settings.model,
            accelerator=settings.accelerator_choice,
            parent=self,
        )
        dialog.exec()

        measured = dialog.result
        if measured is None or not measured.is_usable:
            self.speed_label.setText(self._speed_summary())
            return

        settings.remember_measurement(
            measured.model_key, measured.device, measured.compute_type, measured.cost
        )
        settings.save()
        self.speed_label.setText(measured.summary)
        self._offer_lighter_model(measured)

    def _offer_lighter_model(self, measured) -> None:
        """Propose a model that fits, and let the user decide.

        Switching on their behalf would change the quality of their transcripts
        without asking; saying nothing would leave them with a model that cannot
        keep up. So: offer, and do it only on a yes.
        """
        from PySide6.QtWidgets import QMessageBox

        from app.transcription.calibration import Verdict, suggest_model

        if measured.verdict is not Verdict.TOO_SLOW:
            return
        suggested = suggest_model(measured)
        if suggested == measured.model_key:
            return

        settings = self._settings()
        spec = models.get_spec(suggested)
        answer = QMessageBox.question(
            self,
            "Modello troppo pesante",
            f"{models.get_spec(measured.model_key).display_name} non riesce a "
            f"stare al passo su questo PC.\n\nVuoi passare a "
            f"{spec.display_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes or settings is None:
            return

        settings.model = suggested
        settings.save()
        if hasattr(self.main_window, "select_model"):
            self.main_window.select_model(suggested)
        self.models_label.setText(self._models_summary())
        self.speed_label.setText(self._speed_summary())

    def _restrict_accelerator_choices(self) -> None:
        """Grey out "GPU NVIDIA" when it cannot do anything, and say why.

        A setting that can be selected and then silently does nothing is worse
        than one that is missing: the user changes it, sees no difference, and
        concludes the app is broken rather than that their PC has no NVIDIA card.
        """
        from app.transcription.hardware import gpu_availability

        try:
            availability = gpu_availability()
        except Exception:
            return

        index = self.accelerator_combo.findData(Accelerator.GPU)
        if index < 0:
            return

        model = self.accelerator_combo.model()
        item = model.item(index) if hasattr(model, "item") else None
        if availability.usable:
            if item is not None:
                item.setToolTip(availability.reason)
            return

        if item is not None:
            item.setEnabled(False)
            item.setToolTip(availability.reason)
        self.accelerator_combo.setToolTip(availability.reason)
        if self.accelerator_combo.currentIndex() == index:
            self.accelerator_combo.setCurrentIndex(
                self.accelerator_combo.findData(Accelerator.AUTO)
            )

    def _models_summary(self) -> str:
        installed = models.installed_models()
        if not installed:
            return "Nessun modello scaricato."
        total = sum(models.disk_usage_mb(s.key) for s in installed)
        names = ", ".join(s.display_name for s in installed)
        return f"{names}\n{total} MB in {app_data_dir() / 'models'}"

    # -- advanced ---------------------------------------------------------

    def _advanced_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(16, 18, 16, 16)
        outer.setSpacing(12)

        warning = QLabel(
            "Questi valori sono già regolati per la maggior parte dei casi. "
            "Cambiali solo se sai cosa stai facendo."
        )
        warning.setObjectName("Hint")
        warning.setWordWrap(True)
        outer.addWidget(warning)

        form = QFormLayout()
        form.setSpacing(11)

        self.chunk_spin = QDoubleSpinBox()
        self.chunk_spin.setRange(0.5, 30.0)
        self.chunk_spin.setSingleStep(0.5)
        self.chunk_spin.setSuffix(" s")
        self.chunk_spin.setValue(2.0)
        self.chunk_spin.setToolTip(
            "Ogni quanto viene aggiornato il testo. Valori bassi = meno attesa, "
            "più carico sul processore."
        )
        form.addRow("Intervallo di trascrizione", self.chunk_spin)

        self.buffer_spin = QDoubleSpinBox()
        self.buffer_spin.setRange(5.0, 120.0)
        self.buffer_spin.setSuffix(" s")
        self.buffer_spin.setValue(28.0)
        form.addRow("Buffer massimo", self.buffer_spin)

        self.agreement_spin = QSpinBox()
        self.agreement_spin.setRange(1, 4)
        self.agreement_spin.setValue(2)
        self.agreement_spin.setToolTip(
            "Quante analisi devono concordare prima di considerare il testo "
            "definitivo. Più alto = più accurato ma più lento."
        )
        form.addRow("Conferme richieste", self.agreement_spin)

        self.vad_check = QCheckBox("Salta le parti senza voce")
        self.vad_check.setChecked(True)
        self.vad_check.setToolTip(
            "Evita di trascrivere il silenzio, che altrimenti produce testo inventato."
        )
        form.addRow("", self.vad_check)

        self.vad_spin = QDoubleSpinBox()
        self.vad_spin.setRange(0.1, 0.95)
        self.vad_spin.setSingleStep(0.05)
        self.vad_spin.setValue(0.5)
        form.addRow("Sensibilità alla voce", self.vad_spin)

        self.silence_spin = QSpinBox()
        self.silence_spin.setRange(100, 5000)
        self.silence_spin.setSingleStep(100)
        self.silence_spin.setSuffix(" ms")
        self.silence_spin.setValue(500)
        form.addRow("Durata minima della pausa", self.silence_spin)

        self.speech_spin = QSpinBox()
        self.speech_spin.setRange(50, 2000)
        self.speech_spin.setSingleStep(50)
        self.speech_spin.setSuffix(" ms")
        self.speech_spin.setValue(250)
        form.addRow("Durata minima del parlato", self.speech_spin)

        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setValue(5)
        form.addRow("Ampiezza di ricerca", self.beam_spin)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 64)
        self.threads_spin.setValue(0)
        self.threads_spin.setSpecialValueText("Automatico")
        form.addRow("Thread CPU", self.threads_spin)

        outer.addLayout(form)
        outer.addStretch(1)

        reset = QPushButton("Ripristina valori consigliati")
        reset.clicked.connect(self._reset_advanced)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(reset)
        outer.addLayout(row)

        return page

    @Slot()
    def _reset_advanced(self) -> None:
        self.chunk_spin.setValue(2.0)
        self.buffer_spin.setValue(28.0)
        self.agreement_spin.setValue(2)
        self.vad_check.setChecked(True)
        self.vad_spin.setValue(0.5)
        self.silence_spin.setValue(500)
        self.speech_spin.setValue(250)
        self.beam_spin.setValue(5)
        self.threads_spin.setValue(0)

    # -- about ------------------------------------------------------------

    def _about_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(12)

        from app import __version__

        title = QLabel(f"LiveTranscriber {__version__}")
        title.setObjectName("SectionTitle")

        privacy = QLabel(
            "LiveTranscriber elabora audio e trascrizioni localmente sul dispositivo.\n\n"
            "Nessun audio viene inviato a server esterni per la trascrizione.\n"
            "Nessuna telemetria, nessuna analisi d'uso, nessun account.\n\n"
            "L'unico accesso a Internet avviene quando scarichi un modello, "
            "una sola volta. Dopo il download l'app funziona completamente offline."
        )
        privacy.setWordWrap(True)

        hardware = QLabel(describe_hardware())
        hardware.setObjectName("Hint")

        paths = QLabel(
            f"Dati: {app_data_dir()}\n"
            f"Log:  {logs_dir()}"
        )
        paths.setObjectName("Hint")
        paths.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        licence = QLabel(
            "Licenza MIT. Usa Whisper (OpenAI, MIT), CTranslate2 (MIT), "
            "Qt tramite PySide6 (LGPL) e libsoxr (LGPL)."
        )
        licence.setObjectName("Hint")
        licence.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(privacy)
        layout.addSpacing(6)
        layout.addWidget(hardware)
        layout.addWidget(paths)
        layout.addStretch(1)
        layout.addWidget(licence)
        return page

    # -- actions ----------------------------------------------------------

    @Slot()
    def _choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Cartella delle registrazioni", self.folder_edit.text()
        )
        if chosen:
            self.folder_edit.setText(chosen)

    @Slot()
    def _on_theme_changed(self) -> None:
        # currentData() returns the string Qt stored, not the enum member.
        try:
            mode = ThemeMode(self.theme_combo.currentData())
        except ValueError:
            return
        if hasattr(self.main_window, "set_theme"):
            self.main_window.set_theme(mode)
