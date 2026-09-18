"""Settings, in two tiers.

The first tab holds the handful of things a normal user changes: appearance,
where recordings go, which model. Everything with a number in it lives behind
the second tab, because exposing `beam_size` and `int8_float16` on the main
screen makes the app look like it is for someone else (spec §9, §25).

The advanced tab always offers a way back to the recommended values, so a user
who has experimented is never stranded.
"""

from __future__ import annotations

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
        form.addRow("Elaborazione", self.accelerator_combo)

        form.addRow(QLabel(""))
        models_label = QLabel(self._models_summary())
        models_label.setObjectName("Hint")
        models_label.setWordWrap(True)
        form.addRow("Modelli", models_label)

        return page

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
        mode = self.theme_combo.currentData()
        if mode is not None and hasattr(self.main_window, "set_theme"):
            self.main_window.set_theme(mode)
