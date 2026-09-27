"""The first thing a new user sees (spec §25).

Without it, someone opening the app for the first time meets a list of six
models named Tiny, Base, Small, Medium, Large-v3 and Turbo, which means nothing
unless you already know what Whisper is. The choice that actually matters to
them is much simpler — fast, balanced, or as good as it gets — and the app is
the one that knows which model that translates to *on this PC*.

So the question is asked in their terms, and answered in ours: each option shows
the model it will use, what it costs to download, and what to expect from it
here. The line above says where transcription will run, which is the honest
place to tell someone with no NVIDIA card that the CPU is doing the work.

It also carries the privacy statement, because the first run is when a person
decides whether to trust an app with the audio of their meetings.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from app.transcription import models
from app.transcription.calibration import cap_to_measurements
from app.transcription.hardware import describe_choice, select_accelerator
from app.transcription.models import Quality
from app.ui.sizing import keep_wrapped_text_whole

logger = logging.getLogger(__name__)


class FirstRunDialog(QDialog):
    """Asks what kind of transcription the user wants, once."""

    def __init__(
        self,
        settings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.app_settings = settings
        self.chosen_quality: Quality | None = None
        self.chosen_model: str = ""

        self.setWindowTitle("Benvenuto in LiveTranscriber")
        self.setModal(True)
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        # A minimum width alone switches off Qt's own minimum height, so the
        # dialog could be squeezed until its wrapped text was cut.
        self._wrap_guard = keep_wrapped_text_whole(self)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(14)

        heading = QLabel("Che tipo di trascrizione ti serve?")
        heading.setObjectName("SectionTitle")
        heading.setWordWrap(True)
        layout.addWidget(heading)

        intro = QLabel(
            "Puoi cambiare idea in qualsiasi momento dal menù «Qualità» nella "
            "finestra principale."
        )
        intro.setObjectName("Hint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(self._hardware_line())
        layout.addSpacing(4)

        self.group = QButtonGroup(self)
        for quality in Quality:
            option = self._option(quality)
            layout.addWidget(option)

        layout.addSpacing(6)
        layout.addWidget(self._privacy_note())

        layout.addSpacing(10)

        buttons = QDialogButtonBox()
        self.continue_button = buttons.addButton(
            "Continua", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.continue_button.setObjectName("DialogPrimary")
        buttons.addButton("Scelgo dopo", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # -- the pieces -------------------------------------------------------

    def _accelerator(self):
        try:
            return select_accelerator(self.app_settings.accelerator_choice)
        except Exception:
            logger.exception("Could not resolve the accelerator for the first run")
            return None

    def _hardware_line(self) -> QLabel:
        choice = self._accelerator()
        text = describe_choice(choice) if choice is not None else "CPU"
        label = QLabel(f"Su questo PC la trascrizione userà: {text}")
        label.setObjectName("Hint")
        label.setWordWrap(True)
        return label

    def _model_for(self, quality: Quality) -> str:
        """The model this preset means here, after any measurements."""
        choice = self._accelerator()
        has_gpu = bool(choice and choice.is_gpu)
        preferred = models.recommend_model(quality, has_gpu).key
        if choice is None:
            return preferred
        return cap_to_measurements(
            preferred,
            lambda key: self.app_settings.measured_cost(
                key, choice.device, choice.compute_type
            ),
        )

    def _meaning(self, quality: Quality) -> str:
        """What the choice means, in consequences rather than in model names.

        The catalogue descriptions are written for the settings screen, where
        the reader already knows what a model is; reusing them here would tell a
        first-time user that Small is "the default choice", which it is not, and
        that its latency is contained, which on a CPU it is not.
        """
        choice = self._accelerator()
        has_gpu = bool(choice and choice.is_gpu)

        if quality is Quality.FAST:
            return (
                "Il testo compare quasi subito. Qualità più bassa: buona per "
                "seguire, meno per una trascrizione da rileggere."
            )
        if quality is Quality.BALANCED:
            return (
                "Buona qualità con qualche secondo di ritardo. È la scelta "
                "adatta alla maggior parte dei casi."
            )
        if has_gpu:
            return "La resa migliore, e la tua scheda NVIDIA la regge senza fatica."
        return (
            "La resa migliore, ma su un PC senza scheda NVIDIA può non stare al "
            "passo con l'audio dal vivo. L'app te lo dirà se succede."
        )

    def _option(self, quality: Quality) -> QWidget:
        """One choice, with the concrete consequences of picking it."""
        key = self._model_for(quality)
        spec = models.get_spec(key)
        installed = models.is_available(key)

        frame = QFrame()
        frame.setObjectName("InstrumentPanel")
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(14, 12, 14, 12)
        inner.setSpacing(4)

        button = QRadioButton(quality.label)
        # The value, not the member: Qt stores a StrEnum as its string, so
        # reading the property back gives "balanced" rather than the enum, and
        # an identity check against Quality.BALANCED quietly fails.
        button.setProperty("quality", quality.value)
        button.setProperty("model", key)
        if quality is Quality.BALANCED:
            button.setChecked(True)
        self.group.addButton(button)
        inner.addWidget(button)

        meaning = QLabel(self._meaning(quality))
        meaning.setWordWrap(True)
        inner.addWidget(meaning)

        # The cost, always — including for a model already on disk, because
        # "which one is this going to use" is a fair question at any time.
        cost = QLabel(
            f"{spec.display_name} · {spec.size_label} · "
            + ("già scaricato" if installed else "da scaricare una volta")
        )
        cost.setObjectName("Hint")
        cost.setWordWrap(True)
        inner.addWidget(cost)

        # Clicking anywhere on the card selects it: a 16 px radio dot is a small
        # target, and the card is the thing that looks clickable.
        frame.mousePressEvent = lambda event, b=button: b.setChecked(True)  # type: ignore[method-assign]
        frame.setCursor(Qt.CursorShape.PointingHandCursor)

        # The whole card says which one is chosen, not only its 16 px dot: three
        # identical boxes made the eye hunt for the answer.
        def mark(checked: bool, card: QFrame = frame) -> None:
            card.setProperty("selected", "true" if checked else "false")
            card.style().unpolish(card)
            card.style().polish(card)

        button.toggled.connect(mark)
        mark(button.isChecked())
        return frame

    def _privacy_note(self) -> QLabel:
        label = QLabel(
            "LiveTranscriber elabora audio e trascrizioni localmente sul "
            "dispositivo. Nessun audio viene inviato a server esterni per la "
            "trascrizione. Serve la connessione solo per scaricare il modello, "
            "una volta."
        )
        label.setObjectName("Hint")
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        return label

    # -- result -----------------------------------------------------------

    def accept(self) -> None:
        button = self.group.checkedButton()
        if button is not None:
            self.chosen_quality = Quality(button.property("quality"))
            self.chosen_model = button.property("model")
        super().accept()
