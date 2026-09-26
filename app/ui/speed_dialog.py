"""Measuring what this PC can transcribe live, with a way out.

Exists because the app cannot otherwise know. Core count predicts almost
nothing, and the numbers that circulate for Whisper measure transcribing a file,
which costs about a fifth of transcribing a live stream. So the app measures the
machine it is on, using the real pipeline.

The measurement loads a model and runs inference, so it happens on a worker
thread: on a slow CPU — exactly the case worth measuring — it takes the better
part of a minute.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.transcription import models
from app.transcription.calibration import StreamingCost, Verdict
from app.transcription.hardware import Accelerator

logger = logging.getLogger(__name__)


class _SpeedWorker(QObject):
    """Runs one measurement and reports back by signal."""

    progress = Signal(float)
    finished = Signal(object)      # StreamingCost
    failed = Signal(str)

    def __init__(self, model_key: str, accelerator: Accelerator) -> None:
        super().__init__()
        self.model_key = model_key
        self.accelerator = accelerator
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        from app.transcription.calibration import measure_streaming_cost

        try:
            measured = measure_streaming_cost(
                self.model_key,
                accelerator=self.accelerator,
                progress=self.progress.emit,
                cancelled=lambda: self._cancelled,
            )
        except Exception as exc:
            logger.exception("Speed measurement failed")
            self.failed.emit(
                getattr(exc, "user_message", "Misura non riuscita.")
            )
            return
        self.finished.emit(measured)


class SpeedTestDialog(QDialog):
    """Modal measurement for one model, with the verdict in plain words."""

    def __init__(
        self,
        model_key: str,
        accelerator: Accelerator = Accelerator.AUTO,
        sources: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.spec = models.get_spec(model_key)
        self.sources = sources
        self.result: StreamingCost | None = None
        self._worker: _SpeedWorker | None = None
        self._thread: QThread | None = None

        self.setWindowTitle("Velocità di questo PC")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        heading = QLabel(f"Prova di velocità con {self.spec.display_name}")
        heading.setObjectName("SectionTitle")

        self.detail = QLabel(
            "L'app trascrive una frase di prova per misurare quanto margine "
            "ha questo PC sul tempo reale. Non serve la connessione e non "
            "viene registrato nulla."
        )
        self.detail.setObjectName("Hint")
        self.detail.setWordWrap(True)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)

        self.status = QLabel("Caricamento del modello…")
        self.status.setObjectName("Hint")
        self.status.setWordWrap(True)

        buttons = QHBoxLayout()
        self.close_button = QPushButton("Annulla")
        self.close_button.clicked.connect(self._cancel)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)

        layout.addWidget(heading)
        layout.addWidget(self.detail)
        layout.addWidget(self.bar)
        layout.addWidget(self.status)
        layout.addLayout(buttons)

        self._start(accelerator)

    # -- running ----------------------------------------------------------

    def _start(self, accelerator: Accelerator) -> None:
        self._thread = QThread(self)
        self._worker = _SpeedWorker(self.spec.key, accelerator)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    @Slot(float)
    def _on_progress(self, fraction: float) -> None:
        self.bar.setValue(int(100 * max(0.0, min(1.0, fraction))))
        self.status.setText("Misura in corso…")

    @Slot(object)
    def _on_finished(self, measured: StreamingCost) -> None:
        self._shutdown()
        self.result = measured
        self.bar.setValue(100)
        self.status.setText(self._verdict_text(measured))
        self.close_button.setText("Chiudi")
        self._reconnect_close(self.accept)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._shutdown()
        self.bar.setValue(0)
        self.status.setText(message)
        self.close_button.setText("Chiudi")
        self._reconnect_close(self.reject)

    def _verdict_text(self, measured: StreamingCost) -> str:
        """The result in the terms the user cares about: does it keep up?"""
        if not measured.is_usable:
            return (
                "La misura non ha prodotto alcuna trascrizione, quindi non dice "
                "niente di affidabile su questo PC."
            )

        head = (
            f"{self.spec.display_name} su {measured.device}: "
            f"{measured.headroom:.1f}× più veloce del tempo reale."
        )
        verdict = measured.verdict_for(self.sources)
        if verdict is Verdict.TOO_SLOW:
            return (
                f"{head} Non basta per trascrivere dal vivo: il testo "
                "arriverebbe con ritardo crescente. Conviene un modello più "
                "leggero."
            )
        if verdict is Verdict.TIGHT:
            return (
                f"{head} È al limite: funziona, ma con poco margine. "
                "Registrare PC e microfono insieme raddoppia il carico."
            )
        return f"{head} C'è margine abbondante per trascrivere dal vivo."

    # -- teardown ---------------------------------------------------------

    def _reconnect_close(self, handler) -> None:
        self.close_button.clicked.disconnect()
        self.close_button.clicked.connect(handler)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.status.setText("Annullamento…")
        self.close_button.setEnabled(False)

    def _shutdown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(10000)
            self._thread = None
        self._worker = None

    def reject(self) -> None:
        self._shutdown()
        super().reject()
