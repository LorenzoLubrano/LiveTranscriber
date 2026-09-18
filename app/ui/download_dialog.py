"""Model download, with progress and a way out.

The download runs on a worker thread; a multi-gigabyte fetch on the GUI thread
would freeze the window (spec §4). Cancelling leaves the partial download in
place so a retry resumes rather than starting again.
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

logger = logging.getLogger(__name__)


class _DownloadWorker(QObject):
    """Runs the download and reports back by signal."""

    progress = Signal(int, int)   # bytes done, bytes total
    finished = Signal()
    failed = Signal(str)

    def __init__(self, model_key: str) -> None:
        super().__init__()
        self.model_key = model_key
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            models.download(
                self.model_key,
                on_progress=lambda done, total: self.progress.emit(done, total),
                cancel=lambda: self._cancelled,
            )
        except Exception as exc:
            self.failed.emit(getattr(exc, "user_message", "Download non riuscito."))
            return
        self.finished.emit()


class ModelDownloadDialog(QDialog):
    """Modal progress dialog for one model download."""

    def __init__(self, model_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = models.get_spec(model_key)
        self._worker: _DownloadWorker | None = None
        self._thread: QThread | None = None

        self.setWindowTitle("Download del modello")
        self.setModal(True)
        self.setMinimumWidth(430)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        heading = QLabel(f"Download di {self.spec.display_name}")
        heading.setObjectName("SectionTitle")

        self.detail = QLabel(
            f"{self.spec.size_label} da scaricare. "
            "Al termine l'app funziona senza connessione."
        )
        self.detail.setObjectName("Hint")
        self.detail.setWordWrap(True)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)

        self.status = QLabel("Connessione…")
        self.status.setObjectName("Hint")

        buttons = QHBoxLayout()
        self.cancel_button = QPushButton("Annulla")
        self.cancel_button.clicked.connect(self._cancel)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)

        layout.addWidget(heading)
        layout.addWidget(self.detail)
        layout.addWidget(self.bar)
        layout.addWidget(self.status)
        layout.addLayout(buttons)

        self._start()

    def _start(self) -> None:
        self._thread = QThread(self)
        self._worker = _DownloadWorker(self.spec.key)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    @Slot(int, int)
    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.bar.setRange(0, 100)
            self.bar.setValue(int(100 * done / total))
            self.status.setText(
                f"{done / 1_048_576:.0f} MB di {total / 1_048_576:.0f} MB"
            )
        else:
            self.bar.setRange(0, 0)  # indeterminate until a total is known
            self.status.setText(f"{done / 1_048_576:.0f} MB")

    @Slot()
    def _on_finished(self) -> None:
        self._shutdown()
        self.accept()

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._shutdown()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.status.setText(message)
        self.cancel_button.setText("Chiudi")
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.reject)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.status.setText(
            "Annullamento… il download riprenderà da qui la prossima volta."
        )
        self.cancel_button.setEnabled(False)

    def _shutdown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
        self._worker = None

    def reject(self) -> None:
        self._shutdown()
        super().reject()
