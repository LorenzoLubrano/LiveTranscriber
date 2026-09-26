"""Downloading NVIDIA support, with progress and a way out.

The download is half a gigabyte, so it runs on a worker thread and can be
stopped. Nothing is installed until the archive has been downloaded whole and
its checksum matched, so cancelling — or losing the connection — leaves the
machine exactly as it was.
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

from app.transcription import cuda_pack

logger = logging.getLogger(__name__)


class _PackWorker(QObject):
    progress = Signal(int, int)
    finished = Signal()
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            cuda_pack.install(
                on_progress=lambda done, total: self.progress.emit(done, total),
                cancel=lambda: self._cancelled,
            )
        except cuda_pack.Cancelled:
            self.failed.emit("Download annullato. Non è stato installato nulla.")
            return
        except Exception as exc:
            logger.exception("CUDA pack installation failed")
            self.failed.emit(
                getattr(exc, "user_message", "Installazione non riuscita.")
            )
            return
        self.finished.emit()


class CudaPackDialog(QDialog):
    """Modal download of the NVIDIA support pack."""

    def __init__(self, gpu_name: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.installed = False
        self._worker: _PackWorker | None = None
        self._thread: QThread | None = None

        self.setWindowTitle("Supporto NVIDIA")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        heading = QLabel(
            f"Attivare la GPU {gpu_name}" if gpu_name else "Attivare la GPU NVIDIA"
        )
        heading.setObjectName("SectionTitle")
        heading.setWordWrap(True)

        self.detail = QLabel(
            f"Vengono scaricate le librerie di calcolo NVIDIA ({cuda_pack.size_label()}). "
            "Sono pubblicate da NVIDIA e vengono verificate prima di essere "
            "installate. Serve una volta sola: dopo, l'app funziona di nuovo "
            "senza connessione."
        )
        self.detail.setObjectName("Hint")
        self.detail.setWordWrap(True)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)

        self.status = QLabel("Pronto per il download.")
        self.status.setObjectName("Hint")
        self.status.setWordWrap(True)

        buttons = QHBoxLayout()
        self.action_button = QPushButton("Scarica")
        self.action_button.clicked.connect(self._start)
        self.close_button = QPushButton("Chiudi")
        self.close_button.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        buttons.addWidget(self.action_button)

        layout.addWidget(heading)
        layout.addWidget(self.detail)
        layout.addWidget(self.bar)
        layout.addWidget(self.status)
        layout.addLayout(buttons)

    # -- running ----------------------------------------------------------

    @Slot()
    def _start(self) -> None:
        self.action_button.setEnabled(False)
        self.close_button.setText("Annulla")
        self.close_button.clicked.disconnect()
        self.close_button.clicked.connect(self._cancel)
        self.status.setText("Connessione…")

        self._thread = QThread(self)
        self._worker = _PackWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    @Slot(int, int)
    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.bar.setValue(int(100 * done / total))
        self.status.setText(
            f"{done / 1_048_576:.0f} MB di {total / 1_048_576:.0f} MB"
        )

    @Slot()
    def _on_finished(self) -> None:
        self._shutdown()
        # Registration is cached per process, so a pack installed just now has
        # to be announced or the GPU would stay unused until the next launch.
        from app.transcription.cuda_setup import ensure_cuda_libraries, forget_registration

        forget_registration()
        ensure_cuda_libraries()

        self.installed = True
        self.bar.setValue(100)
        self.status.setText(
            "Installato. La GPU verrà usata dalla prossima registrazione."
        )
        self.action_button.setVisible(False)
        self.close_button.setText("Fatto")
        self.close_button.clicked.disconnect()
        self.close_button.clicked.connect(self.accept)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._shutdown()
        self.bar.setValue(0)
        self.status.setText(message)
        self.action_button.setEnabled(True)
        self.action_button.setText("Riprova")
        self.close_button.setText("Chiudi")
        self.close_button.clicked.disconnect()
        self.close_button.clicked.connect(self.reject)

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
