"""Application entry point.

Sets up logging and high-DPI handling, then opens the main window. Any exception
that escapes a Qt slot would otherwise abort the process silently, so a hook
turns it into a log entry and a message the user can act on (spec §18).
"""

from __future__ import annotations

import contextlib
import logging
import sys
import types

from app.utils.logging_setup import setup_logging
from app.utils.paths import ensure_app_dirs

logger = logging.getLogger(__name__)


def _install_exception_hook(app) -> None:
    """Report crashes instead of vanishing.

    PySide6 terminates the process when an exception escapes a slot. Logging it
    and showing a plain message keeps a recording session recoverable and gives
    the user something to report.
    """
    from PySide6.QtWidgets import QMessageBox

    def handle(kind: type[BaseException], value: BaseException,
               tb: types.TracebackType | None) -> None:
        if issubclass(kind, KeyboardInterrupt):
            sys.__excepthook__(kind, value, tb)
            return

        logger.critical("Unhandled exception", exc_info=(kind, value, tb))
        message = getattr(value, "user_message", None) or (
            "Si è verificato un errore imprevisto.\n\n"
            "I dettagli tecnici sono nel file di log."
        )
        with contextlib.suppress(Exception):
            QMessageBox.critical(None, "LiveTranscriber", str(message))

    sys.excepthook = handle


def main() -> int:
    # Checked before Qt starts: the diagnostic must still run when something
    # about the GUI itself is broken.
    if "--selftest" in sys.argv:
        from app.selftest import main as selftest_main

        return selftest_main(show_dialog="--quiet" not in sys.argv)

    ensure_app_dirs()
    log_path = setup_logging()

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    # Fractional scaling: Windows at 125% and 150% is extremely common on
    # laptops, and rounding it to integers makes the interface either cramped
    # or oversized (spec §9).
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    # Fusion renders the same on every Windows theme and respects a stylesheet
    # consistently; the native style ignores parts of it depending on the
    # system theme, which would make the app look different machine to machine.
    app.setStyle("Fusion")
    app.setApplicationName("LiveTranscriber")
    app.setOrganizationName("LiveTranscriber")
    app.setApplicationDisplayName("LiveTranscriber")

    _install_exception_hook(app)
    logger.info("LiveTranscriber starting; log at %s", log_path)

    from app.audio.devices import terminate_pyaudio
    from app.config.settings import AppSettings
    from app.ui.main_window import MainWindow

    # Loaded here and handed in, so the window and the settings dialog work on
    # one object: two independent loads would let one overwrite the other's
    # changes on save.
    settings = AppSettings.load()

    window = MainWindow(settings)
    window.show()

    try:
        return app.exec()
    finally:
        terminate_pyaudio()
        logger.info("LiveTranscriber exited")


if __name__ == "__main__":
    raise SystemExit(main())
