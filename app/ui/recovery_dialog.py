"""Offering an interrupted recording back to the user (spec §14).

Shown once at startup when a previous session never finished. The wording is
deliberately calm and concrete: it says what exists, not what went wrong. The
user does not need a diagnosis of the crash — they need to know their lecture is
still there.

Both choices are safe. *Recupera* writes the exports next to the audio; *Ignora*
stops offering the session and deletes nothing.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.sessions.recovery import RecoverableSession, dismiss, recover
from app.sessions.transcript import Transcript

logger = logging.getLogger(__name__)


class RecoveryDialog(QDialog):
    """Asks whether to recover interrupted recordings."""

    def __init__(
        self, sessions: list[RecoverableSession], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.sessions = sessions
        self.recovered: Transcript | None = None
        self.recovered_session: RecoverableSession | None = None

        self.setWindowTitle("Registrazione non terminata")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        heading = QLabel(
            "È stata rilevata una registrazione non terminata."
            if len(sessions) == 1
            else f"Sono state rilevate {len(sessions)} registrazioni non terminate."
        )
        heading.setObjectName("SectionTitle")
        heading.setWordWrap(True)

        detail = QLabel(
            "L'audio e il testo trascritto fino all'interruzione sono ancora sul "
            "disco. Puoi recuperarli adesso oppure lasciarli dove sono."
        )
        detail.setObjectName("Hint")
        detail.setWordWrap(True)

        self.list = QListWidget()
        self.list.setAlternatingRowColors(False)
        for session in sessions:
            item = QListWidgetItem(session.summary())
            item.setData(Qt.ItemDataRole.UserRole, session)
            item.setToolTip(str(session.directory))
            self.list.addItem(item)
        self.list.setCurrentRow(0)
        self.list.setMaximumHeight(150)

        buttons = QDialogButtonBox()
        self.recover_button = buttons.addButton(
            "Recupera", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.recover_button.setObjectName("DialogPrimary")
        self.ignore_button = buttons.addButton(
            "Ignora", QDialogButtonBox.ButtonRole.DestructiveRole
        )
        self.recover_button.clicked.connect(self._recover)
        self.ignore_button.clicked.connect(self._ignore)

        note = QHBoxLayout()
        note_label = QLabel("Nessun file viene eliminato in entrambi i casi.")
        note_label.setObjectName("Hint")
        note.addWidget(note_label)
        note.addStretch(1)

        layout.addWidget(heading)
        layout.addWidget(detail)
        layout.addWidget(self.list)
        layout.addLayout(note)
        layout.addWidget(buttons)

    # -- actions ----------------------------------------------------------

    def _selected(self) -> RecoverableSession | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _recover(self) -> None:
        session = self._selected()
        if session is None:
            self.reject()
            return
        try:
            self.recovered = recover(session)
            self.recovered_session = session
        except Exception:
            logger.exception("Recovery failed for %s", session.directory)
            self.recovered = None
        self.accept()

    def _ignore(self) -> None:
        """Stop offering every listed session, without deleting anything."""
        for session in self.sessions:
            try:
                dismiss(session)
            except Exception:
                logger.exception("Could not dismiss %s", session.directory)
        self.reject()
