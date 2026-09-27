"""The page the app opens on.

Before this, launching LiveTranscriber dropped you straight into the instrument:
two device menus, a language, a model and a red button, with no word about what
the thing is or where the audio goes. That is the right screen once you know the
app, and the wrong one the first time you meet it.

So the door comes first. It says hello, states the one fact a person needs in
order to trust it — everything stays on this PC — and offers the only two things
there are to do here: transcribe, or change how transcription works.

The greeting uses the Windows account name, which is already on the screen the
user just came from and never leaves this machine. Anything that does not read
like a name (a service account, an address, a single letter) is answered with a
plain "Benvenuto" rather than an awkward guess.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

#: Spec §33, verbatim. The home page is where a person decides whether to trust
#: the app with the audio of their meetings, so the statement belongs here.
PRIVACY = (
    "LiveTranscriber elabora audio e trascrizioni localmente sul dispositivo. "
    "Nessun audio viene inviato a server esterni per la trascrizione."
)

#: Accounts that are not people. Windows hands out plenty of them, and
#: "Ciao Administrator." reads like a mail merge that went wrong.
NOT_NAMES = frozenset(
    {
        "admin",
        "administrator",
        "amministratore",
        "default",
        "desktop",
        "guest",
        "home",
        "ospite",
        "owner",
        "pc",
        "user",
        "utente",
    }
)

#: A measured column rather than the full window: the greeting and the two
#: choices read as one block of text, and text stops being readable well before
#: 820 px of line length.
COLUMN_WIDTH = 480


def greeting_for(user: str | None) -> str:
    """Greet by name when the account plausibly belongs to a person."""
    name = (user or "").strip()
    name = name.split("@")[0]              # an address: keep what precedes it
    name = name.replace("/", "\\").rsplit("\\", 1)[-1]   # DOMAIN\user
    parts = [part for part in re.split(r"[._]+", name) if part]

    # Every piece has to be a word. "svc_backup_01" and "user123" are accounts
    # rather than people, and "Ciao Svc." is worse than not using a name at all.
    if not parts or not all(part.replace("-", "").isalpha() for part in parts):
        return "Benvenuto."
    first = parts[0]
    if not (2 <= len(first) <= 20) or first.casefold() in NOT_NAMES:
        return "Benvenuto."
    # Only normalise a name written in one case, and by word, so a compound
    # first name keeps both halves: "jean-pierre" is Jean-Pierre, not Jean-pierre.
    display = first.title() if first.islower() or first.isupper() else first
    return f"Ciao {display}."


def current_user() -> str | None:
    import getpass

    try:
        return getpass.getuser()
    except Exception:
        return None


class ChoiceCard(QPushButton):
    """One of the two ways out of this page.

    A button rather than a clickable frame, so it arrives with the keyboard,
    the focus ring and the accessibility name already working. The title and
    the description live in child labels because a button's own text cannot
    carry two sizes; those labels pass their clicks through to the button.
    """

    def __init__(
        self, title: str, hint: str, primary: bool = False, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ChoiceCard")
        # Read by the stylesheet: the primary card carries the record colour on
        # its leading edge, which is the same colour the recording dot uses.
        self.setProperty("primary", "true" if primary else "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # A button sizes itself from its own text, which this one has none of,
        # so it collapsed to an empty pill with the labels clipped out of it.
        # These three say: take the size the layout inside actually needs.
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setAccessibleName(title)
        self.setAccessibleDescription(hint)

        box = QVBoxLayout(self)
        box.setContentsMargins(18, 14, 18, 14)
        box.setSpacing(3)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("ChoiceTitle")
        self.hint_label = QLabel(hint)
        self.hint_label.setObjectName("ChoiceHint")
        self.hint_label.setWordWrap(True)

        for label in (self.title_label, self.hint_label):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            box.addWidget(label)

    # -- the layout decides how big this is, not the (empty) button text

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self.layout().sizeHint()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self.layout().minimumSize()

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt override
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt override
        return self.layout().heightForWidth(width)


class HomeView(QWidget):
    """Greeting, the two choices, and the honest small print."""

    transcribe_requested = Signal()
    settings_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(0)

        column = QWidget()
        # Fixed, not maximum: a wrapped label's preferred width is a guess, and
        # the guess left the privacy statement three lines tall in the space of
        # two, printed over the line below it.
        column.setFixedWidth(COLUMN_WIDTH)
        inner = QVBoxLayout(column)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(0)

        wordmark = QLabel("LiveTranscriber")
        wordmark.setObjectName("Wordmark")

        self.greeting_label = QLabel(greeting_for(current_user()))
        self.greeting_label.setObjectName("Greeting")

        standfirst = QLabel(
            "Trascrivo quello che si sente sul PC o al microfono, mentre lo dicono."
        )
        standfirst.setObjectName("Standfirst")
        standfirst.setWordWrap(True)

        self.transcribe_card = ChoiceCard(
            "Trascrizione dal vivo",
            "Scegli la sorgente audio e comincia. Il testo compare mentre parlano.",
            primary=True,
        )
        self.settings_card = ChoiceCard(
            "Impostazioni",
            "Qualità, lingua, tema, cartella delle registrazioni, GPU NVIDIA.",
        )

        privacy = QLabel(PRIVACY)
        privacy.setObjectName("Fineprint")
        privacy.setWordWrap(True)

        self.status_label = QLabel()
        self.status_label.setObjectName("Fineprint")

        inner.addWidget(wordmark)
        inner.addSpacing(6)
        inner.addWidget(self.greeting_label)
        inner.addSpacing(4)
        inner.addWidget(standfirst)
        inner.addSpacing(26)
        inner.addWidget(self.transcribe_card)
        inner.addSpacing(10)
        inner.addWidget(self.settings_card)
        inner.addSpacing(28)
        inner.addWidget(privacy)
        inner.addSpacing(6)
        inner.addWidget(self.status_label)

        outer.addStretch(1)
        outer.addWidget(column, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)

        self.transcribe_card.clicked.connect(self.transcribe_requested)
        self.settings_card.clicked.connect(self.settings_requested)

    def set_status(self, accelerator: str, version: str) -> None:
        """Where transcription will run, and which build this is."""
        self.status_label.setText(f"{accelerator}   ·   versione {version}")

    def focus_default(self) -> None:
        self.transcribe_card.setFocus()
