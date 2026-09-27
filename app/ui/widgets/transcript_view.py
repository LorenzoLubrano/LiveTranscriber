"""The transcript: a reading surface, not a log.

This is where the product's one distinctive idea lives. Text arrives in two
states and they must be visibly different, because the difference is real:

* **confirmed** — two transcription passes agreed on it. It will not change.
  Full contrast, upright.
* **provisional** — the current best guess, which the next pass may rewrite.
  Lower contrast and italic, so it reads as *not yet settled*.

Watching provisional text resolve into confirmed text is the clearest possible
explanation of what the software is doing, so it is shown rather than explained.

Set in a serif at a generous line height. Transcription tools default to a
monospace log; that is fine for machine output and hostile for two hours of
prose. What the user is producing here is a document, so it is set like one.

Rendered with ``QTextEdit`` and incremental appends rather than rebuilt HTML:
a four-hour lecture is a large document, and re-rendering it on every update
would make the interface heavier the longer it ran.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QKeySequence,
    QShortcut,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import QTextEdit, QWidget

from app.sessions.transcript import Source, format_timestamp
from app.ui.theme import TYPE, Palette

#: How many characters a line of transcript may run to. Typographic practice
#: puts comfortable reading between 45 and 75 characters; a serif face carries
#: the upper end. Without a limit the text simply filled the window, which on a
#: maximised 16:9 screen meant lines of 110 characters and a reader who loses
#: the start of the next line.
MEASURE_CHARACTERS = 78


class TranscriptView(QTextEdit):
    """Read-only transcript with confirmed and provisional text."""

    search_requested = Signal()

    def __init__(self, palette: Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._show_timestamps = False
        self._autoscroll = True
        self._last_source: Source | None = None

        #: Character position where provisional text begins, or -1.
        self._provisional_anchor = -1

        self.setObjectName("TranscriptView")
        self.setReadOnly(True)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QTextEdit.Shape.NoFrame)
        # Wrapping at a width we choose rather than at the window edge.
        self.setLineWrapMode(QTextEdit.LineWrapMode.FixedPixelWidth)

        self._apply_document_style()
        self._install_shortcuts()
        self.set_placeholder()

    # -- setup ------------------------------------------------------------

    def _apply_document_style(self) -> None:
        font = QFont(TYPE.reading_family, TYPE.reading)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self.setFont(font)
        # Paragraph spacing lives in the block format; line height is set per
        # block as text is appended.
        self.document().setDocumentMargin(0)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._keep_the_measure()

    def _keep_the_measure(self) -> None:
        """Wrap at the measure, or at the window if the window is narrower."""
        ideal = int(QFontMetricsF(self.font()).averageCharWidth() * MEASURE_CHARACTERS)
        self.setLineWrapColumnOrWidth(max(160, min(ideal, self.viewport().width())))

    def _install_shortcuts(self) -> None:
        # Ctrl+A and Ctrl+C come free with a read-only QTextEdit; Ctrl+F is ours.
        find = QShortcut(QKeySequence.StandardKey.Find, self)
        find.activated.connect(self.search_requested.emit)

    def set_placeholder(self) -> None:
        self.clear()
        self._provisional_anchor = -1
        self._last_source = None
        self.setPlaceholderText(
            "Il testo comparirà qui mentre registri."
        )

    # -- appearance -------------------------------------------------------

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette

    @property
    def show_timestamps(self) -> bool:
        return self._show_timestamps

    def set_show_timestamps(self, show: bool) -> None:
        self._show_timestamps = show

    @property
    def autoscroll(self) -> bool:
        return self._autoscroll

    def set_autoscroll(self, enabled: bool) -> None:
        self._autoscroll = enabled
        if enabled:
            self.scroll_to_end()

    # -- formats ----------------------------------------------------------

    def _tag_format(self, source: Source) -> QTextCharFormat:
        fmt = QTextCharFormat()
        colour = self._palette.tag_pc if source is Source.PC else self._palette.tag_mic
        fmt.setForeground(QColor(colour))
        fmt.setFontFamilies([TYPE.ui_family, "Segoe UI", "sans-serif"])
        fmt.setFontPointSize(TYPE.small)
        fmt.setFontWeight(QFont.Weight.DemiBold)
        return fmt

    def _time_format(self) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self._palette.ink_faint))
        fmt.setFontFamilies([TYPE.mono_family, "Consolas", "monospace"])
        fmt.setFontPointSize(TYPE.micro)
        return fmt

    def _confirmed_format(self) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self._palette.ink))
        fmt.setFontFamilies([TYPE.reading_family, "Cambria", "Georgia", "serif"])
        fmt.setFontPointSize(TYPE.reading)
        fmt.setFontItalic(False)
        return fmt

    def _provisional_format(self) -> QTextCharFormat:
        """Visibly unsettled: softer and italic, so it reads as in-progress."""
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self._palette.ink_faint))
        fmt.setFontFamilies([TYPE.reading_family, "Cambria", "Georgia", "serif"])
        fmt.setFontPointSize(TYPE.reading)
        fmt.setFontItalic(True)
        return fmt

    # -- writing ----------------------------------------------------------

    def append_confirmed(self, source: Source, start: float, text: str) -> None:
        """Add text that will not change.

        A new speaker tag is written only when the source changes, so a single
        speaker reads as continuous prose instead of a tag on every fragment.
        """
        if not text.strip():
            return

        self._clear_provisional()
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        new_speaker = source is not self._last_source
        if new_speaker:
            if self._last_source is not None:
                cursor.insertBlock()
                cursor.insertBlock()
            self._insert_heading(cursor, source, start)
            cursor.insertBlock()
        elif cursor.block().text():
            # Only separate from existing text; a space at the head of a fresh
            # paragraph shows up as a stray indent.
            cursor.insertText(" ", self._confirmed_format())

        cursor.insertText(text.strip(), self._confirmed_format())
        self._last_source = source
        self._provisional_anchor = -1

        if self._autoscroll:
            self.scroll_to_end()

    def set_provisional(self, source: Source, text: str) -> None:
        """Replace the unsettled tail. Safe to call on every update."""
        self._clear_provisional()
        if not text.strip():
            return

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        if source is not self._last_source and self._last_source is not None:
            cursor.insertBlock()
            cursor.insertBlock()
            self._insert_heading(cursor, source, 0.0, timestamp=False)
            cursor.insertBlock()
            self._last_source = source
            self._provisional_anchor = cursor.position()
        elif self._last_source is None:
            self._insert_heading(cursor, source, 0.0, timestamp=False)
            cursor.insertBlock()
            self._last_source = source
            self._provisional_anchor = cursor.position()
        else:
            # The anchor is taken *before* the separating space so that removing
            # provisional text removes the space with it. Anchoring after it
            # leaves the space behind, and the next confirmed chunk adds its own
            # — which is where the double spaces at every join came from.
            self._provisional_anchor = cursor.position()
            if cursor.block().text():
                cursor.insertText(" ", self._provisional_format())

        cursor.insertText(text.strip(), self._provisional_format())

        if self._autoscroll:
            self.scroll_to_end()

    def _insert_heading(
        self, cursor: QTextCursor, source: Source, start: float, timestamp: bool = True
    ) -> None:
        if timestamp and self._show_timestamps:
            cursor.insertText(
                f"{format_timestamp(start, always_hours=True)}  ", self._time_format()
            )
        cursor.insertText(source.label, self._tag_format(source))

    def _clear_provisional(self) -> None:
        """Remove the current provisional text, if any."""
        if self._provisional_anchor < 0:
            return
        cursor = self.textCursor()
        cursor.setPosition(self._provisional_anchor)
        cursor.movePosition(
            QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor
        )
        cursor.removeSelectedText()
        self._provisional_anchor = -1

    def clear_provisional(self) -> None:
        self._clear_provisional()

    # -- navigation -------------------------------------------------------

    def scroll_to_end(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def is_scrolled_to_end(self) -> bool:
        bar = self.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    def rebuild(self, segments, provisional=None) -> None:
        """Redraw the whole transcript. Used on load and when toggling timestamps."""
        self.clear()
        self._last_source = None
        self._provisional_anchor = -1

        for segment in segments:
            self.append_confirmed(segment.source, segment.start, segment.text)
        for segment in provisional or []:
            self.set_provisional(segment.source, segment.text)

        if self._autoscroll:
            self.scroll_to_end()

    def plain_text(self) -> str:
        return self.toPlainText()
