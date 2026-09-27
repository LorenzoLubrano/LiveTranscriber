"""A menu that says when its entry does not fit.

Qt draws a closed combo box's text clipped at the edge of the field, with no
sign that anything is missing: at the narrowest window the quality menu read
"Turbo · Qualità/velocità — da scaricare (1.6", which looks like a typo rather
than a cut. This one draws the same face and ends a long entry in "…".

Only the closed face changes. The list that opens shows every entry in full,
and the selection, keyboard and signals are QComboBox's own.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QWidget,
)


class ElidingComboBox(QComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    def elided_text(self, width: int) -> str:
        """The current entry as it fits in ``width`` pixels."""
        return self.fontMetrics().elidedText(
            self.currentText(), Qt.TextElideMode.ElideRight, max(0, width)
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)

        # The field is what is left once the style has taken its padding and
        # the arrow, so the ellipsis lands where the clipping used to.
        field = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        option.currentText = self.elided_text(field.width())
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)
