"""Keeping wrapped text whole when a window is made small.

Qt works out how small a window may get from its layout's minimum size, and
that minimum is computed without asking wrapped text how tall it becomes at the
width it is actually given. So a label that needs four lines at the narrowest
width is budgeted one, and when the window reaches its minimum the layout takes
the difference from whatever sits beside it: menus lose their bottom half,
radio buttons turn into clipped ovals, a description loses its second line.

The fix is to ask. Whenever the widget changes width, its minimum height is set
to what its layout needs *at that width*, and Qt's ordinary machinery does the
rest — the window simply stops shrinking where the text would start to suffer.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QWidget


class WrappedTextGuard(QObject):
    """Holds ``widget``'s minimum height at what its wrapped text needs."""

    def __init__(self, widget: QWidget) -> None:
        super().__init__(widget)
        self._widget = widget
        widget.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        if event.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest):
            self.refresh()
        return False

    def refresh(self) -> None:
        """Recompute now; call after changing text that may wrap differently."""
        layout = self._widget.layout()
        if layout is None or not layout.hasHeightForWidth():
            return
        needed = layout.totalHeightForWidth(self._widget.width())
        # Never below what the layout needs anyway, and no call when nothing
        # changed: a minimum set from inside a resize must settle, not loop.
        needed = max(needed, layout.totalMinimumSize().height())
        if needed > 0 and needed != self._widget.minimumHeight():
            self._widget.setMinimumHeight(needed)


def keep_wrapped_text_whole(widget: QWidget) -> WrappedTextGuard:
    return WrappedTextGuard(widget)
