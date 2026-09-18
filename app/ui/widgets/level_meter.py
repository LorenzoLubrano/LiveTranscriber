"""Audio level meter.

The one saturated element in the interface, and the only way to answer the
question that matters most while recording: *is it actually hearing anything?*
A transcript that stops growing is ambiguous — the speaker may simply have
paused — but a dead meter is unambiguous.

Drawn as discrete segments rather than a continuous bar, following hardware
meters: segments are countable at a glance and from an angle, where a smooth
gradient is not. Colour follows the convention anyone who has used recording
equipment already knows: green, amber approaching full scale, red at clipping.

It also carries a **peak-hold marker** that falls back slowly, so a brief
transient stays visible long enough to be seen. Without it, a clip that lasts
30 ms is invisible at any sane repaint rate.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.ui.theme import Palette

#: Level at which segments turn amber, then red. Chosen so speech at a healthy
#: recording level sits in green with headroom.
_WARN_AT = 0.72
_CLIP_AT = 0.92

#: Peak marker fall rate, in level units per second.
_PEAK_FALL_PER_S = 0.55

#: How long a peak sits before it starts to fall.
_PEAK_HOLD_S = 0.9


class LevelMeter(QWidget):
    """Segmented level meter with peak hold."""

    def __init__(
        self,
        palette: Palette,
        segments: int = 14,
        vertical: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette
        self._segments = segments
        self._vertical = vertical

        self._level = 0.0
        self._peak = 0.0
        self._peak_set_at = 0.0
        self._active = False

        if vertical:
            self.setFixedWidth(10)
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        else:
            self.setFixedHeight(10)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

        # 20 Hz: fast enough to look live, slow enough to cost nothing. The
        # audio callback updates a float; this only reads it.
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

    # -- api --------------------------------------------------------------

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self.update()

    def set_level(self, level: float) -> None:
        """Set the current peak level, 0.0–1.0."""
        level = max(0.0, min(1.0, float(level)))
        self._level = level
        if level >= self._peak:
            self._peak = level
            self._peak_set_at = time.monotonic()

    def set_active(self, active: bool) -> None:
        """Start or stop animating. Inactive meters render as an empty track."""
        self._active = active
        if active:
            self._timer.start()
        else:
            self._timer.stop()
            self._level = 0.0
            self._peak = 0.0
            self.update()

    @property
    def level(self) -> float:
        return self._level

    # -- animation --------------------------------------------------------

    def _tick(self) -> None:
        now = time.monotonic()
        if self._peak > 0.0 and (now - self._peak_set_at) > _PEAK_HOLD_S:
            self._peak = max(self._level, self._peak - _PEAK_FALL_PER_S * 0.05)
        self.update()

    # -- painting ---------------------------------------------------------

    def _segment_colour(self, index: int) -> QColor:
        position = (index + 1) / self._segments
        if position >= _CLIP_AT:
            return QColor(self._palette.meter_high)
        if position >= _WARN_AT:
            return QColor(self._palette.meter_mid)
        return QColor(self._palette.meter_low)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)

        rect = self.rect()
        track = QColor(self._palette.meter_track)
        count = self._segments
        gap = 2.0

        if self._vertical:
            total = rect.height()
            size = max(1.0, (total - gap * (count - 1)) / count)
        else:
            total = rect.width()
            size = max(1.0, (total - gap * (count - 1)) / count)

        lit = int(self._level * count + 0.5)
        peak_index = int(self._peak * count + 0.5) - 1

        for i in range(count):
            if self._vertical:
                # Index 0 at the bottom.
                y = rect.bottom() - (i + 1) * size - i * gap
                segment = QRectF(rect.left(), y, rect.width(), size)
            else:
                x = rect.left() + i * (size + gap)
                segment = QRectF(x, rect.top(), size, rect.height())

            if self._active and i < lit:
                colour = self._segment_colour(i)
            elif self._active and i == peak_index and peak_index >= 0:
                colour = self._segment_colour(i)
                colour.setAlpha(150)
            else:
                colour = track

            painter.setBrush(colour)
            painter.drawRoundedRect(segment, 1.5, 1.5)

        painter.end()


class LabelledMeter(QWidget):
    """A meter with its source tag, as used in the recording strip."""

    def __init__(self, tag: str, colour: str, palette: Palette,
                 parent: QWidget | None = None) -> None:
        from PySide6.QtWidgets import QHBoxLayout, QLabel

        super().__init__(parent)
        self._palette = palette

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)

        self.label = QLabel(tag)
        self.label.setStyleSheet(
            f"color: {colour}; font-size: 9pt; font-weight: 600; letter-spacing: 0.3px;"
        )
        self.label.setFixedWidth(34)

        self.meter = LevelMeter(palette)

        layout.addWidget(self.label)
        layout.addWidget(self.meter, 1)

    def set_level(self, level: float) -> None:
        self.meter.set_level(level)

    def set_active(self, active: bool) -> None:
        self.meter.set_active(active)

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self.meter.set_palette(palette)


class RecordingDot(QWidget):
    """The recording indicator: a red dot that breathes while active.

    The only non-user-triggered motion in the interface. It earns its place
    because a static red dot is indistinguishable from a decorative one, and
    this is the single most important piece of state in the application: a
    recording that silently stopped must not look like one that is running.
    """

    def __init__(self, palette: Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._active = False
        self._phase = 0.0
        self.setFixedSize(13, 13)

        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self.update()

    def set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self._timer.start()
        else:
            self._timer.stop()
            self._phase = 0.0
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.06) % 1.0
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        import math

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)

        colour = QColor(self._palette.record)
        if not self._active:
            colour = QColor(self._palette.ink_faint)
            colour.setAlpha(110)
        else:
            # A slow, shallow pulse: alive, not alarming.
            pulse = 0.72 + 0.28 * (0.5 + 0.5 * math.sin(self._phase * 2 * math.pi))
            colour.setAlphaF(pulse)

        painter.setBrush(colour)
        size = 9.0
        offset = (self.width() - size) / 2
        painter.drawEllipse(QRectF(offset, offset, size, size))
        painter.end()
