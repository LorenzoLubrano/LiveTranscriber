"""Small marks drawn in the theme's own ink: stepper arrows and a tick.

Qt draws a spin box's arrows and a check box's tick itself only while those
parts are left unstyled — and unstyled, under this app's stylesheet, the arrows
came out as two specks clipped by the field's rounded corner. Style them and the
marks vanish completely unless the stylesheet supplies an image. So it does.

The images are drawn here rather than shipped, for two reasons: they follow the
palette, so a theme change can never leave dark marks on a dark field; and
there is nothing extra for the build to remember to bundle.

They are written to a directory made for this process alone and removed at
exit. A fixed, predictable path in a shared place would let anything able to
write there choose what the app displays.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen

#: Drawn at twice the size they are shown, so they stay sharp at 150-200 %.
PIXELS = 20

_directory: Path | None = None
_drawn: dict[tuple[str, str], str] = {}


def arrow_images(colour: str) -> tuple[str, str]:
    """Paths to the up and down chevrons in ``colour``, for a stylesheet url()."""
    up = _glyph("up", colour, [(5.0, 13.0), (10.0, 7.0), (15.0, 13.0)])
    down = _glyph("down", colour, [(5.0, 7.0), (10.0, 13.0), (15.0, 7.0)])
    return up, down


def tick_image(colour: str) -> str:
    """Path to a check mark in ``colour``, for a stylesheet url()."""
    return _glyph("tick", colour, [(4.5, 10.5), (8.5, 14.5), (15.5, 6.0)])


def _glyph(name: str, colour: str, points: list[tuple[float, float]]) -> str:
    key = (name, colour.lower())
    if key not in _drawn:
        path = _private_directory() / f"{name}-{colour.lower().lstrip('#')}.png"
        _draw(colour, points).save(str(path))
        # Forward slashes: a stylesheet url() reads a backslash as an escape.
        _drawn[key] = path.as_posix()
    return _drawn[key]


def _private_directory() -> Path:
    global _directory
    if _directory is None:
        _directory = Path(tempfile.mkdtemp(prefix="livetranscriber-ui-"))
        atexit.register(shutil.rmtree, _directory, True)
    return _directory


def _draw(colour: str, points: list[tuple[float, float]]) -> QImage:
    image = QImage(PIXELS, PIXELS, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)

    pen = QPen(QColor(colour))
    pen.setWidthF(2.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(pen)
    painter.drawPolyline([QPointF(x, y) for x, y in points])
    painter.end()
    return image
