"""Draw the application icon.

    python scripts/make_icon.py

Writes ``assets/icon.ico`` (the Windows icon, every size Explorer asks for) and
``assets/icon-512.png`` for the README.

The icon is drawn rather than hand-authored because it has to be right at seven
sizes. At 256 px a page of text under a waveform reads clearly; at 16 px the same
drawing is mud, so small sizes get a deliberately coarser version — no text
lines, fewer and thicker bars. That is the whole reason this is a script.

It says what the app does in one glyph: a page whose top line is a waveform.
Sound arriving, text left behind. The colours are the app's own — the blue-black
instrument surface, the warm paper of the transcript, the blue used for the PC
audio tag — so the icon and the window look like the same object.

Qt does the drawing (PySide6 is already a dependency); the .ico container is
assembled here, because Qt's ICO writer does not emit the PNG-compressed entries
that Windows needs for the large sizes.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath  # noqa: E402

# The app's palette (see app/ui/theme.py), so the icon belongs to the window.
TILE = QColor("#12151C")        # instrument, blue-black
TILE_EDGE = QColor("#39435A")
PAPER = QColor("#FBFAF7")       # the transcript surface
INK = QColor("#6F7887")
WAVE = QColor("#7AA2F7")        # the blue that marks PC audio in the transcript

#: Bar heights as a fraction of the tile, left to right, centred on an axis.
#: Uneven on purpose: a symmetrical waveform reads as a logo, an uneven one
#: reads as sound. Kept modest so nothing touches the tile edge.
BARS = (0.11, 0.20, 0.30, 0.24, 0.15, 0.09)
BARS_SMALL = (0.20, 0.34, 0.26, 0.13)

#: Sizes Windows asks for. 16 and 32 are the ones people actually see most.
SIZES = (16, 24, 32, 48, 64, 128, 256)

#: Below this, detail becomes noise.
COARSE_BELOW = 40


def draw_small(size: int) -> QImage:
    """Render the icon for a size where fractions of a pixel matter.

    Below about 40 px the geometry has to land on whole pixels or antialiasing
    turns three-pixel bars into grey suggestions of bars. So this one computes
    integer rectangles instead of scaling the large drawing down, and carries
    only what survives: the tile, the page, and a waveform of three or four
    thick strokes.
    """
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)

    margin = max(0, round(size * 0.02))
    tile = QPainterPath()
    tile.addRoundedRect(
        QRectF(margin, margin, size - 2 * margin, size - 2 * margin),
        size * 0.20,
        size * 0.20,
    )
    painter.fillPath(tile, TILE)

    inset_x = max(2, round(size * 0.16))
    inset_y = max(2, round(size * 0.19))
    page = QPainterPath()
    page.addRoundedRect(
        QRectF(inset_x, inset_y, size - 2 * inset_x, size - 2 * inset_y),
        max(1.0, size * 0.06),
        max(1.0, size * 0.06),
    )
    painter.fillPath(page, PAPER)

    # Whole-pixel bars, centred as a group inside the page.
    count = 3 if size <= 20 else 4
    width = max(2, round(size * 0.10))
    gap = max(1, round(size * 0.055))
    total = count * width + (count - 1) * gap
    x = round((size - total) / 2)
    axis = size / 2.0

    # Relative to the page, not the tile: measured against the tile, the tallest
    # bar ran the full height of the paper and the waveform lost its shape.
    page_height = size - 2 * inset_y
    heights = (0.40, 0.74, 0.54, 0.28) if count == 4 else (0.42, 0.76, 0.34)
    for index in range(count):
        bar_height = max(2, round(page_height * heights[index]))
        # Even heights around a centred axis keep both ends on a pixel edge.
        top = round(axis - bar_height / 2)
        painter.fillPath(
            _bar_path(x, top, width, bar_height), WAVE
        )
        x += width + gap

    painter.end()
    return image


def _bar_path(x: int, y: int, width: int, height: int) -> QPainterPath:
    path = QPainterPath()
    radius = width / 2.0
    path.addRoundedRect(QRectF(x, y, width, height), radius, radius)
    return path


def draw(size: int) -> QImage:
    """Render the icon at one size."""
    if size < COARSE_BELOW:
        return draw_small(size)
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    coarse = size < COARSE_BELOW
    s = float(size)

    def rect(x: float, y: float, w: float, h: float) -> QRectF:
        return QRectF(x * s, y * s, w * s, h * s)

    # -- the tile ---------------------------------------------------------
    # Full bleed: Windows draws no frame of its own, so the icon is the object.
    tile = QPainterPath()
    tile.addRoundedRect(rect(0.02, 0.02, 0.96, 0.96), 0.20 * s, 0.20 * s)
    painter.fillPath(tile, TILE)

    if not coarse:
        # A hairline edge, so the tile keeps its shape on a dark desktop.
        painter.setPen(TILE_EDGE)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(tile)
        painter.setPen(Qt.PenStyle.NoPen)

    # -- the page ---------------------------------------------------------
    # The transcript itself, and the icon's silhouette. Everything else sits
    # inside it, so nothing can break the tile's outline at any size.
    page = QPainterPath()
    page.addRoundedRect(rect(0.17, 0.20, 0.66, 0.62), 0.06 * s, 0.06 * s)
    painter.fillPath(page, PAPER)

    # -- the waveform, as the page's first line ---------------------------
    # The whole idea in one relationship: the top line of the page is sound,
    # the lines under it are what it became.
    bars = BARS_SMALL if coarse else BARS
    span = 0.46 if not coarse else 0.40
    left = 0.50 - span / 2
    slot = span / len(bars)
    width = slot * (0.50 if not coarse else 0.44)
    axis = 0.35 if not coarse else 0.40

    for index, height in enumerate(bars):
        x = left + slot * index + (slot - width) / 2
        bar = QPainterPath()
        radius = width * s / 2
        bar.addRoundedRect(
            rect(x, axis - height / 2, width, height), radius, radius
        )
        painter.fillPath(bar, WAVE)

    # -- the lines of text ------------------------------------------------
    # Ragged right, like a real paragraph. Dropped entirely when coarse: three
    # 1 px greys inside a 16 px icon are a smudge, not text.
    if not coarse:
        for y, width in ((0.56, 0.42), (0.645, 0.34), (0.73, 0.24)):
            line = QPainterPath()
            height = 0.045
            line.addRoundedRect(
                rect(0.27, y, width, height), height * s / 2, height * s / 2
            )
            painter.fillPath(line, INK)

    painter.end()
    return image


def png_bytes(image: QImage) -> bytes:
    # The QByteArray is bound to a name on purpose: passing a temporary to
    # QBuffer lets Python free it while Qt is still writing into it, which takes
    # the interpreter down with a stack-buffer-overrun rather than an exception.
    store = QByteArray()
    buffer = QBuffer(store)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(store)


def build_ico(images: dict[int, QImage], destination: Path) -> None:
    """Assemble a PNG-compressed .ico.

    The format is a 6-byte header, one 16-byte directory entry per image, then
    the payloads. A size of 256 is stored as 0, which is the convention that
    makes 256 px icons possible at all in a byte-wide field.
    """
    payloads = {size: png_bytes(image) for size, image in images.items()}
    offset = 6 + 16 * len(payloads)

    header = struct.pack("<HHH", 0, 1, len(payloads))
    entries = bytearray()
    for size in sorted(payloads):
        data = payloads[size]
        entries += struct.pack(
            "<BBBBHHII",
            0 if size >= 256 else size,   # width
            0 if size >= 256 else size,   # height
            0,                            # palette entries: none, it is RGBA
            0,                            # reserved
            1,                            # colour planes
            32,                           # bits per pixel
            len(data),
            offset,
        )
        offset += len(data)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        handle.write(header)
        handle.write(entries)
        for size in sorted(payloads):
            handle.write(payloads[size])


#: See main(): the QGuiApplication must outlive every QImage.
_keep_alive: list = []


def main() -> int:
    # A QGuiApplication is needed before any QImage painting.
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)

    images = {size: draw(size) for size in SIZES}

    ico = ROOT / "assets" / "icon.ico"
    build_ico(images, ico)

    # For the README and anywhere else a plain image is wanted.
    large = draw(512)
    png = ROOT / "assets" / "icon-512.png"
    large.save(str(png), "PNG")

    # A contact sheet, so the small sizes can be judged rather than assumed.
    sheet = QImage(560, 300, QImage.Format.Format_ARGB32_Premultiplied)
    sheet.fill(QColor("#8A8F98"))
    painter = QPainter(sheet)
    x = 16
    for size in SIZES:
        painter.drawImage(x, 16, images[size])
        painter.drawImage(x, 16 + 270 - size, images[size])
        x += size + 16
    painter.end()
    sheet.save(str(ROOT / "assets" / "icon-sheet.png"), "PNG")

    print(f"{ico}  ({ico.stat().st_size / 1024:.1f} KB, {len(SIZES)} sizes)")
    print(f"{png}")
    # The application object is deliberately left alive: tearing it down while
    # the QImages still exist crashes the interpreter on exit.
    _keep_alive.append(app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
