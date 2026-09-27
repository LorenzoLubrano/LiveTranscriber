"""Visual design tokens and Qt stylesheet.

The interface is two materials, deliberately different:

**The instrument** — the control strip. Compact, dense, technical. This is the
part glanced at from across a desk during a two-hour lecture, so recording state
and audio level have to be readable at a distance and from the corner of an eye.

**The page** — the transcript. Generous line height, a reading serif, real
margins. Transcription tools default to a monospace log, which is hostile to
read for hours; the transcript here is a document, because that is what the user
is actually producing.

Colour is rationed. The **level meter is the only saturated element** in the
interface — green through amber to red, as on any real meter, because that
mapping is already known to anyone who has used recording equipment. The record
dot is red for the same reason. Everything else is ink, paper and grey, so the
two things that signal "working" are the only things competing for attention.

Typefaces are both native to Windows, so nothing is bundled and the app looks
like a Windows application rather than a web page in a frame:

* **Segoe UI** for controls and chrome.
* **Constantia** for the transcript — a serif designed for on-screen reading.

Monospace appears in exactly one place, the elapsed timer, because proportional
digits jitter as they tick and a shifting clock draws the eye for no reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ThemeMode(StrEnum):
    """Spec §9: light, dark, or follow Windows."""

    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"

    @property
    def label(self) -> str:
        return {
            ThemeMode.SYSTEM: "Sistema",
            ThemeMode.LIGHT: "Chiaro",
            ThemeMode.DARK: "Scuro",
        }[self]


@dataclass(frozen=True)
class Palette:
    """Every colour the interface uses."""

    # Surfaces
    window: str           # app background
    instrument: str       # control strip
    page: str             # transcript surface
    raised: str           # inputs, buttons
    raised_hover: str
    border: str
    border_strong: str

    # Text
    ink: str              # confirmed transcript, primary labels
    ink_soft: str         # secondary labels
    ink_faint: str        # provisional text, hints
    on_accent: str

    # The two signal colours
    record: str           # recording dot
    record_soft: str

    # Meter, low to clipping
    meter_low: str
    meter_mid: str
    meter_high: str
    meter_track: str

    # Source tags
    tag_pc: str
    tag_mic: str

    # States
    focus: str
    danger: str
    ok: str

    @property
    def is_dark(self) -> bool:
        return self.window.lower() in ("#12151c", "#0f1218")


#: Ink and paper. The dark surface is a blue-black rather than a neutral grey:
#: the instrument reads as a made object, and warm greys next to a warm paper
#: transcript would muddy the separation between the two materials.
DARK = Palette(
    window="#12151C",
    instrument="#171B24",
    page="#1B2029",
    raised="#222834",
    raised_hover="#2A3140",
    border="#2C3340",
    border_strong="#3C4557",
    ink="#E8EAED",
    ink_soft="#A8B0BD",
    ink_faint="#6F7887",
    on_accent="#0B0D12",
    record="#F05252",
    record_soft="#7A2020",
    meter_low="#3FB950",
    meter_mid="#D29922",
    meter_high="#F85149",
    meter_track="#2E3643",
    tag_pc="#7AA2F7",
    tag_mic="#C0A0E8",
    focus="#7AA2F7",
    danger="#F85149",
    ok="#3FB950",
)

LIGHT = Palette(
    window="#F2F1ED",
    instrument="#FFFFFF",
    page="#FBFAF7",
    raised="#FFFFFF",
    raised_hover="#F0EFEB",
    border="#DEDCD5",
    border_strong="#C3C0B6",
    ink="#1A1D23",
    ink_soft="#4E5560",
    ink_faint="#8A9199",
    on_accent="#FFFFFF",
    record="#D11A1A",
    record_soft="#F6C9C9",
    meter_low="#2E9E43",
    meter_mid="#B7791F",
    meter_high="#D11A1A",
    meter_track="#E6E4DD",
    tag_pc="#2A5DB0",
    tag_mic="#6B3FA0",
    focus="#2A5DB0",
    danger="#C0281F",
    ok="#2E9E43",
)


@dataclass(frozen=True)
class Typography:
    """Families and a type scale.

    Sizes are in points because Qt scales points with the Windows DPI setting,
    so 125% / 150% / 200% displays are handled by the platform rather than by
    arithmetic here (spec §9).
    """

    ui_family: str = "Segoe UI"
    ui_fallback: str = "Segoe UI Variable, Segoe UI, system-ui, sans-serif"
    reading_family: str = "Constantia"
    reading_fallback: str = "Constantia, Cambria, Georgia, serif"
    mono_family: str = "Consolas"
    mono_fallback: str = "Consolas, Cascadia Mono, monospace"

    micro: int = 8
    small: int = 9
    body: int = 10
    reading: int = 12
    large: int = 13
    timer: int = 22


TYPE = Typography()


def palette_for(mode: ThemeMode | str, system_is_dark: bool) -> Palette:
    """The palette for a mode, whatever form the mode arrives in.

    Qt stores a StrEnum in a widget's data slot as its plain string, so
    ``combo.currentData()`` hands back ``"dark"`` rather than ``ThemeMode.DARK``.
    Comparing with ``is`` made that value match neither branch and fall through
    to "follow Windows", which turned the window light every time the settings
    dialog was opened on a light-configured PC.
    """
    try:
        mode = ThemeMode(mode)
    except ValueError:
        # Not a mode at all: following the system is the safe reading.
        return DARK if system_is_dark else LIGHT

    if mode is ThemeMode.DARK:
        return DARK
    if mode is ThemeMode.LIGHT:
        return LIGHT
    return DARK if system_is_dark else LIGHT


def system_prefers_dark() -> bool:
    """Ask Windows whether apps should use a dark theme.

    Qt 6.5+ reports this through the style hints; older versions fall back to
    the registry value Windows itself uses. Defaults to light if neither works,
    which is the Windows default.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication

        hints = QGuiApplication.styleHints()
        scheme = getattr(hints, "colorScheme", None)
        if scheme is not None:
            return scheme() == Qt.ColorScheme.Dark
    except Exception:
        pass

    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        with key:
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return value == 0
    except Exception:
        return False


def stylesheet(p: Palette) -> str:
    """Qt stylesheet for the whole application."""
    from app.ui.glyphs import arrow_images, tick_image

    ui = TYPE.ui_fallback
    reading = TYPE.reading_fallback
    mono = TYPE.mono_fallback
    up_arrow, down_arrow = arrow_images(p.ink_soft)
    tick = tick_image(p.on_accent)

    return f"""
/* ---------- base ---------- */
QWidget {{
    background: {p.window};
    color: {p.ink};
    font-family: {ui};
    font-size: {TYPE.body}pt;
}}

QMainWindow, QDialog {{ background: {p.window}; }}

/* QWidget above paints every widget, labels included, which draws a band of
   window colour across whatever panel they sit on. Labels carry no surface of
   their own. */
QLabel {{ background: transparent; }}
QRadioButton, QCheckBox {{ background: transparent; }}

/* ---------- the instrument: control surfaces ---------- */
#InstrumentPanel {{
    background: {p.instrument};
    border: 1px solid {p.border};
    border-radius: 10px;
}}

/* A panel that is also a choice (the first-run cards) wears the focus colour
   while it is the chosen one. */
#InstrumentPanel[selected="true"] {{ border: 1px solid {p.focus}; }}

#SummaryBar {{
    background: {p.instrument};
    border: 1px solid {p.border};
    border-radius: 10px;
}}

#FieldLabel {{
    color: {p.ink_soft};
    font-size: {TYPE.small}pt;
}}

#SectionTitle {{
    color: {p.ink};
    font-size: {TYPE.large}pt;
    font-weight: 600;
}}

#Hint {{
    color: {p.ink_faint};
    font-size: {TYPE.small}pt;
}}

/* These two say where the work runs and where the audio stays. They are
   statements, not controls, and while they wore a border and a raised surface
   they were louder than the only real button next to them. */
#StatusChip {{
    color: {p.ink_soft};
    font-size: {TYPE.small}pt;
    padding: 3px 0;
    border: none;
    background: transparent;
}}

/* ---------- the door: the home page ---------- */
#Wordmark {{
    color: {p.ink_faint};
    font-size: {TYPE.small}pt;
    font-weight: 600;
}}

/* The greeting speaks in the transcript's own face: what this app makes is
   written text, so the page that welcomes you is set in the type it produces. */
#Greeting {{
    color: {p.ink};
    font-family: {reading};
    font-size: 19pt;
}}

#Standfirst {{
    color: {p.ink_soft};
    font-size: {TYPE.body}pt;
}}

QPushButton#ChoiceCard {{
    background: {p.instrument};
    border: 1px solid {p.border};
    border-left: 3px solid {p.border_strong};
    border-radius: 10px;
    padding: 0;
    text-align: left;
}}
QPushButton#ChoiceCard:hover {{ background: {p.raised_hover}; }}
QPushButton#ChoiceCard:pressed {{ background: {p.border}; }}
QPushButton#ChoiceCard:focus {{
    border: 2px solid {p.focus};
    border-left: 3px solid {p.focus};
}}
/* The one card that records is marked in the colour that means recording. */
QPushButton#ChoiceCard[primary="true"] {{ border-left: 3px solid {p.record}; }}

#ChoiceTitle {{
    color: {p.ink};
    font-size: {TYPE.large}pt;
    font-weight: 600;
}}

#ChoiceHint {{
    color: {p.ink_soft};
    font-size: {TYPE.small}pt;
}}

#Fineprint {{
    color: {p.ink_faint};
    font-size: {TYPE.micro}pt;
}}

/* ---------- inputs ---------- */
QComboBox {{
    background: {p.raised};
    border: 1px solid {p.border_strong};
    border-radius: 7px;
    padding: 7px 10px;
    min-height: 18px;
    color: {p.ink};
}}
QComboBox:hover {{ background: {p.raised_hover}; }}
QComboBox:focus {{ border: 2px solid {p.focus}; padding: 6px 9px; }}
QComboBox:disabled {{ color: {p.ink_faint}; background: {p.window}; }}
/* Styling ::drop-down suppresses the style's own arrow, and the CSS
   transparent-border triangle renders as a stray dash under Qt, so the arrow
   was left to Fusion: a dark block with its own edge at the end of every menu.
   With a drawn chevron available (app/ui/glyphs.py) the menus use the same mark
   as the numeric fields, and it turns over while the list is open. */
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border: none;
    background: transparent;
}}
QComboBox::down-arrow {{ image: url({down_arrow}); width: 10px; height: 10px; }}
QComboBox::down-arrow:on {{ image: url({up_arrow}); }}
QComboBox QAbstractItemView {{
    background: {p.raised};
    border: 1px solid {p.border_strong};
    border-radius: 7px;
    selection-background-color: {p.focus};
    selection-color: {p.on_accent};
    padding: 4px;
    outline: none;
}}

QLineEdit {{
    background: {p.raised};
    border: 1px solid {p.border_strong};
    border-radius: 7px;
    padding: 7px 10px;
    color: {p.ink};
    selection-background-color: {p.focus};
    selection-color: {p.on_accent};
}}
QLineEdit:focus {{ border: 2px solid {p.focus}; padding: 6px 9px; }}

QRadioButton, QCheckBox {{ color: {p.ink}; spacing: 9px; padding: 3px 0; }}
QRadioButton:disabled, QCheckBox:disabled {{ color: {p.ink_faint}; }}
/* Total size is width + 2*border, so both states are sized to land on 20px
   and the radius is half of that. Without shrinking the width when the border
   thickens, the control would jump as it is selected. */
QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 2px solid {p.border_strong};
    border-radius: 10px;
    background: {p.raised};
}}
QRadioButton::indicator:hover {{ border-color: {p.ink_faint}; }}
QRadioButton::indicator:checked {{
    width: 8px; height: 8px;
    border: 6px solid {p.focus};
    border-radius: 10px;
    background: {p.raised};
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 2px solid {p.border_strong};
    border-radius: 5px;
    background: {p.raised};
}}
QCheckBox::indicator:hover {{ border-color: {p.ink_faint}; }}
QCheckBox::indicator:checked {{
    background: {p.focus};
    border: 2px solid {p.focus};
    image: url({tick});
}}
/* The native dotted focus rectangle is switched off, so focus has to show on
   the indicator itself or a keyboard user cannot tell where they are. A
   selected radio already marks its group's focus: arrows move both at once. */
QRadioButton:focus, QCheckBox:focus {{ outline: none; }}
QRadioButton::indicator:focus, QCheckBox::indicator:focus {{ border-color: {p.focus}; }}
QCheckBox::indicator:checked:focus {{ border-color: {p.ink}; }}

/* ---------- buttons ---------- */
QPushButton {{
    background: {p.raised};
    border: 1px solid {p.border_strong};
    border-radius: 7px;
    padding: 8px 16px;
    color: {p.ink};
    font-weight: 500;
}}
QPushButton:hover {{ background: {p.raised_hover}; }}
QPushButton:pressed {{ background: {p.border}; }}
QPushButton:disabled {{ color: {p.ink_faint}; border-color: {p.border}; background: {p.window}; }}
QPushButton:focus {{ border: 2px solid {p.focus}; padding: 7px 15px; }}

QPushButton#PrimaryButton {{
    background: {p.record};
    border: 1px solid {p.record};
    color: #FFFFFF;
    font-size: {TYPE.large}pt;
    font-weight: 600;
    padding: 13px 26px;
}}
QPushButton#PrimaryButton:hover {{ background: {p.danger}; border-color: {p.danger}; }}
QPushButton#PrimaryButton:disabled {{
    background: {p.raised}; border-color: {p.border}; color: {p.ink_faint};
}}
QPushButton#PrimaryButton:focus {{ border: 2px solid {p.ink}; padding: 12px 25px; }}

QPushButton#QuietButton {{
    background: transparent;
    border: 1px solid transparent;
    color: {p.ink};
    padding: 6px 10px;
}}
QPushButton#QuietButton:hover {{ background: {p.raised_hover}; color: {p.ink}; }}
QPushButton#QuietButton:focus {{ border: 2px solid {p.focus}; padding: 5px 9px; }}
QPushButton#QuietButton:checked {{
    background: {p.raised};
    border: 1px solid {p.border_strong};
    color: {p.ink};
}}

/* The accepting button in a dialog. The big record-red button would shout
   here; this one only has to be the first thing the eye lands on. */
QPushButton#DialogPrimary {{
    background: {p.focus};
    border: 1px solid {p.focus};
    color: {p.on_accent};
    font-weight: 600;
}}
QPushButton#DialogPrimary:hover {{ background: {p.tag_pc}; border-color: {p.tag_pc}; }}
QPushButton#DialogPrimary:disabled {{
    background: {p.raised}; border-color: {p.border}; color: {p.ink_faint};
}}
QPushButton#DialogPrimary:focus {{ border: 2px solid {p.ink}; padding: 7px 15px; }}

QAbstractSpinBox {{
    background: {p.raised};
    border: 1px solid {p.border_strong};
    border-radius: 7px;
    padding: 7px 22px 7px 10px;
    min-height: 18px;
    color: {p.ink};
    selection-background-color: {p.focus};
    selection-color: {p.on_accent};
}}
QAbstractSpinBox:hover {{ background: {p.raised_hover}; }}
QAbstractSpinBox:focus {{ border: 2px solid {p.focus}; padding: 6px 21px 6px 9px; }}
QAbstractSpinBox:disabled {{ color: {p.ink_faint}; background: {p.window}; }}
/* Styling these buttons makes Qt stop drawing their arrows, so the arrows
   are supplied: see app/ui/glyphs.py. */
QAbstractSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border: none;
    border-left: 1px solid {p.border};
    border-top-right-radius: 7px;
    background: transparent;
}}
QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border: none;
    border-left: 1px solid {p.border};
    border-bottom-right-radius: 7px;
    background: transparent;
}}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {p.border};
}}
QAbstractSpinBox::up-arrow {{ image: url({up_arrow}); width: 10px; height: 10px; }}
QAbstractSpinBox::down-arrow {{ image: url({down_arrow}); width: 10px; height: 10px; }}

/* ---------- the page: transcript ---------- */
#TranscriptView {{
    background: {p.page};
    border: 1px solid {p.border};
    border-radius: 10px;
    padding: 18px 18px;
    font-family: {reading};
    font-size: {TYPE.reading}pt;
    color: {p.ink};
    selection-background-color: {p.focus};
    selection-color: {p.on_accent};
}}

#ElapsedTime {{
    font-family: {mono};
    font-size: {TYPE.timer}pt;
    font-weight: 600;
    color: {p.ink};
}}

#ElapsedTime[recording="false"] {{ color: {p.ink_faint}; }}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{
    background: transparent; width: 11px; margin: 4px 2px 4px 0;
}}
QScrollBar::handle:vertical {{
    background: {p.border_strong}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {p.ink_faint}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QScrollBar:horizontal {{ height: 0; }}

/* ---------- misc ---------- */
QToolTip {{
    background: {p.instrument};
    color: {p.ink};
    border: 1px solid {p.border_strong};
    border-radius: 6px;
    padding: 6px 9px;
}}

QProgressBar {{
    background: {p.meter_track};
    border: none;
    border-radius: 4px;
    height: 7px;
    text-align: center;
    color: {p.ink_soft};
}}
QProgressBar::chunk {{ background: {p.focus}; border-radius: 4px; }}

QGroupBox {{
    border: 1px solid {p.border};
    border-radius: 9px;
    margin-top: 14px;
    padding: 14px 12px 12px 12px;
    color: {p.ink_soft};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {p.ink};
    font-weight: 600;
}}

QTabWidget::pane {{
    border: none;
    border-top: 1px solid {p.border};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {p.ink_soft};
    padding: 9px 16px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {p.ink}; border-bottom: 2px solid {p.focus}; }}
QTabBar::tab:hover {{ color: {p.ink}; }}

QMenu {{
    background: {p.raised};
    border: 1px solid {p.border_strong};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{ padding: 7px 22px 7px 14px; border-radius: 5px; }}
QMenu::item:selected {{ background: {p.focus}; color: {p.on_accent}; }}

QSplitter::handle {{ background: transparent; }}
"""
