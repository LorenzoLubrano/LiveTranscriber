"""Export formats.

A registry rather than a hard-coded list, so adding DOCX, PDF or Markdown later
(spec §13, §30) means writing one module and adding one entry — no caller
changes anywhere else.

Every exporter takes the same two things: a :class:`Transcript` and a
destination path. Format-specific options have defaults, so a new format can be
added without widening the shared call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.export import json_export, srt, txt, vtt
from app.export.base import ExportError
from app.sessions.transcript import Transcript

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportFormat:
    """One writable format."""

    key: str
    label: str
    extension: str
    writer: Callable[..., Path]
    description: str = ""

    @property
    def filter_string(self) -> str:
        """Qt file-dialog filter, e.g. ``Testo (*.txt)``."""
        return f"{self.label} (*{self.extension})"

    def write(self, transcript: Transcript, path: Path, title: str = "") -> Path:
        """Write ``transcript`` to ``path``.

        Title is passed only to the formats that have somewhere to put it, so
        each writer keeps a signature that suits its own format.
        """
        if self.key in ("txt", "vtt", "json"):
            return self.writer(transcript, path, title)
        return self.writer(transcript, path)


FORMATS: tuple[ExportFormat, ...] = (
    ExportFormat(
        key="txt",
        label="Testo",
        extension=".txt",
        writer=txt.export,
        description="Testo semplice con orari e sorgente.",
    ),
    ExportFormat(
        key="srt",
        label="Sottotitoli SubRip",
        extension=".srt",
        writer=srt.export,
        description="Sottotitoli per lettori video.",
    ),
    ExportFormat(
        key="vtt",
        label="Sottotitoli WebVTT",
        extension=".vtt",
        writer=vtt.export,
        description="Sottotitoli per il web.",
    ),
    ExportFormat(
        key="json",
        label="Dati strutturati",
        extension=".json",
        writer=json_export.export,
        description="Tutti i segmenti con tempi e sorgente.",
    ),
)

_BY_KEY = {fmt.key: fmt for fmt in FORMATS}

#: Written automatically when a recording stops.
DEFAULT_FORMATS = ("txt", "srt", "vtt", "json")


def get_format(key: str) -> ExportFormat:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise ExportError(
            f"Formato non supportato: {key}",
            f"unknown export format {key!r}; known: {sorted(_BY_KEY)}",
        ) from None


def list_formats() -> tuple[ExportFormat, ...]:
    return FORMATS


def export_all(
    transcript: Transcript,
    directory: Path,
    title: str = "",
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    stem: str = "transcript",
) -> dict[str, Path]:
    """Write every requested format into ``directory``.

    One format failing must not cost the others: each is attempted
    independently and failures are logged, because losing a transcript to a
    subtitle-formatting problem would be absurd.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for key in formats:
        try:
            fmt = get_format(key)
            path = fmt.write(transcript, directory / f"{stem}{fmt.extension}", title)
            written[key] = path
        except Exception:
            logger.exception("Export to %s failed", key)
    return written


__all__ = [
    "DEFAULT_FORMATS",
    "FORMATS",
    "ExportError",
    "ExportFormat",
    "export_all",
    "get_format",
    "json_export",
    "list_formats",
    "srt",
    "txt",
    "vtt",
]
