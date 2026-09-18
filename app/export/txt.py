"""Plain text export (spec §13).

The format the spec fixes::

    [00:12:31] [PC]
    Consideriamo adesso l'Hamiltoniana...

    [00:12:38] [MIC]
    Quindi siamo nel limite adiabatico?

The header and the text sit on separate lines so the prose keeps a clean left
margin: a reader scanning the transcript should see sentences, not a column of
timestamps interrupting them.
"""

from __future__ import annotations

from pathlib import Path

from app.export.base import (
    GroupingRules,
    clock_timestamp,
    group_segments,
    write_text,
)
from app.sessions.transcript import Transcript

EXTENSION = ".txt"


def render(
    transcript: Transcript,
    title: str = "",
    include_timestamps: bool = True,
    include_source: bool = True,
) -> str:
    """Render the transcript as plain text."""
    groups = group_segments(transcript.segments, GroupingRules.for_paragraphs())

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

    for group in groups:
        header_parts = []
        if include_timestamps:
            header_parts.append(f"[{clock_timestamp(group.start)}]")
        if include_source:
            header_parts.append(f"[{group.label}]")

        if header_parts:
            lines.append(" ".join(header_parts))
        lines.append(group.text)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export(
    transcript: Transcript,
    path: Path,
    title: str = "",
    include_timestamps: bool = True,
    include_source: bool = True,
) -> Path:
    return write_text(
        path, render(transcript, title, include_timestamps, include_source)
    )
