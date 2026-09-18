"""WebVTT (.vtt) export (spec §13).

Differs from SRT in three ways that matter: the file must begin with the
literal ``WEBVTT`` header, timestamps use a full stop before the milliseconds,
and the cue index is optional.

Speakers are marked with WebVTT's own voice span, ``<v PC>``, rather than a
bracketed prefix. Players that understand it can style PC and microphone
differently; players that do not simply show the name.
"""

from __future__ import annotations

from pathlib import Path

from app.export.base import (
    GroupingRules,
    group_segments,
    vtt_timestamp,
    wrap_cue,
    write_text,
)
from app.sessions.transcript import Transcript

EXTENSION = ".vtt"

MIN_CUE_DURATION = 0.4


def render(transcript: Transcript, title: str = "", include_source: bool = True) -> str:
    groups = group_segments(transcript.segments, GroupingRules.for_subtitles())

    lines = ["WEBVTT"]
    if title:
        # A NOTE block is the only sanctioned place for free text in a header.
        lines.append("")
        lines.append(f"NOTE {title}")
    lines.append("")

    previous_end = 0.0
    for index, group in enumerate(groups, start=1):
        start = max(group.start, previous_end)
        end = max(group.end, start + MIN_CUE_DURATION)
        previous_end = end

        text = wrap_cue(group.text)
        if include_source:
            text = f"<v {group.label}>{text}"

        lines.append(str(index))
        lines.append(f"{vtt_timestamp(start)} --> {vtt_timestamp(end)}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def export(
    transcript: Transcript, path: Path, title: str = "", include_source: bool = True
) -> Path:
    return write_text(path, render(transcript, title, include_source))
