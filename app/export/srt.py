"""SubRip (.srt) export (spec §13).

Strict SRT: a 1-based index, a timing line with a comma before the
milliseconds, the text, and a blank line between cues.

Two rules that make the difference between a valid file and one a player
rejects: cues must not overlap, and a cue must not have zero duration. Both
happen in practice here, because PC and microphone are transcribed
independently and can genuinely speak over each other — so cues are ordered
and nudged apart rather than written out as-is.
"""

from __future__ import annotations

from pathlib import Path

from app.export.base import (
    GroupingRules,
    group_segments,
    srt_timestamp,
    wrap_cue,
    write_text,
)
from app.sessions.transcript import Transcript

EXTENSION = ".srt"

#: Shortest cue a player will reliably display.
MIN_CUE_DURATION = 0.4


def render(transcript: Transcript, include_source: bool = True) -> str:
    groups = group_segments(transcript.segments, GroupingRules.for_subtitles())
    if not groups:
        return ""

    blocks: list[str] = []
    previous_end = 0.0

    for index, group in enumerate(groups, start=1):
        start = max(group.start, previous_end)
        end = max(group.end, start + MIN_CUE_DURATION)
        previous_end = end

        text = wrap_cue(group.text)
        if include_source:
            text = f"[{group.label}] {text}"

        blocks.append(
            f"{index}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}\n"
        )

    return "\n".join(blocks)


def export(transcript: Transcript, path: Path, include_source: bool = True) -> Path:
    return write_text(path, render(transcript, include_source))
