"""Shared export machinery: time formats and segment grouping.

**Why grouping exists.** The streaming pipeline confirms text in whatever
fragments two passes happened to agree on, so a single spoken sentence often
lands as three or four segments:

    [PC] Consideriamo adesso la Hamiltoniana del sistema,
    [PC] dove il termine di interazione
    [PC] dipende dal tempo.

That is correct as a record of what was confirmed when, and wrong as a document.
Exports therefore regroup: consecutive segments from the same source are merged
while they belong together, and split where a reader needs a break. The rules
differ by format — a subtitle cue has hard limits a paragraph does not — so each
exporter asks for its own grouping rather than sharing one compromise.

Timestamps are written to the precision each format defines: SRT uses a comma
before milliseconds, WebVTT a full stop, and both are strict about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.sessions.transcript import Source, TranscriptSegment


def srt_timestamp(seconds: float) -> str:
    """``HH:MM:SS,mmm`` — the comma is part of the SRT specification."""
    seconds = max(0.0, seconds)
    milliseconds = int(round(seconds * 1000))
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def vtt_timestamp(seconds: float) -> str:
    """``HH:MM:SS.mmm`` — WebVTT requires a full stop, not a comma."""
    return srt_timestamp(seconds).replace(",", ".")


def clock_timestamp(seconds: float) -> str:
    """``HH:MM:SS`` for human-facing text."""
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class GroupingRules:
    """How aggressively to merge confirmed fragments back together."""

    #: Silence longer than this always starts a new group.
    max_gap: float = 2.0
    #: A group never spans longer than this.
    max_duration: float = 30.0
    #: A group never grows past this many characters.
    max_chars: int = 1_000
    #: End a group after a sentence-ending fragment once it is this long.
    sentence_break_after: float = 0.0

    @classmethod
    def for_paragraphs(cls) -> GroupingRules:
        """Readable prose: merge freely, break on real pauses."""
        return cls(max_gap=2.0, max_duration=60.0, max_chars=1_200)

    @classmethod
    def for_subtitles(cls) -> GroupingRules:
        """Subtitle cues: short enough to read before they disappear.

        Roughly two lines of about 42 characters, which is the convention
        broadcast subtitling settled on, and at most seven seconds on screen.
        """
        return cls(max_gap=1.0, max_duration=7.0, max_chars=84,
                   sentence_break_after=1.5)


@dataclass
class ExportSegment:
    """A merged group, ready to write."""

    source: Source
    start: float
    end: float
    text: str
    parts: list[TranscriptSegment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def label(self) -> str:
        return self.source.label


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "…", ":"))


def group_segments(
    segments: list[TranscriptSegment],
    rules: GroupingRules | None = None,
) -> list[ExportSegment]:
    """Merge confirmed fragments into readable units.

    A group is broken when the source changes, when the pause between fragments
    is long enough to be a real break, when the group has run too long, or when
    a sentence ended and the group is already substantial.
    """
    rules = rules or GroupingRules.for_paragraphs()
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    groups: list[ExportSegment] = []
    current: list[TranscriptSegment] = []

    def flush() -> None:
        if not current:
            return
        groups.append(
            ExportSegment(
                source=current[0].source,
                start=current[0].start,
                end=current[-1].end,
                text=" ".join(s.text.strip() for s in current if s.text.strip()),
                parts=list(current),
            )
        )
        current.clear()

    for segment in ordered:
        if not segment.text.strip():
            continue
        if not current:
            current.append(segment)
            continue

        previous = current[-1]
        pending_text = " ".join(s.text.strip() for s in current)

        breaks = (
            segment.source is not previous.source
            or segment.start - previous.end > rules.max_gap
            or segment.end - current[0].start > rules.max_duration
            or len(pending_text) + len(segment.text) + 1 > rules.max_chars
            or (
                rules.sentence_break_after > 0
                and _ends_sentence(previous.text)
                and previous.end - current[0].start >= rules.sentence_break_after
            )
        )
        if breaks:
            flush()
        current.append(segment)

    flush()
    return groups


def wrap_cue(text: str, width: int = 42, max_lines: int = 2) -> str:
    """Break a subtitle cue onto at most ``max_lines`` balanced lines.

    Splitting a cue in the middle is worse than a slightly long line, so the
    text is balanced across the available lines rather than greedily filled.
    """
    words = text.split()
    if not words:
        return ""
    if len(text) <= width:
        return text

    lines: list[str] = []
    line = ""
    # Aim for even lines. Filling greedily to the full width leaves a stub
    # second line - "Consideriamo adesso la miltoniana del / sistema, dove il" -
    # so when the text fits in the available lines, the target is the balanced
    # share rather than the maximum.
    balanced = -(-len(text) // max_lines)          # ceiling division
    target = min(width, balanced) if balanced <= width else width
    for word in words:
        candidate = f"{line} {word}".strip()
        if line and len(candidate) > target and len(lines) < max_lines - 1:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return "\n".join(lines[:max_lines]) if len(lines) <= max_lines else "\n".join(
        [*lines[: max_lines - 1], " ".join(lines[max_lines - 1 :])]
    )


def write_text(path: Path, content: str) -> Path:
    """Write UTF-8 text with the platform's newlines, creating parents.

    Written to a sibling temporary file and moved into place, so an export
    interrupted halfway never replaces a good file with a truncated one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(content, encoding="utf-8", newline="\r\n")
    temporary.replace(path)
    return path


class ExportError(RuntimeError):
    """Export failure, with a message safe to show to a user."""

    def __init__(self, user_message: str, technical: str = "") -> None:
        super().__init__(technical or user_message)
        self.user_message = user_message
        self.technical = technical or user_message
