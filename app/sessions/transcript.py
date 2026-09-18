"""The transcript: the single source of truth for text in a session.

Holds every segment from every source on one timeline, so the GUI, the autosave
and all the exporters read the same data rather than each keeping their own
copy.

Segments carry the fields the spec requires (§12)::

    {"source": "pc", "start": 123.42, "end": 128.14,
     "text": "Consideriamo adesso il sistema.", "confirmed": true}

Two sources are interleaved by time rather than kept in separate lists, because
that is how a conversation reads:

    [PC]  Consideriamo adesso l'Hamiltoniana...
    [MIC] Quindi siamo nel limite adiabatico?
    [PC]  Esatto, purche' la variazione sia lenta...

No speaker-identification model is involved: the stream an utterance arrived on
already says who was speaking (spec §3).

Provisional text is stored alongside confirmed text but kept separate: it is
replaced wholesale on every update and never reaches an export.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Source(StrEnum):
    """Which stream an utterance came from."""

    PC = "pc"
    MIC = "mic"

    @property
    def label(self) -> str:
        """Tag shown in the transcript view and in exports."""
        return "PC" if self is Source.PC else "MIC"


def format_timestamp(seconds: float, always_hours: bool = False) -> str:
    """``HH:MM:SS`` for display; hours dropped when the session is short."""
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours or always_hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class TranscriptSegment:
    """One utterance on the session timeline."""

    source: Source
    start: float
    end: float
    text: str
    confirmed: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "confirmed": self.confirmed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptSegment:
        raw_source = str(data.get("source", "pc")).lower()
        source = Source.MIC if raw_source in ("mic", "microphone") else Source.PC
        return cls(
            source=source,
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")),
            confirmed=bool(data.get("confirmed", True)),
        )


@dataclass(frozen=True)
class SearchHit:
    """One match from :meth:`Transcript.search`."""

    index: int
    segment: TranscriptSegment
    start_offset: int
    end_offset: int


def _fold(text: str) -> str:
    """Accent- and case-insensitive form, for searching.

    Users type "perche" and expect to find "perché"; on an Italian transcript
    an accent-sensitive search is close to useless.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


class Transcript:
    """Thread-safe store of the session's text.

    Written by the transcription workers, read by the GUI, the autosave timer
    and the exporters, so every public method takes the lock.
    """

    def __init__(self) -> None:
        self._segments: list[TranscriptSegment] = []
        self._provisional: dict[Source, TranscriptSegment] = {}
        self._lock = threading.RLock()
        self._revision = 0

    # -- reading ----------------------------------------------------------

    @property
    def revision(self) -> int:
        """Increments on every change; lets autosave skip unchanged state."""
        with self._lock:
            return self._revision

    @property
    def segments(self) -> list[TranscriptSegment]:
        """Confirmed segments, in timeline order."""
        with self._lock:
            return list(self._segments)

    @property
    def provisional(self) -> list[TranscriptSegment]:
        """Current unconfirmed text, at most one per source."""
        with self._lock:
            return [self._provisional[s] for s in sorted(self._provisional, key=str)]

    def all_segments(self, include_provisional: bool = False) -> list[TranscriptSegment]:
        with self._lock:
            out = list(self._segments)
            if include_provisional:
                out.extend(self._provisional.values())
                out.sort(key=lambda s: (s.start, s.end))
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._segments)

    def __bool__(self) -> bool:
        return len(self) > 0

    @property
    def duration(self) -> float:
        """Timeline position of the last confirmed text."""
        with self._lock:
            return max((s.end for s in self._segments), default=0.0)

    @property
    def word_count(self) -> int:
        with self._lock:
            return sum(len(s.text.split()) for s in self._segments)

    def text(self, include_source: bool = True, include_timestamps: bool = False) -> str:
        """Plain text of the whole transcript."""
        lines: list[str] = []
        for seg in self.segments:
            prefix = ""
            if include_timestamps:
                prefix += f"[{format_timestamp(seg.start, always_hours=True)}] "
            if include_source:
                prefix += f"[{seg.source.label}] "
            lines.append(f"{prefix}{seg.text}" if prefix else seg.text)
        return "\n".join(lines)

    def sources_used(self) -> set[Source]:
        with self._lock:
            return {s.source for s in self._segments}

    # -- writing ----------------------------------------------------------

    def add(
        self,
        source: Source,
        start: float,
        end: float,
        text: str,
        confirmed: bool = True,
    ) -> TranscriptSegment | None:
        """Append confirmed text. Returns the stored segment, or None if empty."""
        cleaned = text.strip()
        if not cleaned:
            return None

        segment = TranscriptSegment(source, start, end, cleaned, confirmed)
        with self._lock:
            self._insert_locked(segment)
            self._revision += 1
        return segment

    def add_segment(self, segment: TranscriptSegment) -> None:
        with self._lock:
            self._insert_locked(segment)
            self._revision += 1

    def _insert_locked(self, segment: TranscriptSegment) -> None:
        """Insert keeping the list ordered by start time.

        Appending is the common case — text arrives in order — so the search
        starts from the end rather than bisecting from the front.
        """
        segments = self._segments
        if not segments or segment.start >= segments[-1].start:
            segments.append(segment)
            return
        index = len(segments) - 1
        while index > 0 and segments[index - 1].start > segment.start:
            index -= 1
        segments.insert(index, segment)

    def set_provisional(self, source: Source, start: float, end: float, text: str) -> None:
        """Replace the unconfirmed text for one source.

        Provisional text is whole-replaced rather than appended, because the
        next inference run may revise all of it.
        """
        cleaned = text.strip()
        with self._lock:
            if cleaned:
                self._provisional[source] = TranscriptSegment(
                    source, start, end, cleaned, confirmed=False
                )
            else:
                self._provisional.pop(source, None)
            self._revision += 1

    def clear_provisional(self, source: Source | None = None) -> None:
        with self._lock:
            if source is None:
                self._provisional.clear()
            else:
                self._provisional.pop(source, None)
            self._revision += 1

    def clear(self) -> None:
        with self._lock:
            self._segments.clear()
            self._provisional.clear()
            self._revision += 1

    # -- search -----------------------------------------------------------

    def search(self, query: str, case_sensitive: bool = False) -> list[SearchHit]:
        """Find ``query`` in confirmed text (spec §12).

        Matching ignores accents and case by default.
        """
        if not query.strip():
            return []

        hits: list[SearchHit] = []
        needle = query if case_sensitive else _fold(query)

        for index, segment in enumerate(self.segments):
            haystack = segment.text if case_sensitive else _fold(segment.text)
            start = haystack.find(needle)
            while start != -1:
                hits.append(SearchHit(index, segment, start, start + len(needle)))
                start = haystack.find(needle, start + 1)
        return hits

    def segment_at(self, seconds: float) -> TranscriptSegment | None:
        """Segment covering a point on the timeline, if any."""
        for segment in self.segments:
            if segment.start <= seconds <= segment.end:
                return segment
        return None

    # -- serialisation ----------------------------------------------------

    def to_dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.segments]

    def load(self, data: list[dict[str, Any]]) -> None:
        """Replace the contents, e.g. when recovering a session."""
        restored = [TranscriptSegment.from_dict(d) for d in data]
        restored.sort(key=lambda s: (s.start, s.end))
        with self._lock:
            self._segments = restored
            self._provisional.clear()
            self._revision += 1


@dataclass
class TranscriptStats:
    """Summary shown in the UI and written to ``session.json``."""

    segments: int = 0
    words: int = 0
    duration: float = 0.0
    by_source: dict[str, int] = field(default_factory=dict)

    @classmethod
    def of(cls, transcript: Transcript) -> TranscriptStats:
        segments = transcript.segments
        by_source: dict[str, int] = {}
        for seg in segments:
            by_source[str(seg.source)] = by_source.get(str(seg.source), 0) + 1
        return cls(
            segments=len(segments),
            words=sum(len(s.text.split()) for s in segments),
            duration=max((s.end for s in segments), default=0.0),
            by_source=by_source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": self.segments,
            "words": self.words,
            "duration": round(self.duration, 3),
            "by_source": self.by_source,
        }


_SENTENCE_END = re.compile(r"[.!?…]$")


def looks_complete(text: str) -> bool:
    """True if ``text`` ends on a sentence boundary.

    Used to decide whether a new utterance should start a fresh paragraph or
    continue the previous one in the display.
    """
    return bool(_SENTENCE_END.search(text.strip()))
