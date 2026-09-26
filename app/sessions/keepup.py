"""Deciding whether inference is keeping up with the audio.

On a GPU this never matters. On a four-core laptop with no NVIDIA card it is the
difference between a working app and one that looks broken, because streaming
costs far more than transcribing a file: LocalAgreement re-transcribes a growing
buffer, so every second of audio passes through Whisper four or five times. A
model that handles a recording faster than real time can still fall behind live.

What falling behind costs, in the order it happens:

1. **Latency grows.** Text stops arriving word by word and comes in bursts,
   once the working buffer hits its cap and force-commits.
2. **Audio is lost.** The capture ring buffer is drained only between inference
   passes (see :mod:`app.sessions.session`), so a pass longer than the ring's
   depth overwrites frames that were never written to disk.

The second one is silent, unrecoverable, and the reason this module exists.

The load is measured against *audio time*, not wall-clock: a pass that takes
0.9 s for every second of audio recorded leaves 10% of the machine for the
capture callbacks, the GUI and the rest of the system — which is not enough.
"""

from __future__ import annotations

import logging
from collections import deque
from enum import StrEnum

logger = logging.getLogger(__name__)

#: Load above which the machine is visibly struggling: latency climbs, and a
#: second source or a busy moment tips it into losing audio.
TIGHT_LOAD = 0.85

#: At or above this, inference is slower than the audio arriving. The backlog
#: grows without bound until the buffer cap force-commits it.
OVERRUN_LOAD = 1.0

#: Only recent history counts, so a mitigation that works is visible.
WINDOW_SECONDS = 60.0

#: Never judge the machine on less audio than this. A single slow pass happens
#: everywhere — the first one of a session always does, warming caches.
MIN_AUDIO_SECONDS = 15.0


class Level(StrEnum):
    """How the machine is coping."""

    OK = "ok"
    TIGHT = "tight"
    OVERRUN = "overrun"

    @property
    def rank(self) -> int:
        return {Level.OK: 0, Level.TIGHT: 1, Level.OVERRUN: 2}[self]


class KeepUpMonitor:
    """Rolling measure of inference cost per second of audio.

    Contains no Qt and no I/O: the session feeds it timings and asks what to do,
    which keeps it testable without a model, a sound card or a screen.
    """

    def __init__(
        self,
        window_seconds: float = WINDOW_SECONDS,
        min_audio_seconds: float = MIN_AUDIO_SECONDS,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_audio_seconds = min_audio_seconds
        #: (stream position when the pass ran, seconds of inference it took)
        self._passes: deque[tuple[float, float]] = deque()
        self._worst_reported = Level.OK

    # -- feeding ----------------------------------------------------------

    @property
    def last_position(self) -> float:
        return self._passes[-1][0] if self._passes else 0.0

    def record(self, inference_time: float, stream_position: float) -> Level:
        """Note one inference pass and return the current verdict."""
        self._passes.append((stream_position, inference_time))
        cutoff = stream_position - self.window_seconds
        while len(self._passes) > 2 and self._passes[0][0] < cutoff:
            self._passes.popleft()
        return self.level

    # -- reading ----------------------------------------------------------

    @property
    def span(self) -> float:
        """Seconds of audio the retained passes cover."""
        if len(self._passes) < 2:
            return 0.0
        return self._passes[-1][0] - self._passes[0][0]

    @property
    def load(self) -> float:
        """Inference seconds per second of audio, or 0.0 when unknown.

        The oldest retained pass contributes its position but not its cost: the
        audio it processed arrived before the window, so charging it here would
        overstate the load.
        """
        span = self.span
        if span <= 0:
            return 0.0
        spent = sum(cost for _, cost in list(self._passes)[1:])
        return spent / span

    @property
    def level(self) -> Level:
        if self.span < self.min_audio_seconds:
            return Level.OK
        load = self.load
        if load >= OVERRUN_LOAD:
            return Level.OVERRUN
        if load >= TIGHT_LOAD:
            return Level.TIGHT
        return Level.OK

    # -- reporting --------------------------------------------------------

    def take_warning(self) -> Level | None:
        """The level to warn about, once, or None.

        Only escalations are reported. Telling someone twice that their machine
        is slow is nagging, and telling them it recovered and then struggled
        again is noise — they already know, and the mitigation is in place.
        """
        level = self.level
        if level.rank <= self._worst_reported.rank:
            return None
        self._worst_reported = level
        logger.warning(
            "Inference is not keeping up: load %.2f over %.0fs of audio (%s)",
            self.load,
            self.span,
            level.value,
        )
        return level
