"""Long-run robustness test (spec §17, §24).

The app has to survive a four-hour lecture. Everything verified so far ran for
seconds, which proves it works and says nothing about whether it *keeps*
working. This measures the things that only fail slowly:

* memory that grows with elapsed time instead of staying flat;
* Windows handles leaking a few per cycle;
* the Qt document getting slower to append to as it fills;
* autosave cost rising as the transcript grows — it rewrites the whole file;
* real-time headroom collapsing as the machine heats up.

Two modes, because they answer different questions:

``--accelerated`` (default, minutes)
    Feeds audio far faster than real time. Catches unbounded growth, leaks and
    quadratic costs quickly. Cannot catch anything thermal or time-based.

``--realtime --hours N``
    Runs at wall-clock speed. The only way to see thermal throttling, Windows
    power management and device dropouts.

Usage::

    python scripts/soak_test.py
    python scripts/soak_test.py --hours 0.5
    python scripts/soak_test.py --realtime --hours 2
    python scripts/soak_test.py --component transcript
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.resampler import TARGET_SAMPLE_RATE  # noqa: E402
from app.sessions.transcript import Source, Transcript  # noqa: E402
from app.utils.logging_setup import setup_logging  # noqa: E402

SAMPLE_RATE = TARGET_SAMPLE_RATE


# ------------------------------------------------------------------ sampling

@dataclass
class Sample:
    """One observation of the process, taken periodically."""

    elapsed: float           # simulated session seconds
    wall: float              # real seconds since start
    rss_mb: float
    handles: int
    threads: int
    segments: int
    document_chars: int = 0
    autosave_ms: float = 0.0
    append_ms: float = 0.0
    vram_mb: float = 0.0
    runs: int = 0

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in vars(self).items()}


@dataclass
class Series:
    samples: list[Sample] = field(default_factory=list)

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)

    def growth(self, attribute: str) -> float:
        """Change in an attribute from the first sample to the last."""
        if len(self.samples) < 2:
            return 0.0
        return getattr(self.samples[-1], attribute) - getattr(self.samples[0], attribute)

    def growth_after_warmup(self, attribute: str, skip: int = 2) -> float:
        """Growth ignoring the first samples, where caches are still filling."""
        if len(self.samples) < skip + 2:
            return 0.0
        return getattr(self.samples[-1], attribute) - getattr(self.samples[skip], attribute)

    def slope_per_hour(self, attribute: str, skip: int = 2) -> float:
        """Least-squares growth per simulated hour, after warm-up.

        A two-point slope is badly misleading here. Windows trims a process's
        working set when it wants memory back, so RSS can fall hundreds of
        megabytes late in a run while the steady state was perfectly flat —
        measured on this machine, 784 MB held for an hour then trimmed to 131.
        Fitting every sample describes the trend instead of two accidents.
        """
        usable = self.samples[skip:]
        if len(usable) < 3:
            return 0.0

        xs = [s.elapsed / 3600 for s in usable]
        ys = [float(getattr(s, attribute)) for s in usable]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator <= 0:
            return 0.0
        return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator

    def peak(self, attribute: str) -> float:
        return max((float(getattr(s, attribute)) for s in self.samples), default=0.0)

    def plateau(self, attribute: str, skip: int = 2) -> float:
        """Median of the post-warm-up samples: the level actually held."""
        import statistics

        usable = [float(getattr(s, attribute)) for s in self.samples[skip:]]
        return statistics.median(usable) if usable else 0.0


def process_metrics() -> tuple[float, int, int]:
    """RSS in MB, open handles, thread count."""
    import psutil

    process = psutil.Process()
    try:
        handles = process.num_handles()
    except (AttributeError, psutil.Error):
        handles = 0
    return process.memory_info().rss / 1_048_576, handles, process.num_threads()


def vram_mb() -> float:
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0


# ----------------------------------------------------------------- component

def soak_transcript(hours: float, series: Series) -> None:
    """The transcript store and autosave, without audio or inference.

    Isolated deliberately: autosave rewrites the whole transcript every few
    seconds, so its cost grows as the transcript does. That is the clearest
    candidate for a quadratic cost in the whole application.
    """
    import tempfile

    from app.export.json_export import SessionRecord
    from app.sessions.autosave import Autosave

    transcript = Transcript()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        autosave = Autosave(
            directory, transcript, SessionRecord(title="Soak", directory=str(directory))
        )

        # A lecture confirms a fragment roughly every 4 seconds.
        step = 4.0
        total_steps = int(hours * 3600 / step)
        sample_every = max(1, total_steps // 40)
        started = time.monotonic()

        for i in range(total_steps):
            elapsed = i * step
            transcript.add(
                Source.PC if i % 7 else Source.MIC,
                elapsed,
                elapsed + step * 0.9,
                "Consideriamo adesso il termine di interazione del sistema, "
                f"passaggio numero {i}.",
            )

            # Autosave runs every 5s of session time.
            autosave_ms = 0.0
            if i % max(1, int(5.0 / step)) == 0:
                t0 = time.perf_counter()
                autosave.save()
                autosave_ms = (time.perf_counter() - t0) * 1000

            if i % sample_every == 0:
                rss, handles, threads = process_metrics()
                series.add(Sample(
                    elapsed=elapsed,
                    wall=time.monotonic() - started,
                    rss_mb=rss, handles=handles, threads=threads,
                    segments=len(transcript),
                    autosave_ms=autosave_ms,
                ))

        autosave.stop()


def soak_document(hours: float, series: Series) -> None:
    """The Qt transcript widget.

    A four-hour lecture is ~31,000 words in one QTextDocument, appended to
    every couple of seconds with a scroll to the end each time. If Qt relays out
    the whole document on each append, the interface gets slower exactly as the
    lecture gets longer.
    """
    from PySide6.QtWidgets import QApplication

    from app.ui.theme import DARK
    from app.ui.widgets.transcript_view import TranscriptView

    app = QApplication.instance() or QApplication([])
    view = TranscriptView(DARK)
    view.resize(800, 600)

    step = 4.0
    total_steps = int(hours * 3600 / step)
    sample_every = max(1, total_steps // 40)
    started = time.monotonic()

    for i in range(total_steps):
        elapsed = i * step
        source = Source.PC if i % 7 else Source.MIC

        view.set_provisional(source, "testo provvisorio in arrivo")
        t0 = time.perf_counter()
        view.append_confirmed(
            source, elapsed,
            "Consideriamo adesso il termine di interazione del sistema, "
            f"passaggio numero {i}.",
        )
        append_ms = (time.perf_counter() - t0) * 1000

        if i % sample_every == 0:
            app.processEvents()
            rss, handles, threads = process_metrics()
            series.add(Sample(
                elapsed=elapsed,
                wall=time.monotonic() - started,
                rss_mb=rss, handles=handles, threads=threads,
                segments=i,
                document_chars=view.document().characterCount(),
                append_ms=append_ms,
            ))

    view.deleteLater()
    app.processEvents()


def _speech_loop() -> np.ndarray:
    """Real synthesised speech, to feed the pipeline with.

    An earlier version of this test used a tone plus noise, which sounded
    speech-shaped but is not speech: Silero correctly rejected every block, no
    inference ever ran, and the test reported flat memory because nothing was
    happening. A soak test that exercises nothing is worse than no soak test,
    because it looks green.
    """
    import subprocess
    import tempfile
    import wave

    from app.audio.resampler import resample_array

    text = (
        "Consideriamo adesso l'Hamiltoniana del sistema, dove il termine di "
        "interazione dipende dal tempo. Nel limite adiabatico la variazione e' "
        "sufficientemente lenta, quindi il sistema resta nello stato fondamentale."
    )
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / "speech.wav"
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$v = $s.GetInstalledVoices() | "
            "Where-Object { $_.VoiceInfo.Culture.Name -eq 'it-IT' } | Select-Object -First 1; "
            "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
            f"$s.SetOutputToWaveFile('{destination}'); "
            f"$s.Speak(@'\n{text}\n'@); $s.SetOutputToNull(); $s.Dispose()"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0 or not destination.exists():
            raise SystemExit(
                "Sintesi vocale non disponibile: il soak della pipeline "
                "richiede una voce italiana installata in Windows."
            )
        with wave.open(str(destination), "rb") as w:
            rate, channels = w.getframerate(), w.getnchannels()
            raw = w.readframes(w.getnframes())

    speech = resample_array(np.frombuffer(raw, dtype=np.int16), rate, SAMPLE_RATE, channels)
    # A pause between repetitions, so the commit path and the silence path are
    # both exercised on every loop.
    return np.concatenate([speech, np.zeros(int(1.5 * SAMPLE_RATE), dtype=np.float32)])


def soak_pipeline(hours: float, series: Series, realtime: bool, model: str) -> None:
    """The whole transcription pipeline, fed real synthesised speech.

    Uses real inference, so it exercises the model, the VAD, the hypothesis
    buffer and the buffer trimming together.
    """
    from app.transcription.engine import TranscriptionEngine, TranscriptionSettings
    from app.transcription.hardware import Accelerator
    from app.transcription.streaming import StreamingSettings, StreamingTranscriber

    engine = TranscriptionEngine(model, accelerator=Accelerator.AUTO)
    engine.load()
    transcriber = StreamingTranscriber(
        engine, TranscriptionSettings(language="it"), StreamingSettings()
    )

    loop = _speech_loop()
    print(f"      voce di prova: {loop.size / SAMPLE_RATE:.1f}s in ciclo", flush=True)

    block = int(0.25 * SAMPLE_RATE)
    total_blocks = int(hours * 3600 / 0.25)
    sample_every = max(1, total_blocks // 40)
    started = time.monotonic()
    position = 0

    try:
        for i in range(total_blocks):
            chunk = np.take(loop, range(position, position + block), mode="wrap")
            position = (position + block) % loop.size
            transcriber.feed(chunk.astype(np.float32))

            if transcriber.should_run():
                transcriber.process()

            if realtime:
                target = started + (i + 1) * 0.25
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

            if i % sample_every == 0:
                rss, handles, threads = process_metrics()
                series.add(Sample(
                    elapsed=i * 0.25,
                    wall=time.monotonic() - started,
                    rss_mb=rss, handles=handles, threads=threads,
                    segments=len(transcriber.confirmed_words),
                    vram_mb=vram_mb(),
                    runs=transcriber.runs,
                ))
    finally:
        transcriber.finish()
        engine.unload()
        gc.collect()


# -------------------------------------------------------------------- report

def report(series: Series, title: str, hours: float) -> int:
    print(f"\n{'-' * 72}")
    print(f"{title}  ({hours:.2f} h simulate, {len(series.samples)} campioni)")
    print("-" * 72)

    if len(series.samples) < 3:
        print("  campioni insufficienti")
        return 1

    first, last = series.samples[0], series.samples[-1]
    print(f"  {'':12s} {'iniziale':>10s} {'tipico':>10s} {'picco':>10s} "
          f"{'crescita/ora':>14s}")
    rows = [
        ("RAM (MB)", "rss_mb"),
        ("handle", "handles"),
        ("thread", "threads"),
    ]
    if any(s.document_chars for s in series.samples):
        rows.append(("caratteri", "document_chars"))
    if any(s.vram_mb for s in series.samples):
        rows.append(("VRAM (MB)", "vram_mb"))

    for label, attribute in rows:
        print(f"  {label:12s} {getattr(first, attribute):10.0f} "
              f"{series.plateau(attribute):10.0f} {series.peak(attribute):10.0f} "
              f"{series.slope_per_hour(attribute):+14.1f}")

    problems = 0

    # A soak test that exercised nothing must never report success.
    if any(s.runs for s in series.samples):
        total_runs = series.samples[-1].runs
        print(f"\n  inferenze eseguite: {total_runs}   "
              f"parole confermate: {last.segments}")
        if total_runs == 0 or last.segments == 0:
            print("  PROBLEMA: nessuna trascrizione prodotta; il test non ha "
                  "esercitato la pipeline")
            problems += 1

    # RAM: some growth is inherent (the transcript is kept), but it must be
    # modest and must not accelerate.
    rss_per_hour = series.slope_per_hour("rss_mb")
    if rss_per_hour > 150:
        print(f"\n  PROBLEMA: RAM cresce di {rss_per_hour:.0f} MB/ora")
        problems += 1

    handles_per_hour = series.slope_per_hour("handles")
    if handles_per_hour > 50:
        print(f"  PROBLEMA: handle crescono di {handles_per_hour:.0f}/ora (leak)")
        problems += 1

    threads_growth = series.growth_after_warmup("threads")
    if threads_growth > 2:
        print(f"  PROBLEMA: {threads_growth:.0f} thread in piu' a fine corsa")
        problems += 1

    # Costs that should stay flat as the session grows.
    for label, attribute in (("autosave", "autosave_ms"), ("append Qt", "append_ms")):
        values = [getattr(s, attribute) for s in series.samples if getattr(s, attribute) > 0]
        if len(values) < 6:
            continue
        early = sum(values[: len(values) // 4]) / max(1, len(values) // 4)
        late = sum(values[-len(values) // 4 :]) / max(1, len(values) // 4)
        ratio = late / early if early > 0 else 1.0
        print(f"\n  {label:10s} inizio {early:7.2f} ms   fine {late:7.2f} ms   "
              f"x{ratio:.1f}")
        if ratio > 4.0 and late > 20.0:
            print(f"  PROBLEMA: il costo di {label} cresce con la sessione")
            problems += 1

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="LiveTranscriber soak test")
    parser.add_argument("--hours", type=float, default=4.0,
                        help="ore di sessione da simulare (default 4)")
    parser.add_argument("--realtime", action="store_true",
                        help="a velocita' reale invece che accelerata")
    parser.add_argument("--component", default="all",
                        choices=["all", "transcript", "document", "pipeline"])
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--json", type=Path, help="salva i campioni grezzi")
    args = parser.parse_args()

    setup_logging(console=False)

    print("=" * 72)
    print(f"SOAK TEST  -  {args.hours} ore "
          f"({'tempo reale' if args.realtime else 'accelerato'})")
    print("=" * 72)

    results: dict[str, Series] = {}
    problems = 0
    started = time.monotonic()

    if args.component in ("all", "transcript"):
        print("\n[1] trascritto + autosave ...", flush=True)
        series = Series()
        soak_transcript(args.hours, series)
        results["transcript"] = series
        problems += report(series, "TRASCRITTO + AUTOSAVE", args.hours)
        gc.collect()

    if args.component in ("all", "document"):
        print("\n[2] documento Qt ...", flush=True)
        series = Series()
        soak_document(args.hours, series)
        results["document"] = series
        problems += report(series, "DOCUMENTO QT", args.hours)
        gc.collect()

    if args.component in ("all", "pipeline"):
        print(f"\n[3] pipeline completa (modello {args.model}) ...", flush=True)
        series = Series()
        soak_pipeline(args.hours, series, args.realtime, args.model)
        results["pipeline"] = series
        problems += report(series, "PIPELINE COMPLETA", args.hours)
        gc.collect()

    if args.json:
        args.json.write_text(
            json.dumps(
                {name: [s.to_dict() for s in series.samples]
                 for name, series in results.items()},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\ncampioni salvati in {args.json}")

    print(f"\n{'=' * 72}")
    print(f"durata reale del test: {(time.monotonic() - started) / 60:.1f} minuti")
    if problems:
        print(f"RISULTATO: {problems} problema/i rilevato/i")
    else:
        print("RISULTATO: nessuna crescita anomala rilevata")
    print("=" * 72)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
