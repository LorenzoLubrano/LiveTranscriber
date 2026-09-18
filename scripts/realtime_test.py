"""Real-time pipeline diagnostic (milestone 4).

Feeds audio through the streaming transcriber **at wall-clock speed**, exactly
as a live recording would, and measures what the user actually experiences:

* how long after a word is spoken its text appears;
* whether confirmed text ever duplicates or loses words;
* whether silence produces hallucinated text;
* how much headroom is left over real time.

Usage::

    python scripts/realtime_test.py
    python scripts/realtime_test.py --model medium --device gpu
    python scripts/realtime_test.py --wav lezione.wav
    python scripts/realtime_test.py --silence      # hallucination check only
    python scripts/realtime_test.py --compare      # streaming vs one-shot
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.resampler import TARGET_SAMPLE_RATE  # noqa: E402
from app.transcription import models  # noqa: E402
from app.transcription.engine import (  # noqa: E402
    TranscriptionEngine,
    TranscriptionSettings,
)
from app.transcription.hardware import Accelerator  # noqa: E402
from app.transcription.streaming import (  # noqa: E402
    StreamingSettings,
    StreamingTranscriber,
)
from app.utils.logging_setup import setup_logging  # noqa: E402
from scripts.whisper_test import (  # noqa: E402
    hr,
    load_wav,
    synthesize,
    word_accuracy,
)

#: Distinct sentences, so a repeated phrase in the output is a real bug rather
#: than the sample repeating itself. Lecture register, with terms Whisper finds
#: hard and numbers that expose misheard digits.
SAMPLE_SENTENCES = [
    "Consideriamo adesso l'Hamiltoniana del sistema, "
    "dove il termine di interazione dipende dal tempo.",
    "Nel limite adiabatico la variazione e' sufficientemente lenta, "
    "quindi il sistema resta nello stato fondamentale.",
    "Passiamo al secondo esempio: una particella in una buca di potenziale "
    "di larghezza pari a tre nanometri.",
    "La correzione al primo ordine si annulla per simmetria, "
    "mentre il secondo ordine contribuisce con un termine negativo.",
]

#: How much audio the capture layer hands over at a time, in seconds.
FEED_INTERVAL = 0.25


def build_sample(tmp: Path, sentences: int = 3) -> tuple[np.ndarray, str]:
    """Synthesise distinct sentences separated by pauses.

    The pauses give the streaming loop real commit points to find, and the
    sentences differ from one another so that any repeated phrase in the output
    is a genuine duplication bug.
    """
    parts: list[np.ndarray] = []
    reference_parts: list[str] = []
    gap = np.zeros(int(1.2 * TARGET_SAMPLE_RATE), dtype=np.float32)

    for i in range(min(sentences, len(SAMPLE_SENTENCES))):
        text = SAMPLE_SENTENCES[i]
        destination = tmp / f"part{i}.wav"
        if synthesize(text, True, destination) is None:
            raise SystemExit("Sintesi vocale non disponibile; usa --wav")
        parts.append(load_wav(destination))
        reference_parts.append(text)
        parts.append(gap)

    return np.concatenate(parts), " ".join(reference_parts)


def run_realtime(
    transcriber: StreamingTranscriber,
    audio: np.ndarray,
    pace: bool = True,
    verbose: bool = True,
) -> dict:
    """Feed ``audio`` at real-time speed and record what happened.

    Feeding and inference run on **separate threads**, because that is the real
    architecture: audio arrives on a PortAudio callback and goes straight into a
    ring buffer, while a worker thread transcribes independently. Doing both on
    one thread would make a slow model look like it was dropping audio, when in
    truth a slow model only adds latency — the capture side never stalls.

    So ``behind_realtime`` here measures genuine capture lag (expected: zero),
    and the latencies measure the delay a user would actually see.
    """
    chunk = int(FEED_INTERVAL * TARGET_SAMPLE_RATE)
    started = time.monotonic()
    latencies: list[float] = []
    confirmed_events: list[tuple[float, str]] = []
    behind_realtime = 0.0
    feeding_done = threading.Event()
    lock = threading.Lock()

    def worker() -> None:
        """Transcribe whatever has accumulated, as fast as it can."""
        while not feeding_done.is_set():
            if not transcriber.should_run():
                time.sleep(0.02)
                continue
            update = transcriber.process()
            if update is None or not update.confirmed:
                continue
            now = time.monotonic() - started
            spoken_at = update.confirmed[-1].end
            with lock:
                latencies.append(now - spoken_at)
                confirmed_events.append((now, update.confirmed_text))
            if verbose:
                print(f"  [{now:6.2f}s] (+{now - spoken_at:4.2f}s) {update.confirmed_text}")

    thread = threading.Thread(target=worker, name="transcribe", daemon=True)
    thread.start()

    try:
        for offset in range(0, audio.size, chunk):
            block = audio[offset : offset + chunk]
            audio_time = (offset + block.size) / TARGET_SAMPLE_RATE

            if pace:
                # Wait until this audio would really have been captured.
                delay = (started + audio_time) - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    behind_realtime = max(behind_realtime, -delay)

            transcriber.feed(block)
    finally:
        feeding_done.set()
        thread.join(timeout=120)

    final = transcriber.finish()
    if final.confirmed:
        now = time.monotonic() - started
        confirmed_events.append((now, final.confirmed_text))
        if verbose:
            print(f"  [{now:6.2f}s] (finale)   {final.confirmed_text}")

    return {
        "wall_time": time.monotonic() - started,
        "audio_duration": audio.size / TARGET_SAMPLE_RATE,
        "latencies": latencies,
        "events": confirmed_events,
        "text": transcriber.confirmed_text,
        "runs": transcriber.runs,
        "skipped": transcriber.skipped_silent_runs,
        "inference_total": transcriber.total_inference_time,
        "behind_realtime": behind_realtime,
    }


def find_duplicates(text: str, window: int = 6) -> list[str]:
    """Detect repeated word runs — the classic streaming failure."""
    tokens = text.lower().split()
    found: list[str] = []
    for size in range(window, 2, -1):
        for i in range(len(tokens) - size * 2 + 1):
            first = tokens[i : i + size]
            if first == tokens[i + size : i + size * 2]:
                found.append(" ".join(first))
    return found


def report(result: dict, reference: str) -> int:
    problems = 0
    latencies = result["latencies"]

    hr("LATENZA")
    if latencies:
        print(f"  media        : {sum(latencies) / len(latencies):5.2f}s")
        print(f"  minima       : {min(latencies):5.2f}s")
        print(f"  massima      : {max(latencies):5.2f}s")
    else:
        print("  nessun testo confermato")
        problems += 1

    hr("THROUGHPUT")
    audio_duration = result["audio_duration"]
    inference = result["inference_total"]
    print(f"  audio        : {audio_duration:6.2f}s")
    print(f"  wall clock   : {result['wall_time']:6.2f}s")
    print(f"  inferenza    : {inference:6.2f}s  "
          f"({100 * inference / audio_duration:.0f}% del tempo audio)")
    print(f"  esecuzioni   : {result['runs']}  (saltate per silenzio: {result['skipped']})")
    headroom = audio_duration / inference if inference else float("inf")
    print(f"  margine      : {headroom:5.1f}x il tempo reale")
    if result["behind_realtime"] > 0.5:
        print(f"  ATTENZIONE   : la cattura e' rimasta indietro di "
              f"{result['behind_realtime']:.2f}s")
        problems += 1
    else:
        print("  cattura      : nessun ritardo")
    if latencies and max(latencies) > 8.0:
        print(f"  ATTENZIONE   : latenza massima {max(latencies):.1f}s, troppo alta")
        problems += 1

    hr("QUALITA'")
    text = result["text"]
    duplicates = find_duplicates(text)
    if duplicates:
        print(f"  DUPLICAZIONI rilevate: {duplicates[:3]}")
        problems += 1
    else:
        print("  duplicazioni : nessuna")

    if reference:
        accuracy = word_accuracy(reference, text)
        print(f"  parole       : {accuracy * 100:.0f}% del riferimento")
        if accuracy < 0.6:
            print("  ATTENZIONE   : accuratezza bassa")
            problems += 1

    hr("TRASCRIZIONE")
    print(f"  {text}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="LiveTranscriber realtime diagnostic")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--language", default="it")
    parser.add_argument("--chunk", type=float, default=2.0, help="intervallo di inferenza (s)")
    parser.add_argument("--sentences", type=int, default=4)
    parser.add_argument("--no-pace", action="store_true", help="il piu' veloce possibile")
    parser.add_argument("--silence", action="store_true", help="verifica solo il silenzio")
    parser.add_argument("--compare", action="store_true", help="confronta con la trascrizione unica")
    args = parser.parse_args()

    setup_logging(console=False)

    if not models.is_available(args.model):
        spec = models.get_spec(args.model)
        print(f"Il modello {spec.display_name} non e' scaricato ({spec.size_label}).")
        print(f"Scaricalo con: python scripts/whisper_test.py --download {args.model}")
        return 1

    import tempfile

    reference = ""
    with tempfile.TemporaryDirectory() as tmp:
        if args.silence:
            audio = np.zeros(int(12 * TARGET_SAMPLE_RATE), dtype=np.float32)
        elif args.wav:
            audio = load_wav(args.wav)
        else:
            audio, reference = build_sample(Path(tmp), args.sentences)

    hr("CONFIGURAZIONE")
    print(f"  modello      : {args.model}")
    print(f"  audio        : {audio.size / TARGET_SAMPLE_RATE:.2f}s")
    print(f"  chunk        : {args.chunk}s")
    print(f"  ritmo        : {'reale' if not args.no_pace else 'massimo'}")

    engine = TranscriptionEngine(args.model, accelerator=Accelerator(args.device))
    try:
        choice = engine.load()
    except Exception as exc:
        print(f"\nERRORE: {getattr(exc, 'user_message', exc)}")
        return 1
    print(f"  esecuzione   : {choice}")
    if engine.load_warning:
        print(f"  ! {engine.load_warning}")

    settings = TranscriptionSettings(language=args.language)
    streaming = StreamingSettings(chunk_duration=args.chunk)
    transcriber = StreamingTranscriber(engine, settings, streaming)

    try:
        if args.silence:
            hr("VERIFICA SILENZIO")
            print("  12s di silenzio digitale; non deve produrre testo.\n")
            result = run_realtime(transcriber, audio, pace=not args.no_pace)
            text = result["text"].strip()
            hr("RISULTATO")
            if text:
                print(f"  FALLITO: allucinazione sul silenzio -> {text!r}")
                return 1
            print(f"  OK: nessun testo prodotto "
                  f"({result['skipped']} esecuzioni saltate, "
                  f"{result['runs']} inferenze)")
            return 0

        hr("TRASCRIZIONE IN TEMPO REALE")
        result = run_realtime(transcriber, audio, pace=not args.no_pace)
        problems = report(result, reference)

        if args.compare:
            hr("CONFRONTO CON TRASCRIZIONE UNICA")
            one_shot = engine.transcribe(audio, settings)
            print(f"  streaming : {result['text']}")
            print(f"  one-shot  : {one_shot.text}")
            similarity = word_accuracy(one_shot.text, result["text"])
            print(f"  parole in comune: {similarity * 100:.0f}%")
            if similarity < 0.8:
                print("  ATTENZIONE: lo streaming perde testo rispetto alla trascrizione unica")
                problems += 1

        hr()
        if problems:
            print(f"RISULTATO: {problems} problema/i rilevato/i")
            return 1
        print("RISULTATO: pipeline in tempo reale funzionante")
        return 0
    finally:
        engine.unload()


if __name__ == "__main__":
    raise SystemExit(main())
