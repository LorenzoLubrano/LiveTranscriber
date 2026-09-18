"""Transcription engine diagnostic and benchmark (spec §22, §24).

Verifies the Whisper engine independently of audio capture, and measures the
real-time factor that decides which models this machine can use live.

Usage::

    python scripts/whisper_test.py                      # default model, self-made sample
    python scripts/whisper_test.py --model small
    python scripts/whisper_test.py --device cpu
    python scripts/whisper_test.py --wav path/to/audio.wav
    python scripts/whisper_test.py --benchmark          # every installed model, CPU and GPU
    python scripts/whisper_test.py --list               # what is downloaded

With no ``--wav`` the script synthesises its own speech sample through the
Windows speech engine, so it needs no recording and no human.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.resampler import TARGET_SAMPLE_RATE, resample_array  # noqa: E402
from app.transcription import models  # noqa: E402
from app.transcription.cuda_setup import describe as describe_cuda  # noqa: E402
from app.transcription.engine import (  # noqa: E402
    TranscriptionEngine,
    TranscriptionSettings,
)
from app.transcription.hardware import (  # noqa: E402
    Accelerator,
    describe_hardware,
    detect_gpus,
)
from app.utils.logging_setup import setup_logging  # noqa: E402

#: Spoken by the synthesised sample. Deliberately in the register the app is
#: built for — a physics lecture — including terms Whisper finds hard.
SAMPLE_TEXT_IT = (
    "Consideriamo adesso l'Hamiltoniana del sistema. "
    "Nel limite adiabatico la variazione e' sufficientemente lenta, "
    "quindi questo termine puo' essere trascurato."
)
SAMPLE_TEXT_EN = (
    "The system can be described by this Hamiltonian. "
    "In the adiabatic limit the variation is slow enough, "
    "so this term can be neglected."
)


def hr(title: str = "") -> None:
    print(f"\n{'-' * 70}")
    if title:
        print(title)
        print("-" * 70)


# ------------------------------------------------------------------ sample

def synthesize(text: str, italian: bool, destination: Path) -> Path | None:
    """Generate a speech sample with the Windows speech engine.

    Returns None when no suitable voice is installed, so the caller can fall
    back to asking for a real recording.
    """
    voice = "it-IT" if italian else "en-US"
    script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$v = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -eq '{voice}' }} |
     Select-Object -First 1
if ($null -eq $v) {{ $s.Dispose(); exit 2 }}
$s.SelectVoice($v.VoiceInfo.Name)
$s.SetOutputToWaveFile('{destination}')
$s.Speak(@'
{text}
'@)
$s.SetOutputToNull()
$s.Dispose()
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  sintesi vocale non disponibile: {exc}")
        return None

    if result.returncode != 0 or not destination.exists():
        print(f"  nessuna voce {voice} installata in Windows")
        return None
    return destination


def load_wav(path: Path) -> np.ndarray:
    """Read any PCM WAV and return 16 kHz mono float32."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise SystemExit(f"Formato WAV non supportato: {width * 8} bit")

    return resample_array(np.frombuffer(raw, dtype=dtype), rate, TARGET_SAMPLE_RATE, channels)


# ------------------------------------------------------------------ scoring

def word_accuracy(reference: str, hypothesis: str) -> float:
    """Fraction of reference words present in the hypothesis.

    A deliberately forgiving measure — enough to tell a working engine from a
    broken one without pulling in an edit-distance dependency.

    Accents are folded before comparing. The reference is written in ASCII
    ("e'", "puo'") because that is what survives the trip through PowerShell to
    the speech engine, while Whisper correctly writes "è" and "può"; without
    folding, the metric would punish the engine for being right.
    """
    import re
    import unicodedata

    def normalise(text: str) -> list[str]:
        folded = unicodedata.normalize("NFD", text.lower())
        folded = "".join(c for c in folded if not unicodedata.combining(c))
        # Treat a trailing apostrophe as an accent: "e'" -> "e".
        folded = re.sub(r"([aeiou])'", r"\1", folded)
        return re.sub(r"[^\w\s]", " ", folded).split()

    ref, hyp = normalise(reference), set(normalise(hypothesis))
    if not ref:
        return 0.0
    return sum(1 for word in ref if word in hyp) / len(ref)


# ------------------------------------------------------------------ running

def run_one(
    model_key: str,
    accelerator: Accelerator,
    audio: np.ndarray,
    language: str,
    reference: str = "",
    verbose: bool = True,
) -> dict | None:
    """Load a model, transcribe, and report. Returns None if it could not run."""
    duration = audio.size / TARGET_SAMPLE_RATE

    load_started = time.monotonic()
    engine = TranscriptionEngine(model_key, accelerator=accelerator)
    try:
        choice = engine.load()
    except Exception as exc:
        user_message = getattr(exc, "user_message", str(exc))
        print(f"  {model_key:9s} {accelerator.value:4s}  ERRORE: {user_message}")
        return None
    load_time = time.monotonic() - load_started

    if engine.load_warning and verbose:
        print(f"  ! {engine.load_warning}")

    try:
        result = engine.transcribe(audio, TranscriptionSettings(language=language))
    except Exception as exc:
        print(f"  {model_key:9s} {accelerator.value:4s}  ERRORE: "
              f"{getattr(exc, 'user_message', exc)}")
        engine.unload()
        return None

    accuracy = word_accuracy(reference, result.text) if reference else float("nan")

    row = {
        "model": model_key,
        "requested": accelerator.value,
        "device": choice.device,
        "compute_type": choice.compute_type,
        "load_time": load_time,
        "inference_time": result.inference_time,
        "rtf": result.real_time_factor,
        "audio_duration": duration,
        "language": result.language,
        "language_probability": result.language_probability,
        "text": result.text,
        "accuracy": accuracy,
        "segments": len(result.segments),
    }
    engine.unload()
    return row


def print_detail(row: dict, reference: str) -> None:
    hr(f"{row['model'].upper()}  |  {row['device']}/{row['compute_type']}")
    print(f"  caricamento    : {row['load_time']:6.2f}s  (include warm-up)")
    print(f"  inferenza      : {row['inference_time']:6.2f}s  su "
          f"{row['audio_duration']:.2f}s di audio")
    print(f"  real-time factor: {row['rtf']:5.1f}x"
          + ("   (sufficiente per il tempo reale)" if row["rtf"] >= 1.5 else
             "   (TROPPO LENTO per il tempo reale)"))
    print(f"  lingua         : {row['language']} ({row['language_probability']:.2f})")
    print(f"  segmenti       : {row['segments']}")
    if reference:
        print(f"  parole corrette: {row['accuracy'] * 100:.0f}%")
        print(f"\n  atteso   : {reference}")
    print(f"  ottenuto : {row['text']}")


def benchmark(audio: np.ndarray, language: str, reference: str) -> int:
    installed = models.installed_models()
    if not installed:
        print("\nNessun modello scaricato. Usa --download <modello>.")
        return 1

    accelerators = [Accelerator.CPU]
    if detect_gpus():
        accelerators.insert(0, Accelerator.GPU)

    rows: list[dict] = []
    hr("BENCHMARK")
    print(f"  audio: {audio.size / TARGET_SAMPLE_RATE:.2f}s\n")
    for spec in installed:
        for accelerator in accelerators:
            print(f"  ... {spec.key} su {accelerator.value}", flush=True)
            row = run_one(spec.key, accelerator, audio, language, reference, verbose=False)
            if row:
                rows.append(row)

    hr("RISULTATI")
    header = f"  {'modello':10s} {'device':14s} {'load':>7s} {'infer':>7s} {'RTF':>7s} {'parole':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        accuracy = "" if row["accuracy"] != row["accuracy"] else f"{row['accuracy'] * 100:5.0f}%"
        print(f"  {row['model']:10s} {row['device'] + '/' + row['compute_type']:14s} "
              f"{row['load_time']:6.2f}s {row['inference_time']:6.2f}s "
              f"{row['rtf']:6.1f}x {accuracy:>7s}")

    realtime = [r for r in rows if r["rtf"] >= 1.5]
    hr("CONCLUSIONE")
    if realtime:
        best = max(realtime, key=lambda r: (r["accuracy"] if r["accuracy"] == r["accuracy"] else 0,
                                            r["rtf"]))
        print("  Miglior modello utilizzabile in tempo reale su questa macchina:")
        print(f"    {best['model']} su {best['device']}/{best['compute_type']} "
              f"(RTF {best['rtf']:.1f}x)")
    else:
        print("  Nessuna configurazione raggiunge il tempo reale. Usa un modello piu' piccolo.")
    return 0


# ------------------------------------------------------------------ entry

def main() -> int:
    parser = argparse.ArgumentParser(description="LiveTranscriber transcription diagnostic")
    parser.add_argument("--model", default="", help="modello da usare (default: il piu' piccolo installato)")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--wav", type=Path, help="file WAV da trascrivere")
    parser.add_argument("--language", default="it", help="lingua (it, en, auto)")
    parser.add_argument("--english", action="store_true", help="usa il campione inglese")
    parser.add_argument("--benchmark", action="store_true", help="prova tutti i modelli installati")
    parser.add_argument("--list", action="store_true", help="elenca i modelli")
    parser.add_argument("--download", default="", help="scarica un modello ed esci")
    args = parser.parse_args()

    setup_logging(console=False)

    hr("HARDWARE")
    print(describe_hardware())
    print(describe_cuda())

    hr("MODELLI")
    for spec in models.list_models():
        state = f"scaricato ({models.disk_usage_mb(spec.key)} MB)" if models.is_available(spec.key) \
                else f"non scaricato ({spec.size_label})"
        star = "*" if spec.recommended else " "
        print(f" {star} {spec.display_name:10s} {spec.quality:18s} {state}")
    if args.list:
        return 0

    if args.download:
        spec = models.get_spec(args.download)
        print(f"\nScarico {spec.display_name} ({spec.size_label})...")
        last = [0.0]

        def progress(done: int, total: int) -> None:
            now = time.time()
            if now - last[0] > 0.5:
                last[0] = now
                pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
                print(f"\r  {pct} {done / 1048576:7.1f}/{total / 1048576:7.1f} MB",
                      end="", flush=True)

        models.download(args.download, on_progress=progress)
        print(f"\n  fatto: {models.disk_usage_mb(args.download)} MB")
        return 0

    # Pick a model: the requested one, else the smallest installed.
    if args.model:
        model_key = args.model
        if not models.is_available(model_key):
            spec = models.get_spec(model_key)
            print(f"\nIl modello {spec.display_name} non e' scaricato "
                  f"({spec.size_label}).")
            print(f"Scaricalo con: python scripts/whisper_test.py --download {model_key}")
            return 1
    else:
        installed = models.installed_models()
        if not installed:
            print("\nNessun modello scaricato.")
            print("Scarica il modello consigliato con: "
                  "python scripts/whisper_test.py --download small")
            return 1
        model_key = installed[0].key

    # Get audio.
    reference = ""
    if args.wav:
        if not args.wav.exists():
            print(f"\nFile non trovato: {args.wav}")
            return 1
        audio = load_wav(args.wav)
        print(f"\nAudio: {args.wav} ({audio.size / TARGET_SAMPLE_RATE:.2f}s)")
    else:
        hr("CAMPIONE AUDIO")
        reference = SAMPLE_TEXT_EN if args.english else SAMPLE_TEXT_IT
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "sample.wav"
            print("  sintesi vocale di Windows...")
            if synthesize(reference, not args.english, destination) is None:
                print("\n  Impossibile generare il campione. Usa --wav con un file audio.")
                return 1
            audio = load_wav(destination)
        print(f"  generato: {audio.size / TARGET_SAMPLE_RATE:.2f}s")

    language = args.language
    if args.english and language == "it":
        language = "en"

    if args.benchmark:
        return benchmark(audio, language, reference)

    accelerator = Accelerator(args.device)
    row = run_one(model_key, accelerator, audio, language, reference)
    if row is None:
        return 1
    print_detail(row, reference)

    hr()
    if row["rtf"] < 1.5:
        print("ATTENZIONE: questa configurazione e' troppo lenta per il tempo reale.")
        return 0
    print("RISULTATO: motore di trascrizione funzionante.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
