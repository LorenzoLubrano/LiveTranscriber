"""Real-hardware audio diagnostic (spec §22).

Lists devices, identifies WASAPI loopbacks, records from a chosen source and
verifies the captured signal is not silent.

Usage::

    python scripts/audio_test.py                 # interactive picker
    python scripts/audio_test.py --list          # enumerate and exit
    python scripts/audio_test.py --loopback      # record default PC audio
    python scripts/audio_test.py --mic           # record default microphone
    python scripts/audio_test.py --both          # both at once, three WAVs
    python scripts/audio_test.py --seconds 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.capture import CaptureStream, CaptureWatchdog  # noqa: E402
from app.audio.devices import (  # noqa: E402
    AudioDevice,
    DeviceError,
    default_loopback,
    default_microphone,
    list_loopbacks,
    list_microphones,
    list_outputs,
    terminate_pyaudio,
)
from app.audio.recorder import MixRecorder  # noqa: E402
from app.audio.resampler import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    StreamResampler,
    peak_level,
    rms_dbfs,
)
from app.audio.ring_buffer import RingBuffer  # noqa: E402
from app.utils.logging_setup import setup_logging  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"

#: Below this RMS a recording is treated as silence rather than quiet audio.
SILENCE_DBFS = -70.0


# ---------------------------------------------------------------- reporting

def hr(title: str = "") -> None:
    print(f"\n{'-' * 68}")
    if title:
        print(title)
        print("-" * 68)


def print_devices() -> None:
    hr("MICROFONI (WASAPI)")
    mics = list_microphones()
    for i, d in enumerate(mics):
        print(f"  [{i}] {d.label}")
        print(f"      {d.host_api_name}  idx={d.index}  "
              f"{d.max_input_channels}ch  {d.default_sample_rate} Hz")
    if not mics:
        print("  (nessuno)")

    hr("AUDIO PC - loopback WASAPI")
    loops = list_loopbacks()
    for i, d in enumerate(loops):
        print(f"  [{i}] {d.label}")
        print(f"      mirrors: {d.render_endpoint}")
        print(f"      {d.host_api_name}  idx={d.index}  "
              f"{d.max_input_channels}ch  {d.default_sample_rate} Hz")
    if not loops:
        print("  (nessuno - collega altoparlanti o cuffie)")

    hr("USCITE (riproduzione)")
    for d in list_outputs():
        print(f"  - {d.label}  ({d.default_sample_rate} Hz)")


def choose(devices: list[AudioDevice], what: str) -> AudioDevice | None:
    if not devices:
        print(f"Nessun dispositivo disponibile per: {what}")
        return None
    if len(devices) == 1:
        print(f"Unico dispositivo {what}: {devices[0].display_name}")
        return devices[0]

    print(f"\nScegli {what}:")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d.label}")
    try:
        raw = input(f"Numero [0-{len(devices) - 1}], invio per il default: ").strip()
    except EOFError:
        raw = ""
    if not raw:
        return next((d for d in devices if d.is_default), devices[0])
    try:
        return devices[int(raw)]
    except (ValueError, IndexError):
        print("Scelta non valida.")
        return None


# ---------------------------------------------------------------- recording

class SourceRun:
    """Capture + resample + record one source for the duration of the test."""

    def __init__(self, device: AudioDevice, label: str) -> None:
        self.device = device
        self.label = label
        self.ring = RingBuffer(
            capacity_frames=int(device.default_sample_rate * 30),
            channels=device.capture_channels,
        )
        self.resampler = StreamResampler(
            device.default_sample_rate, device.capture_channels
        )
        self.stream = CaptureStream(device, self.ring, on_error=self._on_error)
        self.collected: list[np.ndarray] = []
        self.error: str | None = None

    def _on_error(self, message: str) -> None:
        self.error = message

    def drain(self) -> np.ndarray:
        """Pull whatever the callback has produced and convert it to 16 kHz mono."""
        raw = self.ring.read_all()
        if raw.size == 0:
            return np.zeros(0, dtype=np.float32)
        out = self.resampler.process(raw)
        if out.size:
            self.collected.append(out)
        return out

    def finish(self) -> np.ndarray:
        self.drain()
        tail = self.resampler.flush()
        if tail.size:
            self.collected.append(tail)
        if not self.collected:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.collected)


def record(devices: dict[str, AudioDevice], seconds: float) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {label: SourceRun(dev, label) for label, dev in devices.items()}

    hr("REGISTRAZIONE")
    for label, run in runs.items():
        print(f"  {label:4s} <- {run.device.display_name} "
              f"({run.device.default_sample_rate} Hz, {run.device.capture_channels}ch)")

    if "pc" in runs:
        print("\n  >>> RIPRODUCI DELL'AUDIO SUL PC ADESSO (video, musica, YouTube) <<<")
    if "mic" in runs:
        print("  >>> PARLA NEL MICROFONO <<<")

    watchdog = CaptureWatchdog()
    try:
        for run in runs.values():
            run.stream.start()
            watchdog.add(run.stream)
        watchdog.start()
    except DeviceError as exc:
        print(f"\nERRORE: {exc.user_message}")
        print(f"        ({exc.technical})")
        for run in runs.values():
            run.stream.stop()
        watchdog.stop()
        return 1

    recorder = MixRecorder(
        OUT_DIR / "pc_audio.wav" if "pc" in runs else None,
        OUT_DIR / "microphone.wav" if "mic" in runs else None,
        OUT_DIR / "mixed.wav",
        TARGET_SAMPLE_RATE,
    )

    print()
    start = time.monotonic()
    try:
        while (elapsed := time.monotonic() - start) < seconds:
            for label, run in runs.items():
                chunk = run.drain()
                if chunk.size:
                    if label == "pc":
                        recorder.write_pc(chunk)
                    else:
                        recorder.write_mic(chunk)

            meters = "   ".join(
                f"{label.upper():3s} {bar(run.stream.level)}" for label, run in runs.items()
            )
            print(f"\r  {elapsed:5.1f}s / {seconds:.0f}s   {meters}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n  interrotto")
    finally:
        print()
        watchdog.stop()
        for run in runs.values():
            run.stream.stop()

    finals = {label: run.finish() for label, run in runs.items()}
    for label, audio in finals.items():
        if audio.size:
            if label == "pc":
                recorder.write_pc(np.zeros(0, dtype=np.float32))
            else:
                recorder.write_mic(np.zeros(0, dtype=np.float32))
    recorder.close()

    return verify(runs, finals, recorder, seconds)


def bar(level: float, width: int = 20) -> str:
    filled = int(min(1.0, level) * width)
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def verify(runs, finals, recorder, seconds: float) -> int:
    hr("VERIFICA")
    failures = 0

    for label, run in runs.items():
        audio = finals[label]
        stats = run.stream.stats
        duration = audio.size / TARGET_SAMPLE_RATE
        dbfs = rms_dbfs(audio)
        peak = peak_level(audio)

        print(f"\n  [{label.upper()}] {run.device.display_name}")
        print(f"    callbacks     : {stats.callback_count}")
        print(f"    frames        : {stats.frames_captured}")
        print(f"    dropped       : {stats.frames_dropped} ({stats.dropped_ratio:.2%})")
        print(f"    overflows     : {stats.input_overflows}")
        print(f"    durata 16 kHz : {duration:.2f}s")
        print(f"    RMS           : {dbfs:.1f} dBFS")
        print(f"    picco         : {peak:.4f}")

        if run.error:
            print(f"    FAIL: {run.error}")
            failures += 1
            continue
        if stats.callback_count == 0:
            print("    FAIL: nessun callback audio ricevuto")
            failures += 1
            continue
        if duration < seconds * 0.8:
            print(f"    FAIL: durata troppo breve (attesi ~{seconds:.0f}s)")
            failures += 1
            continue
        if dbfs < SILENCE_DBFS:
            print("    FAIL: segnale silenzioso")
            if label == "pc":
                print("          -> l'audio era davvero in riproduzione?")
                print("          -> il dispositivo scelto e' quello in uso da Windows?")
            else:
                print("          -> il microfono e' mutato o l'accesso e' bloccato?")
            failures += 1
            continue
        print("    OK: segnale valido")

    hr("FILE")
    for name, path in recorder.paths.items():
        if path.exists():
            size = path.stat().st_size
            print(f"  {name:6s} {path}  ({size / 1024:.0f} KB)")
            if size <= 44:
                print("         FAIL: file vuoto")
                failures += 1

    hr()
    if failures:
        print(f"RISULTATO: {failures} problema/i rilevato/i")
    else:
        print("RISULTATO: tutto OK")
    print(f"Cartella: {OUT_DIR}")
    return 1 if failures else 0


# ---------------------------------------------------------------- entrypoint

def main() -> int:
    parser = argparse.ArgumentParser(description="LiveTranscriber audio diagnostic")
    parser.add_argument("--list", action="store_true", help="elenca i dispositivi ed esci")
    parser.add_argument("--loopback", action="store_true", help="registra l'audio del PC")
    parser.add_argument("--mic", action="store_true", help="registra il microfono")
    parser.add_argument("--both", action="store_true", help="registra entrambi")
    parser.add_argument("--seconds", type=float, default=10.0, help="durata (default 10)")
    args = parser.parse_args()

    setup_logging(console=False)

    try:
        print_devices()
        if args.list:
            return 0

        devices: dict[str, AudioDevice] = {}
        if args.both or args.loopback:
            dev = default_loopback()
            if dev is None:
                print("\nNessun dispositivo loopback disponibile.")
                return 1
            devices["pc"] = dev
        if args.both or args.mic:
            dev = default_microphone()
            if dev is None:
                print("\nNessun microfono disponibile.")
                return 1
            devices["mic"] = dev

        if not devices:
            print("\nCosa vuoi testare?")
            print("  [1] Audio PC (loopback)")
            print("  [2] Microfono")
            print("  [3] Entrambi")
            try:
                choice = input("Scelta [1]: ").strip() or "1"
            except EOFError:
                choice = "1"

            if choice in ("1", "3"):
                dev = choose(list_loopbacks(), "audio PC")
                if dev is None:
                    return 1
                devices["pc"] = dev
            if choice in ("2", "3"):
                dev = choose(list_microphones(), "microfono")
                if dev is None:
                    return 1
                devices["mic"] = dev

        return record(devices, args.seconds)
    finally:
        terminate_pyaudio()


if __name__ == "__main__":
    raise SystemExit(main())
