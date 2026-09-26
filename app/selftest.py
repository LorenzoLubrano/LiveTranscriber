"""Self-diagnosis, runnable from the packaged application.

    LiveTranscriber.exe --selftest

A frozen GUI build has no console, so a problem on someone else's machine is
otherwise invisible: the window either appears or it does not. This runs the
checks that actually matter — audio devices, the bundled VAD model, whether a
Whisper model can load and transcribe — and writes a plain report the user can
read or send.

It is deliberately ordered cheapest-first and never stops at the first failure:
knowing that audio works but the model does not is far more useful than knowing
only that something is wrong.
"""

from __future__ import annotations

import contextlib
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    advice: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def add(self, name: str, passed: bool, detail: str = "", advice: str = "") -> Check:
        check = Check(name, passed, detail, advice)
        self.checks.append(check)
        return check

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [
            "LiveTranscriber - diagnostica",
            "=" * 60,
            "",
        ]
        for check in self.checks:
            mark = "OK  " if check.passed else "FALLITO"
            lines.append(f"[{mark}] {check.name}")
            if check.detail:
                lines.append(f"         {check.detail}")
            if not check.passed and check.advice:
                lines.append(f"         -> {check.advice}")
        lines.append("")
        lines.append("=" * 60)
        lines.append(
            "Tutto funzionante."
            if self.ok
            else f"{len(self.failures)} problema/i rilevato/i."
        )
        lines.append(f"Durata: {time.time() - self.started:.1f}s")
        return "\n".join(lines)


def _check_environment(report: Report) -> None:
    from app import __version__
    from app.utils.paths import app_data_dir, is_frozen

    report.add(
        "Ambiente",
        True,
        f"LiveTranscriber {__version__} | {platform.system()} {platform.release()} | "
        f"{'pacchetto' if is_frozen() else 'sorgenti'} | Python {platform.python_version()}",
    )
    report.add("Cartella dati", True, str(app_data_dir()))


def _check_audio(report: Report) -> None:
    try:
        from app.audio.devices import list_loopbacks, list_microphones
    except Exception as exc:
        report.add(
            "Sistema audio", False, f"{type(exc).__name__}: {exc}",
            "La libreria audio non si carica. La build potrebbe essere incompleta.",
        )
        return

    try:
        loopbacks = list_loopbacks()
        microphones = list_microphones()
    except Exception as exc:
        report.add(
            "Dispositivi audio", False, f"{type(exc).__name__}: {exc}",
            "Windows non ha restituito i dispositivi audio.",
        )
        return

    report.add(
        "Audio PC (loopback)",
        bool(loopbacks),
        ", ".join(d.display_name for d in loopbacks) or "nessuno",
        "Collega altoparlanti o cuffie, oppure abilita un dispositivo di "
        "riproduzione nelle impostazioni audio di Windows.",
    )
    report.add(
        "Microfoni",
        bool(microphones),
        ", ".join(d.display_name for d in microphones) or "nessuno",
        "Controlla Impostazioni > Privacy > Microfono.",
    )


def _check_vad(report: Report) -> None:
    """The Silero model is bundled data; a packaging slip loses it silently."""
    try:
        from app.transcription.vad import SpeechDetector

        detector = SpeechDetector()
        detector.speech_regions(np.zeros(16000, dtype=np.float32))
        report.add(
            "Rilevamento voce",
            detector.using_silero,
            "modello Silero caricato" if detector.using_silero else "in modalita' ridotta",
            "Il modello VAD non e' incluso nel pacchetto; la qualita' cala ma "
            "l'app funziona.",
        )
    except Exception as exc:
        report.add(
            "Rilevamento voce", False, f"{type(exc).__name__}: {exc}",
            "onnxruntime non si carica.",
        )


def _check_hardware(report: Report) -> None:
    try:
        from app.transcription.hardware import (
            Accelerator,
            cpu_supports_avx2,
            describe_choice,
            select_accelerator,
        )

        choice = select_accelerator(Accelerator.AUTO)
        report.add("Elaborazione", True, describe_choice(choice))

        # Only worth reporting when it is missing, and only when the CPU is what
        # the work will actually run on.
        if not choice.is_gpu and cpu_supports_avx2() is False:
            report.add(
                "Istruzioni CPU",
                False,
                "questa CPU non ha AVX2",
                "La trascrizione funziona ma e' molto piu' lenta. Usa il "
                "modello Tiny e non aspettarti il tempo reale.",
            )
    except Exception as exc:
        report.add("Elaborazione", False, f"{type(exc).__name__}: {exc}")


def _check_models(report: Report) -> None:
    try:
        from app.transcription import models

        installed = models.installed_models()
        report.add(
            "Modelli scaricati",
            bool(installed),
            ", ".join(f"{s.display_name} ({models.disk_usage_mb(s.key)} MB)"
                      for s in installed) or "nessuno",
            "Apri l'app e scarica un modello: serve la connessione una sola volta.",
        )
        return installed
    except Exception as exc:
        report.add("Modelli scaricati", False, f"{type(exc).__name__}: {exc}")
        return []


def _pick_model(installed) -> str:
    """The model the app would really use, so the checks test the real case.

    The configured one when it is on disk, otherwise the smallest installed:
    diagnosing the model the user has selected is the whole point, and testing
    a different one would answer a question nobody asked.
    """
    smallest = min(installed, key=lambda s: s.approx_size_mb).key
    try:
        from app.config.settings import AppSettings

        configured = AppSettings.load().model
    except Exception:
        return smallest
    return configured if any(s.key == configured for s in installed) else smallest


def _check_speed(report: Report, key: str) -> None:
    """Measure whether this machine can keep up, live.

    The single most useful line in the whole report for someone whose app "is
    slow": it separates a machine that is too slow for the chosen model from one
    that is broken, and those two have completely different fixes.
    """
    try:
        from app.transcription.calibration import CalibrationError, measure_streaming_cost

        # Twice, keeping the worse. One measurement is taken at whatever clock
        # the CPU happens to be boosting to: the same model on this machine read
        # 0.71 and 0.90 on consecutive runs, which straddles the threshold and
        # made this check report a different verdict each time. A diagnostic that
        # flips is worse than no diagnostic.
        try:
            runs = [measure_streaming_cost(key), measure_streaming_cost(key)]
        except CalibrationError as exc:
            report.add("Velocita'", True, f"non misurabile: {exc.user_message}")
            return

        usable = [m for m in runs if m.is_usable]
        if not usable:
            report.add(
                "Velocita'", False, "la misura non ha prodotto alcuna inferenza",
                "Il rilevamento della voce o il modello non stanno funzionando.",
            )
            return

        measured = max(usable, key=lambda m: m.cost)
        best = min(usable, key=lambda m: m.cost)
        detail = (
            f"{key}: {measured.headroom:.1f}x il tempo reale "
            f"({measured.verdict.label})"
        )
        # Only when the two runs really disagree, not when they round the same.
        if best.cost > 0 and measured.cost / best.cost > 1.1:
            detail += f"; nella prova migliore {best.headroom:.1f}x"
        if (
            measured.verdict.value != "too_slow"
            and measured.verdict_for(2).value == "too_slow"
        ):
            detail += "; con PC e microfono insieme il costo raddoppia"

        # Fails only when the machine is genuinely slower than the audio. A model
        # with a thin margin is a note, not a fault: it works, and the app says
        # so while recording if it stops working.
        report.add(
            "Velocita'",
            measured.cost < 1.0,
            detail,
            "Questo modello e' piu' lento del tempo reale su questo PC: "
            "scegline uno piu' leggero, altrimenti il testo arriva con ritardo "
            "crescente.",
        )
    except Exception as exc:
        report.add("Velocita'", False, f"{type(exc).__name__}: {exc}")


def _check_transcription(report: Report, key: str) -> None:
    """The real proof: load a model and transcribe a synthetic signal."""
    try:
        from app.transcription.engine import TranscriptionEngine, TranscriptionSettings

        started = time.monotonic()
        engine = TranscriptionEngine(key)
        choice = engine.load()
        load_time = time.monotonic() - started

        # Two seconds of silence: this must produce no text at all, which
        # exercises the model, the VAD gate and the decoder in one go.
        result = engine.transcribe(
            np.zeros(32000, dtype=np.float32), TranscriptionSettings(language="it")
        )
        engine.unload()

        hallucinated = result.text.strip()
        report.add(
            "Trascrizione",
            not hallucinated,
            f"modello {key} caricato in {load_time:.1f}s su {choice.device}/"
            f"{choice.compute_type}"
            + (f"; testo inatteso sul silenzio: {hallucinated!r}" if hallucinated else ""),
            "Il motore produce testo dal silenzio.",
        )
    except Exception as exc:
        report.add(
            "Trascrizione", False,
            f"{type(exc).__name__}: {exc}",
            "Il motore di trascrizione non si avvia. Controlla il file di log.",
        )


def run(write_to: Path | None = None) -> Report:
    """Run every check and return the report."""
    report = Report()
    _check_environment(report)
    _check_audio(report)
    _check_vad(report)
    _check_hardware(report)
    installed = _check_models(report)
    if installed:
        key = _pick_model(installed)
        _check_transcription(report, key)
        _check_speed(report, key)
    else:
        report.add(
            "Trascrizione", False, "nessun modello disponibile per la prova",
            "Scarica un modello dall'app, poi ripeti la diagnostica.",
        )

    if write_to is not None:
        try:
            write_to.parent.mkdir(parents=True, exist_ok=True)
            write_to.write_text(report.render(), encoding="utf-8")
        except OSError:
            pass
    return report


def main(show_dialog: bool = True) -> int:
    """Entry point for ``--selftest``."""
    from app.utils.logging_setup import setup_logging
    from app.utils.paths import app_data_dir, ensure_app_dirs

    ensure_app_dirs()
    setup_logging(console=False)

    destination = app_data_dir() / "diagnostica.txt"
    report = run(destination)
    text = report.render()

    # A frozen GUI build has no console, so print AND show it.
    with contextlib.suppress(Exception):
        print(text)

    if show_dialog:
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance() or QApplication(sys.argv)
            box = QMessageBox()
            box.setWindowTitle("LiveTranscriber - diagnostica")
            box.setIcon(
                QMessageBox.Icon.Information if report.ok else QMessageBox.Icon.Warning
            )
            box.setText(
                "Tutto funzionante."
                if report.ok
                else f"{len(report.failures)} problema/i rilevato/i."
            )
            box.setDetailedText(text)
            box.setInformativeText(f"Rapporto salvato in:\n{destination}")
            box.exec()
            del app
        except Exception:
            pass

    return 0 if report.ok else 1
