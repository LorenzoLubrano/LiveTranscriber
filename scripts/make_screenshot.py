"""Capture the README screenshot from a real recording.

    python scripts/make_screenshot.py [seconds]

Drives the actual main window through an actual recording: Windows' speech
synthesiser speaks a passage through the speakers, the app captures it through
WASAPI loopback exactly as it would capture a lecture, and the window is grabbed
while the transcript is arriving. Nothing in the picture is mocked up, which is
the only honest way to show what an app does.

Recordings are written to a temporary folder rather than the user's Documents,
and removed afterwards.

Plays audio out loud for the duration, so do not run it in a meeting.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASSAGE = (
    "Riprendiamo il discorso sul limite adiabatico. Se il termine di "
    "interazione varia lentamente rispetto al divario di energia, il sistema "
    "resta nello stato fondamentale e possiamo trattare la perturbazione come "
    "una correzione. Quando invece la variazione diventa rapida, quella "
    "approssimazione cade e servono i metodi dipendenti dal tempo. "
    "Nella prossima lezione vedremo un esempio numerico, e poi passeremo "
    "al caso degenere, che richiede di diagonalizzare la perturbazione "
    "nel sottospazio degli stati con la stessa energia."
)


def speak(text: str) -> subprocess.Popen:
    """Start Windows speaking, without waiting for it to finish."""
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$v = $s.GetInstalledVoices() | "
        "Where-Object { $_.VoiceInfo.Culture.Name -eq 'it-IT' } | Select-Object -First 1; "
        "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
        "$s.Rate = -1; "
        f"$s.Speak(@'\n{text}\n'@); $s.Dispose()"
    )
    return subprocess.Popen(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 55.0

    from PySide6.QtWidgets import QApplication

    from app.config.settings import AppSettings
    from app.ui.main_window import MainWindow
    from app.ui.theme import DARK, LIGHT, ThemeMode, stylesheet

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    workspace = Path(tempfile.mkdtemp(prefix="livetranscriber-shot-"))
    settings = AppSettings()
    settings.output_folder = str(workspace)
    settings.first_run_done = True      # the point here is the main window
    settings.show_timestamps = True

    window = MainWindow(settings)
    # Sized so the transcript fills the frame rather than trailing off into
    # empty page: a screenshot is a picture of the app working.
    window.resize(960, 620)
    window.title_edit.setText("Lezione di meccanica quantistica")
    window.show()
    app.processEvents()

    window.start_recording()
    app.processEvents()

    voice = speak(PASSAGE)
    started = time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            app.processEvents()
            time.sleep(0.02)

        shots = {}
        for name, palette, mode in (
            ("dark", DARK, ThemeMode.DARK),
            ("light", LIGHT, ThemeMode.LIGHT),
        ):
            window.set_theme(mode)
            app.setStyleSheet(stylesheet(palette))
            for _ in range(8):
                app.processEvents()
            destination = ROOT / "docs" / f"screenshot-{name}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(destination), "PNG")
            shots[name] = destination

        words = len(window.transcript.text().split())
        print(f"transcript: {words} words")
        for name, path in shots.items():
            print(f"{name}: {path}")
        if words < 10:
            print(
                "WARNING: almost nothing was transcribed. Check that the "
                "speakers are not muted: loopback captures what Windows plays.",
                file=sys.stderr,
            )
    finally:
        window.stop_recording()
        app.processEvents()
        voice.terminate()
        shutil.rmtree(workspace, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
