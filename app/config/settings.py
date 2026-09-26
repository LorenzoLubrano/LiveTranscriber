"""Persistent settings.

Backed by ``QSettings``, which on Windows means the registry under
``HKCU\\Software\\LiveTranscriber``. That is the platform convention and it
survives moving or reinstalling the application folder — which matters for a
portable build, where the app directory may well be a USB stick.

Values are plain Python types. QSettings on Windows returns everything as a
string, so every getter coerces explicitly rather than trusting the type back.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings

from app.transcription.hardware import Accelerator
from app.transcription.models import DEFAULT_MODEL
from app.ui.theme import ThemeMode
from app.utils.paths import APP_NAME, default_recordings_dir

logger = logging.getLogger(__name__)

ORGANISATION = "LiveTranscriber"


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class AppSettings:
    """Everything the app remembers between runs (spec §15)."""

    # Devices, stored by stable key rather than index: PortAudio indices move
    # whenever hardware is plugged in, so an index would silently select the
    # wrong device after a reboot.
    loopback_device_key: str = ""
    microphone_device_key: str = ""
    input_mode: str = "pc"

    language: str = "auto"
    translate_to_english: bool = False
    model: str = DEFAULT_MODEL
    accelerator: str = str(Accelerator.AUTO)

    theme: str = str(ThemeMode.SYSTEM)
    output_folder: str = ""
    show_timestamps: bool = False
    autoscroll: bool = True

    # Advanced (spec §16)
    chunk_duration: float = 2.0
    max_buffer_duration: float = 28.0
    agreement_runs: int = 2
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    beam_size: int = 5
    temperature: float = 0.0
    compute_type: str = "auto"
    cpu_threads: int = 0

    first_run_done: bool = False

    #: Measured streaming cost per model on *this* machine, as JSON:
    #: ``{"cpu/int8": {"small": 1.03}, "cuda/float16": {"small": 0.13}}``.
    #: A machine fact rather than a preference, but it belongs with the other
    #: per-machine state — QSettings is per user per machine, which is exactly
    #: the scope a speed measurement has.
    speed_measurements: str = ""

    #: Fields that are not written to disk.
    _transient: tuple[str, ...] = field(default=("_transient",), repr=False)

    # -- persistence ------------------------------------------------------

    @classmethod
    def load(cls) -> AppSettings:
        store = QSettings(ORGANISATION, APP_NAME)
        settings = cls()
        defaults = cls()

        for name, default in vars(defaults).items():
            if name.startswith("_"):
                continue
            raw = store.value(name)
            if raw is None:
                continue
            if isinstance(default, bool):
                value: Any = _as_bool(raw, default)
            elif isinstance(default, int):
                value = _as_int(raw, default)
            elif isinstance(default, float):
                value = _as_float(raw, default)
            else:
                value = str(raw)
            setattr(settings, name, value)

        if not settings.output_folder:
            settings.output_folder = str(default_recordings_dir())
        return settings

    def save(self) -> None:
        store = QSettings(ORGANISATION, APP_NAME)
        for name, value in vars(self).items():
            if name.startswith("_"):
                continue
            store.setValue(name, value)
        store.sync()
        logger.debug("Settings saved")

    def reset_advanced(self) -> None:
        """Spec §16: restore the recommended values, keeping user choices."""
        defaults = AppSettings()
        for name in (
            "chunk_duration",
            "max_buffer_duration",
            "agreement_runs",
            "vad_enabled",
            "vad_threshold",
            "min_speech_duration_ms",
            "min_silence_duration_ms",
            "beam_size",
            "temperature",
            "compute_type",
            "cpu_threads",
        ):
            setattr(self, name, getattr(defaults, name))

    # -- typed views ------------------------------------------------------

    @property
    def theme_mode(self) -> ThemeMode:
        try:
            return ThemeMode(self.theme)
        except ValueError:
            return ThemeMode.SYSTEM

    @property
    def accelerator_choice(self) -> Accelerator:
        try:
            return Accelerator(self.accelerator)
        except ValueError:
            return Accelerator.AUTO

    @property
    def output_path(self) -> Path:
        return Path(self.output_folder) if self.output_folder else default_recordings_dir()

    def streaming_settings(self):
        from app.transcription.streaming import StreamingSettings

        return StreamingSettings(
            chunk_duration=self.chunk_duration,
            max_buffer_duration=self.max_buffer_duration,
            agreement_runs=self.agreement_runs,
        ).validate()

    def vad_settings(self):
        from app.transcription.vad import VadSettings

        return VadSettings(
            threshold=self.vad_threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
        )

    # -- measured speed ---------------------------------------------------

    def _measurements(self) -> dict[str, dict[str, float]]:
        if not self.speed_measurements:
            return {}
        try:
            data = json.loads(self.speed_measurements)
        except (TypeError, ValueError):
            logger.warning("Ignoring unreadable speed measurements")
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(device): {str(k): float(v) for k, v in models.items()}
            for device, models in data.items()
            if isinstance(models, dict)
        }

    def measured_cost(self, model_key: str, device: str, compute_type: str) -> float | None:
        """Streaming cost measured for this model here, or None.

        Keyed by device *and* compute type: a float16 GPU result says nothing
        about what the CPU can sustain, and choosing a model from the wrong one
        is how a machine ends up with a model it cannot keep up with.
        """
        bucket = self._measurements().get(f"{device}/{compute_type}", {})
        value = bucket.get(model_key)
        return float(value) if value else None

    def remember_measurement(
        self, model_key: str, device: str, compute_type: str, cost: float
    ) -> None:
        """Store one measurement, replacing any earlier one for that model."""
        if cost <= 0:
            return
        data = self._measurements()
        data.setdefault(f"{device}/{compute_type}", {})[model_key] = round(cost, 4)
        self.speed_measurements = json.dumps(data, separators=(",", ":"))
