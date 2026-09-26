"""Whisper model catalogue, download and offline availability.

Models are stored in the application's own directory rather than the shared
Hugging Face cache, so uninstalling LiveTranscriber takes its multi-gigabyte
downloads with it and the user can see what is using their disk.

Nothing is ever downloaded implicitly. :func:`is_available` tells the caller
whether a model is on disk; the UI asks before fetching one, showing its size
(spec §6). Once downloaded, loading passes ``local_files_only=True`` so the app
never touches the network again — that is what makes it work offline.
"""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tqdm import tqdm

from app.utils.paths import models_dir

logger = logging.getLogger(__name__)

#: Called with (downloaded_bytes, total_bytes). total may be 0 while unknown.
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the model picker."""

    key: str            # our identifier, also the settings value
    repo_id: str        # full Hugging Face repository holding the CT2 conversion
    display_name: str
    quality: str        # short label shown next to the name
    description: str
    approx_size_mb: int
    recommended: bool = False

    @property
    def size_label(self) -> str:
        if self.approx_size_mb >= 1024:
            return f"{self.approx_size_mb / 1024:.1f} GB"
        return f"{self.approx_size_mb} MB"

    @property
    def directory(self) -> Path:
        return models_dir() / self.key

    def __str__(self) -> str:
        return f"{self.display_name} ({self.size_label})"


#: Sizes are of the CTranslate2 float16 conversions actually downloaded.
CATALOGUE: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="tiny",
        repo_id="Systran/faster-whisper-tiny",
        display_name="Tiny",
        quality="Molto veloce",
        description="Il più rapido. Qualità bassa, utile su macchine lente.",
        approx_size_mb=75,
    ),
    ModelSpec(
        key="base",
        repo_id="Systran/faster-whisper-base",
        display_name="Base",
        quality="Veloce",
        description="Buon compromesso su CPU modeste.",
        approx_size_mb=145,
    ),
    ModelSpec(
        key="small",
        repo_id="Systran/faster-whisper-small",
        display_name="Small",
        quality="Consigliato",
        description="Qualità solida con latenza contenuta. Scelta predefinita.",
        approx_size_mb=484,
        recommended=True,
    ),
    ModelSpec(
        key="medium",
        repo_id="Systran/faster-whisper-medium",
        display_name="Medium",
        quality="Alta qualità",
        description="Più preciso di Small, ma più lento di Turbo a pari qualità.",
        approx_size_mb=1530,
    ),
    ModelSpec(
        key="large-v3",
        repo_id="Systran/faster-whisper-large-v3",
        display_name="Large-v3",
        quality="Massima qualità",
        description="La massima accuratezza. Richiede GPU per l'uso in tempo reale.",
        approx_size_mb=3090,
    ),
    ModelSpec(
        key="turbo",
        repo_id="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        display_name="Turbo",
        quality="Qualità/velocità",
        description="Qualità vicina a Large-v3 a metà del costo di Medium. La scelta migliore con GPU.",
        approx_size_mb=1620,
    ),
)

_BY_KEY = {spec.key: spec for spec in CATALOGUE}

DEFAULT_MODEL = "small"


class Quality(StrEnum):
    """First-run choice (spec §25), in words a normal user understands."""

    FAST = "fast"
    BALANCED = "balanced"
    BEST = "best"

    @property
    def label(self) -> str:
        return {
            Quality.FAST: "Veloce",
            Quality.BALANCED: "Bilanciata",
            Quality.BEST: "Massima qualità",
        }[self]


#: Which model each quality maps to, with and without a GPU.
#:
#: The split exists because **streaming costs far more than transcribing a file
#: once**. LocalAgreement re-transcribes a growing buffer, so each second of
#: audio passes through Whisper roughly four or five times, and a one-shot
#: benchmark must never be used to choose a live model.
#:
#: These two columns are a hardware *guess*, and the only thing they know is
#: whether there is a GPU. That is not enough, which is why
#: :mod:`app.transcription.calibration` measures the machine and overrides them.
#: Measured here (Ryzen 7 260, six physical cores; RTX 5050), cost per second of
#: audio, medians of repeated runs — lower is better and 1.0 cannot keep up:
#:
#: ========  =========  =========
#: model     CPU int8   GPU fp16
#: ========  =========  =========
#: tiny      0.18       0.07
#: base      0.55       -
#: small     0.97       0.13
#: turbo     1.38       0.20
#: medium    1.82       0.45
#: large-v3  1.81       0.36
#: ========  =========  =========
#:
#: `small` is excellent on this GPU and unusable on the same machine's CPU. That
#: gap is the whole reason the app measures rather than assumes.
#:
#: "Massima qualità" on GPU is Turbo rather than Medium: measured, it costs less
#: than half as much and is close to large-v3 in accuracy.
_QUALITY_MAP: dict[Quality, dict[bool, str]] = {
    #                      GPU         CPU
    Quality.FAST:     {True: "tiny",  False: "tiny"},
    Quality.BALANCED: {True: "small", False: "base"},
    Quality.BEST:     {True: "turbo", False: "small"},
}


def recommend_model(quality: Quality, has_gpu: bool) -> ModelSpec:
    """Model for a quality preset, as a starting point only.

    A preset is what the user *wants*; whether this machine can deliver it live
    is a separate question, and the answer varies by more than a factor of two
    between CPUs that look alike on paper. So the caller is expected to pass the
    result through :func:`app.transcription.calibration.cap_to_measurements`,
    which steps it down to what has actually been measured here.

    On CPU, "Massima qualità" deliberately still names `small`: it is the right
    answer on a fast desktop, and on a slow laptop the measurement takes it away
    again with a reason the user can read.
    """
    return get_spec(_QUALITY_MAP[quality][bool(has_gpu)])


def quality_of(model_key: str, has_gpu: bool) -> Quality | None:
    """Reverse lookup, so the settings screen can show the current preset."""
    for quality, mapping in _QUALITY_MAP.items():
        if mapping[bool(has_gpu)] == model_key:
            return quality
    return None


class ModelError(RuntimeError):
    """Model problem, with a message safe to show to a user."""

    def __init__(self, user_message: str, technical: str = "") -> None:
        super().__init__(technical or user_message)
        self.user_message = user_message
        self.technical = technical or user_message


def get_spec(key: str) -> ModelSpec:
    """Look up a model by key. Raises :class:`ModelError` if unknown."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise ModelError(
            f"Modello sconosciuto: {key}",
            f"unknown model key {key!r}; known: {sorted(_BY_KEY)}",
        ) from None


def list_models() -> tuple[ModelSpec, ...]:
    return CATALOGUE


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------


def _model_files(directory: Path) -> tuple[Path | None, list[Path]]:
    """Return (model.bin, other required files) found under ``directory``."""
    if not directory.is_dir():
        return None, []
    weights = next(iter(directory.rglob("model.bin")), None)
    return weights, [p for p in directory.rglob("*") if p.is_file()]


def is_available(key: str) -> bool:
    """True if the model is on disk and usable without a network connection.

    Checks for the actual weights rather than merely the directory: an
    interrupted download leaves a directory full of ``.incomplete`` files, and
    treating that as success produces a confusing failure at load time.
    """
    spec = get_spec(key)
    weights, _ = _model_files(spec.directory)
    if weights is None:
        return False
    if weights.stat().st_size < 1024 * 1024:
        logger.warning("Model %s has a suspiciously small model.bin", key)
        return False
    return True


def installed_models() -> list[ModelSpec]:
    return [spec for spec in CATALOGUE if is_available(spec.key)]


def disk_usage_mb(key: str) -> int:
    """How much space a downloaded model actually occupies."""
    spec = get_spec(key)
    _, files = _model_files(spec.directory)
    return int(sum(f.stat().st_size for f in files) / 1_048_576)


def delete_model(key: str) -> bool:
    """Remove a downloaded model. Returns True if something was deleted."""
    spec = get_spec(key)
    if not spec.directory.exists():
        return False
    shutil.rmtree(spec.directory, ignore_errors=True)
    logger.info("Deleted model %s", key)
    return not spec.directory.exists()


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


#: Files a CTranslate2 Whisper repository actually needs. Downloading only
#: these skips the PyTorch weights some repositories also publish, which can be
#: larger than the model we want.
_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


class _ProgressTqdm(tqdm):
    """Adapts huggingface_hub's progress bars onto a plain callback.

    ``snapshot_download`` reports progress by instantiating a tqdm class and
    calling ``update``; ``tqdm_class`` is the supported injection point. This
    keeps Qt out of this module entirely.

    It subclasses the real :class:`tqdm.tqdm` rather than reimplementing its
    surface, because huggingface_hub uses more of that surface than ``update``:
    it reads and *writes* ``.total`` to build an aggregate bar, so a stand-in
    that answers every attribute generically breaks with a TypeError. Output is
    disabled; only the numbers are observed.

    Progress is the **maximum** over the byte-counting bars, not their sum.
    huggingface_hub instantiates this class more than once per download — a
    transfer bar and a reconstruct bar — and grows *each* of them to the full
    size, while the individual files are tracked by an internal class that never
    reaches us. Summing therefore double-counts the total and reports ~50% on a
    finished download. Bars that count files rather than bytes are excluded.

    (faster-whisper's own ``download_model`` cannot be used for this: it hard-
    codes ``tqdm_class=disabled_tqdm``, and it still passes the
    ``local_dir_use_symlinks`` argument that huggingface_hub 1.x removed, so it
    raises TypeError whenever an output directory is given.)
    """

    _sink: ProgressCallback | None = None
    _agg_lock = threading.Lock()
    #: id -> (completed, total) for each byte-counting bar seen.
    _bars: dict[int, tuple[int, int]] = {}

    def __init__(self, *args, **kwargs) -> None:
        # tqdm uses `unit` for formatting but does not keep it as an instance
        # attribute, so record it before delegating.
        self._lt_counts_bytes = kwargs.get("unit") == "B"
        # A disabled tqdm returns from update() before touching self.n, so the
        # byte count has to be kept here. `total`, by contrast, is assigned as a
        # plain attribute by huggingface_hub and stays correct.
        self._lt_n = int(kwargs.get("initial") or 0)
        kwargs["disable"] = True  # never write to the console
        super().__init__(*args, **kwargs)

    def update(self, n: int = 1) -> bool | None:
        result = super().update(n)
        cls = type(self)

        if self._lt_counts_bytes:
            self._lt_n += int(n or 0)
            with cls._agg_lock:
                cls._bars[id(self)] = (self._lt_n, int(self.total or 0))
                downloaded = max((b[0] for b in cls._bars.values()), default=0)
                total = max((b[1] for b in cls._bars.values()), default=0)

            sink = cls._sink
            if sink is not None:
                try:
                    sink(downloaded, total)
                except Exception:
                    logger.exception("Download progress callback raised")
        return result

    @classmethod
    def reset(cls, sink: ProgressCallback | None) -> None:
        with cls._agg_lock:
            cls._sink = sink
            cls._bars = {}


def download(
    key: str,
    on_progress: ProgressCallback | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Path:
    """Fetch a model into the application's model directory.

    ``cancel`` is polled between files; a cancelled download leaves the partial
    data in place so a retry resumes rather than restarting.

    Raises :class:`ModelError` with a user-facing message on failure.
    """
    spec = get_spec(key)
    spec.directory.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading model %s (~%s) to %s", key, spec.size_label, spec.directory)

    from huggingface_hub import snapshot_download

    _ProgressTqdm.reset(on_progress)
    try:
        path = snapshot_download(
            spec.repo_id,
            local_dir=str(spec.directory),
            allow_patterns=_ALLOW_PATTERNS,
            local_files_only=False,
            tqdm_class=_ProgressTqdm,
        )
    except Exception as exc:
        raise ModelError(
            _friendly_download_error(exc, spec),
            f"download of {spec.repo_id} failed: {exc!r}",
        ) from exc
    finally:
        _ProgressTqdm.reset(None)

    if cancel is not None and cancel():
        raise ModelError("Download annullato.", "cancelled by caller")

    if not is_available(key):
        raise ModelError(
            f"Il download del modello {spec.display_name} non è riuscito.",
            f"model.bin missing under {spec.directory} after download",
        )

    logger.info("Model %s ready (%d MB on disk)", key, disk_usage_mb(key))
    return Path(path)


def resolve_model_path(key: str) -> str:
    """Path to pass to ``WhisperModel``, for a model already on disk."""
    spec = get_spec(key)
    weights, _ = _model_files(spec.directory)
    if weights is None:
        raise ModelError(
            f"Il modello {spec.display_name} non è disponibile. "
            f"Scaricalo dalle impostazioni ({spec.size_label}).",
            f"no model.bin under {spec.directory}",
        )
    return str(weights.parent)


def _friendly_download_error(exc: Exception, spec: ModelSpec) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(tag in text for tag in ("connection", "network", "dns", "timeout", "resolve")):
        return (
            f"Impossibile scaricare il modello {spec.display_name}: "
            "nessuna connessione a Internet."
        )
    if "space" in text or "disk" in text or "errno 28" in text:
        return (
            f"Spazio su disco insufficiente per il modello {spec.display_name} "
            f"({spec.size_label})."
        )
    if "permission" in text or "access is denied" in text:
        return "Permessi insufficienti per salvare il modello."
    return f"Impossibile scaricare il modello {spec.display_name}."
