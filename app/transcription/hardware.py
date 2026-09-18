"""Accelerator detection and compute-type selection.

Choosing where to run inference is not just "is there a GPU". CTranslate2's
:func:`ctranslate2.get_supported_compute_types` reports what the *build*
supports, not what the *card* supports — on this development machine it happily
lists ``int8`` and ``int8_float16`` for CUDA, while the RTX 5050 (Blackwell,
sm_120) cannot run them:

* CTranslate2 **4.6.2** — *"Disable INT8 for sm120 - Blackwell GPUs."*
* CTranslate2 **4.8.2** — sm_120 runs via PTX JIT.

Asking for INT8 on such a card fails at inference time with
``CUBLAS_STATUS_NOT_SUPPORTED``, well after the model has loaded and the user
has pressed Start. So compute capability is queried explicitly and INT8 is
withheld from Blackwell, rather than trusting the capability list.

Nothing here raises: a machine with no NVIDIA driver, no ``nvidia-smi``, or a
broken CUDA install must fall back to CPU quietly.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)

#: Compute capability at which CTranslate2 stops supporting INT8 on GPU.
BLACKWELL_COMPUTE_CAPABILITY = 12.0

#: Rough VRAM needed per model in float16, in MB. Used to warn before loading,
#: not to forbid it — CTranslate2 allocates lazily and the numbers vary.
MODEL_VRAM_MB = {
    "tiny": 400,
    "base": 600,
    "small": 1300,
    "medium": 3000,
    "large-v3": 5500,
    "turbo": 3200,
}


class Accelerator(StrEnum):
    """What the user asked for, not necessarily what they get."""

    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class GpuInfo:
    """A CUDA device CTranslate2 can actually see."""

    index: int
    name: str
    compute_capability: float
    total_memory_mb: int
    free_memory_mb: int
    driver_version: str = ""

    @property
    def is_blackwell(self) -> bool:
        """sm_120 and above: INT8 is unavailable in CTranslate2."""
        return self.compute_capability >= BLACKWELL_COMPUTE_CAPABILITY

    @property
    def supports_int8(self) -> bool:
        return not self.is_blackwell

    @property
    def short_name(self) -> str:
        """`NVIDIA GeForce RTX 5050 Laptop GPU` -> `RTX 5050 Laptop`."""
        name = self.name.replace("NVIDIA ", "").replace("GeForce ", "").replace(" GPU", "")
        return name.strip()

    def __str__(self) -> str:
        return (
            f"{self.name} (sm_{int(self.compute_capability * 10)}, "
            f"{self.total_memory_mb} MB)"
        )


@dataclass(frozen=True)
class AcceleratorChoice:
    """The resolved decision, ready to hand to CTranslate2."""

    device: str            # "cuda" or "cpu"
    compute_type: str      # "float16", "int8", ...
    device_index: int = 0
    cpu_threads: int = 0
    gpu: GpuInfo | None = None
    #: Set when the user asked for something that could not be honoured.
    fallback_reason: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.device == "cuda"

    @property
    def description(self) -> str:
        """One line for the status bar (spec §7)."""
        if self.is_gpu and self.gpu is not None:
            return f"Accelerazione: NVIDIA {self.gpu.short_name}"
        return "Accelerazione: CPU"

    def __str__(self) -> str:
        base = f"{self.device}/{self.compute_type}"
        if self.is_gpu:
            return f"{base} on device {self.device_index}"
        return f"{base} with {self.cpu_threads or 'auto'} threads"


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def _cuda_device_count() -> int:
    """How many CUDA devices CTranslate2 can use. Never raises."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception as exc:
        logger.info("CUDA unavailable to CTranslate2: %r", exc)
        return 0


def _query_nvidia_smi() -> list[dict[str, str]]:
    """Read GPU properties from nvidia-smi. Returns [] if anything goes wrong."""
    fields = "index,name,compute_cap,memory.total,memory.free,driver_version"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info("nvidia-smi not usable: %r", exc)
        return []

    if result.returncode != 0:
        logger.info("nvidia-smi failed (rc=%s)", result.returncode)
        return []

    rows: list[dict[str, str]] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(dict(zip(fields.split(","), parts, strict=True)))
    return rows


def detect_gpus() -> list[GpuInfo]:
    """Enumerate usable CUDA GPUs.

    A GPU counts only if CTranslate2 can see it *and* nvidia-smi can describe
    it: a card without a working driver is not usable for inference, however it
    appears in Device Manager.
    """
    if _cuda_device_count() == 0:
        return []

    gpus: list[GpuInfo] = []
    for row in _query_nvidia_smi():
        try:
            gpus.append(
                GpuInfo(
                    index=int(row["index"]),
                    name=row["name"],
                    compute_capability=float(row["compute_cap"]),
                    total_memory_mb=int(float(row["memory.total"])),
                    free_memory_mb=int(float(row["memory.free"])),
                    driver_version=row.get("driver_version", ""),
                )
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Unparsable nvidia-smi row %r: %r", row, exc)

    if not gpus:
        logger.info("CTranslate2 reports CUDA devices but nvidia-smi described none")
    return gpus


def default_cpu_threads() -> int:
    """Threads to give CTranslate2 on CPU.

    Physical cores, not logical: Whisper inference is compute-bound and
    hyper-threaded siblings contend for the same FP units, so using every
    logical processor tends to be slower, not faster. Two cores are left for the
    audio callbacks, the GUI and the rest of the system.
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None

    if not physical:
        physical = os.cpu_count() or 4
    return max(1, min(physical, physical - 2) if physical > 4 else physical)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def compute_type_for_gpu(gpu: GpuInfo, preferred: str = "") -> str:
    """Pick a compute type a given card can really execute.

    ``float16`` is the right default for every supported NVIDIA card. INT8 is
    refused on Blackwell regardless of what the caller asked for — honouring it
    would surface as a cuBLAS error mid-transcription.
    """
    if preferred and preferred != "auto":
        if "int8" in preferred and not gpu.supports_int8:
            logger.warning(
                "Ignoring compute_type %r: %s (sm_%d) does not support INT8 in "
                "CTranslate2; using float16",
                preferred,
                gpu.short_name,
                int(gpu.compute_capability * 10),
            )
            return "float16"
        return preferred
    return "float16"


def compute_type_for_cpu(preferred: str = "") -> str:
    """INT8 is both the fastest and the smallest option on CPU."""
    if preferred and preferred != "auto":
        return preferred
    return "int8"


def select_accelerator(
    preference: Accelerator = Accelerator.AUTO,
    preferred_compute_type: str = "",
    model_name: str = "",
    cpu_threads: int = 0,
) -> AcceleratorChoice:
    """Resolve a user preference into a concrete CTranslate2 configuration.

    Never raises and never returns something unusable: when a GPU is requested
    but absent, the result is a CPU configuration carrying ``fallback_reason``
    so the UI can explain what happened.
    """
    threads = cpu_threads or default_cpu_threads()

    def cpu_choice(reason: str = "") -> AcceleratorChoice:
        return AcceleratorChoice(
            device="cpu",
            compute_type=compute_type_for_cpu(preferred_compute_type),
            cpu_threads=threads,
            fallback_reason=reason,
        )

    if preference is Accelerator.CPU:
        return cpu_choice()

    gpus = detect_gpus()
    if not gpus:
        if preference is Accelerator.GPU:
            return cpu_choice(
                "Nessuna GPU NVIDIA compatibile rilevata. "
                "La trascrizione utilizzerà la CPU."
            )
        return cpu_choice()

    gpu = gpus[0]

    # Warn — but do not refuse — when the model looks too big for free VRAM.
    # CTranslate2 allocates lazily and these figures are approximate, so the
    # real authority is whether loading succeeds.
    needed = MODEL_VRAM_MB.get(model_name, 0)
    if needed and gpu.free_memory_mb and needed > gpu.free_memory_mb:
        logger.warning(
            "Model %r needs ~%d MB VRAM, %d MB free on %s",
            model_name,
            needed,
            gpu.free_memory_mb,
            gpu.short_name,
        )

    return AcceleratorChoice(
        device="cuda",
        compute_type=compute_type_for_gpu(gpu, preferred_compute_type),
        device_index=gpu.index,
        cpu_threads=threads,
        gpu=gpu,
    )


def describe_hardware() -> str:
    """Multi-line hardware summary for logs, About and the benchmark."""
    import platform

    lines = [f"OS:  {platform.system()} {platform.release()}"]
    try:
        import psutil

        lines.append(
            f"CPU: {platform.processor() or 'unknown'} "
            f"({psutil.cpu_count(logical=False)} cores / "
            f"{psutil.cpu_count(logical=True)} threads)"
        )
        lines.append(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    except Exception:
        lines.append(f"CPU: {platform.processor() or 'unknown'}")

    gpus = detect_gpus()
    if gpus:
        for gpu in gpus:
            note = " [INT8 unsupported]" if gpu.is_blackwell else ""
            lines.append(f"GPU: {gpu}{note}")
    else:
        lines.append("GPU: none usable for inference")
    return "\n".join(lines)
