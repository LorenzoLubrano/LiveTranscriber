"""Make pip-installed NVIDIA CUDA libraries loadable on Windows.

``pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`` puts its DLLs inside the
site-packages tree::

    .venv/Lib/site-packages/nvidia/cublas/bin/cublas64_12.dll
    .venv/Lib/site-packages/nvidia/cudnn/bin/cudnn64_9.dll
    .venv/Lib/site-packages/nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll

Nothing puts those directories on the DLL search path. On Linux the wheels
patch an RPATH and it works; on Windows CTranslate2 fails at the first CUDA
operation with::

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

Since Python 3.8, Windows ignores ``PATH`` for extension-module dependencies —
``os.add_dll_directory`` is the supported mechanism. Calling it here, before any
CUDA work, is what makes the GPU path function at all.

Frozen builds are handled too: PyInstaller copies the same tree next to the
executable, so the bundle directory is searched as well — and so is the optional
pack the user can download from inside the app
(:mod:`app.transcription.cuda_pack`), which lands in per-user data rather than in
the application folder.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from app.utils.paths import bundle_dir, is_frozen

logger = logging.getLogger(__name__)

#: Packages that ship CUDA DLLs we depend on.
_CUDA_PACKAGES = ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime")

_registered: list[Path] | None = None
_handles: list[object] = []  # keep the cookies alive for the process lifetime


def _nvidia_roots() -> list[Path]:
    """Directories that may contain an ``nvidia`` package tree."""
    roots: list[Path] = []

    # First: the pack the user downloaded from inside the app, if any. Searched
    # ahead of the bundle so that a downloaded runtime wins over a stale one
    # shipped with an older copy of the application folder.
    try:
        from app.transcription.cuda_pack import pack_dir

        downloaded = pack_dir()
        if (downloaded / "nvidia").is_dir():
            roots.append(downloaded)
    except Exception as exc:  # pragma: no cover - only if paths are unavailable
        logger.debug("Could not check for a downloaded CUDA pack: %r", exc)

    if is_frozen():
        roots.append(bundle_dir())

    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
        if spec is not None:
            for location in spec.submodule_search_locations or []:
                roots.append(Path(location).parent)
    except Exception as exc:  # pragma: no cover - import machinery edge cases
        logger.debug("Could not locate the nvidia package: %r", exc)

    # Fall back to scanning sys.path for a site-packages directory.
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry)
        if (candidate / "nvidia").is_dir():
            roots.append(candidate)

    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def find_cuda_dirs() -> list[Path]:
    """Every directory holding NVIDIA CUDA DLLs, in load order."""
    found: list[Path] = []
    for root in _nvidia_roots():
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for package in _CUDA_PACKAGES:
            for sub in ("bin", "lib"):
                path = nvidia / package / sub
                if path.is_dir() and any(path.glob("*.dll")):
                    found.append(path)
    return found


def ensure_cuda_libraries() -> list[Path]:
    """Register CUDA DLL directories with Windows. Idempotent and safe.

    Returns the directories that were made loadable. An empty list means no
    pip-installed CUDA libraries are present, which is normal on a CPU-only
    install and must not be treated as an error.
    """
    global _registered
    if _registered is not None:
        return _registered

    dirs = find_cuda_dirs()
    registered: list[Path] = []

    for path in dirs:
        try:
            # Python 3.8+ on Windows; absent on other platforms.
            if hasattr(os, "add_dll_directory"):
                _handles.append(os.add_dll_directory(str(path)))
            registered.append(path)
        except OSError as exc:
            logger.warning("Could not add DLL directory %s: %r", path, exc)

    # Some libraries still consult PATH; harmless to help them along.
    if registered:
        extra = os.pathsep.join(str(p) for p in registered)
        os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")
        logger.info("Registered %d CUDA library directories", len(registered))
    else:
        logger.info("No pip-installed CUDA libraries found (CPU-only install)")

    _registered = registered
    return registered


def forget_registration() -> None:
    """Drop the cached result, so a newly installed pack is picked up.

    Directories already handed to ``add_dll_directory`` stay registered — the
    cookies are deliberately kept alive — so this only allows new ones to be
    added, which is exactly what installing the pack needs.
    """
    global _registered
    _registered = None


def cuda_libraries_available() -> bool:
    """True if the CUDA runtime DLLs this build needs are present on disk."""
    dirs = ensure_cuda_libraries()
    if not dirs:
        return False
    names = {f.name.lower() for path in dirs for f in path.glob("*.dll")}
    return any(n.startswith("cublas64") for n in names)


def describe() -> str:
    """Human-readable summary for diagnostics."""
    dirs = ensure_cuda_libraries()
    if not dirs:
        return "CUDA libraries: not installed (pip install -e \".[gpu]\")"
    lines = [f"CUDA libraries: {len(dirs)} directories registered"]
    lines.extend(f"  {path}" for path in dirs)
    return "\n".join(lines)
