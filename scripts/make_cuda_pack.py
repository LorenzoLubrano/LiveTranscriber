"""Build the downloadable NVIDIA support pack.

    python scripts/make_cuda_pack.py

Produces ``dist/livetranscriber-cuda-<version>-windows-x64.zip`` and prints the
SHA-256 to pin in :mod:`app.transcription.cuda_pack`.

The pack exists so there can be one download for everybody. The plain build is
127 MB and runs anywhere; someone with an NVIDIA card presses a button and gets
the GPU, instead of fetching a separate two-gigabyte build.

It contains cuBLAS and nothing else, which is the measured requirement — see
``scripts/LiveTranscriber.spec`` for how that was established. The layout matches
what :mod:`app.transcription.cuda_setup` looks for::

    nvidia/cublas/bin/cublas64_12.dll
    nvidia/cublas/bin/cublasLt64_12.dll
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACKAGES = ("cublas",)


def source_root() -> Path:
    import nvidia

    # A namespace package: __file__ is None, only __path__ exists.
    return Path(list(nvidia.__path__)[0])


def pack_version() -> str:
    try:
        return version("nvidia-cublas-cu12")
    except PackageNotFoundError:
        return "unknown"


def main() -> int:
    root = source_root()
    files: list[tuple[Path, str]] = []
    for package in PACKAGES:
        for dll in sorted((root / package).rglob("*.dll")):
            files.append((dll, str(Path("nvidia") / dll.relative_to(root)).replace("\\", "/")))

    if not files:
        print("No CUDA DLLs found. Install the GPU extras first:", file=sys.stderr)
        print('    pip install -e ".[gpu]"', file=sys.stderr)
        return 1

    destination = ROOT / "dist" / f"livetranscriber-cuda-{pack_version()}-windows-x64.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Written to a temporary file and moved into place, so an interrupted run
    # never leaves a half-written archive that looks finished.
    with tempfile.NamedTemporaryFile(
        delete=False, dir=destination.parent, suffix=".part"
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for source, arcname in files:
                print(f"  {arcname}  ({source.stat().st_size / 1048576:.0f} MB)")
                archive.write(source, arcname)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    size = destination.stat().st_size

    print()
    print(f"archive : {destination}")
    print(f"size    : {size / 1048576:.0f} MB ({size} bytes)")
    print(f"sha256  : {digest}")
    print()
    print("Pin these in app/transcription/cuda_pack.py:")
    print(f'    PACK_VERSION = "{pack_version()}"')
    print(f"    PACK_SIZE = {size}")
    print(f'    PACK_SHA256 = "{digest}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
