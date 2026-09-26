"""The optional NVIDIA support pack, fetched on request.

Transcribing on an NVIDIA GPU needs cuBLAS, which is 736 MB of libraries NVIDIA
publishes but nobody else can be expected to install. Shipping it in the
application means a separate two-gigabyte download that is useless to everyone
without such a card; leaving it out means the app runs several times slower on
the machines that could be fastest.

So it is a download of its own, made when the user asks for it: one 127 MB
application for everybody, and a button for the people who have the hardware.

Downloading executable code deserves care, and this module is where that care
lives:

* the archive's SHA-256 is pinned in this file at build time, so only the exact
  bytes this version was built against are ever unpacked;
* entry paths are checked before extraction, because a zip can name
  ``..\\..\\Windows\\System32`` as happily as anything else;
* the result is assembled in a temporary directory and moved into place, so an
  interrupted download cannot leave a half-installed runtime that looks whole.

After installation the app is offline again: nothing here is re-checked, phoned
home, or updated behind the user's back.
"""

from __future__ import annotations

import hashlib
import io
import logging
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

#: Published alongside the application, from NVIDIA's own wheel, unmodified.
PACK_VERSION = "12.9.2.10"
PACK_URL = (
    "https://github.com/LorenzoLubrano/LiveTranscriber/releases/download/"
    "cuda-runtime-12.9/livetranscriber-cuda-12.9.2.10-windows-x64.zip"
)
PACK_SIZE = 553096949
PACK_SHA256 = "6736d4d74c6c573b869ae64b07b96a07d37d3d615ea0df1feeb7ae38e4b50c87"

#: What has to be present for the installation to count as complete.
REQUIRED = ("nvidia/cublas/bin/cublas64_12.dll", "nvidia/cublas/bin/cublasLt64_12.dll")

_CHUNK = 1024 * 512


class CudaPackError(RuntimeError):
    """Something went wrong, with a message safe to show a user."""

    def __init__(self, user_message: str, technical: str = "") -> None:
        super().__init__(technical or user_message)
        self.user_message = user_message
        self.technical = technical or user_message


class Cancelled(Exception):
    """The user stopped the download."""


# --------------------------------------------------------------------------
# Where it lives
# --------------------------------------------------------------------------


def pack_dir() -> Path:
    """Beside the models, not beside the executable.

    The application folder may be read-only, on a USB stick, or replaced
    wholesale by the next version; per-user data survives all three.
    """
    from app.utils.paths import app_data_dir

    return app_data_dir() / "cuda"


def is_installed() -> bool:
    return all((pack_dir() / name).exists() for name in REQUIRED)


def size_label() -> str:
    return f"{PACK_SIZE / 1_048_576:.0f} MB"


def remove() -> None:
    """Delete the pack. Used when someone wants the disk space back."""
    shutil.rmtree(pack_dir(), ignore_errors=True)


# --------------------------------------------------------------------------
# Getting it
# --------------------------------------------------------------------------


def _fetch(
    url: str,
    on_chunk: Callable[[int], None],
    cancel: Callable[[], bool] | None,
) -> bytes:
    """Download to memory, reporting bytes received. Raises Cancelled."""
    request = Request(url, headers={"User-Agent": "LiveTranscriber"})
    received = bytearray()
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URL
        while True:
            if cancel is not None and cancel():
                raise Cancelled()
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            received += chunk
            on_chunk(len(received))
    return bytes(received)


def _safe_members(archive: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    """Entries that stay inside ``destination``.

    Zip entry names come from the archive, which is to say from outside this
    program. An entry called ``../../evil.dll`` extracts exactly where it says
    unless someone checks, and this pack's whole job is to place DLLs that the
    app will later load.
    """
    root = destination.resolve()
    members = []
    for member in archive.infolist():
        if member.is_dir():
            continue
        name = member.filename.replace("\\", "/")
        target = (destination / name).resolve()
        if not target.is_relative_to(root):
            raise CudaPackError(
                "Il pacchetto scaricato non è valido e non è stato installato.",
                f"unsafe archive member: {member.filename!r}",
            )
        members.append(member)
    return members


def install(
    on_progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Path:
    """Download, verify and unpack the NVIDIA support pack.

    Blocking and slow: call it from a worker thread. Returns the directory the
    libraries were installed into.
    """
    report = (lambda done: on_progress(done, PACK_SIZE)) if on_progress else (lambda _: None)

    logger.info("Downloading the NVIDIA support pack (%s)", size_label())
    try:
        payload = _fetch(PACK_URL, report, cancel)
    except Cancelled:
        raise
    except Exception as exc:
        raise CudaPackError(
            "Download non riuscito. Controlla la connessione e riprova.",
            repr(exc),
        ) from exc

    digest = hashlib.sha256(payload).hexdigest()
    if digest != PACK_SHA256:
        raise CudaPackError(
            "La verifica del pacchetto scaricato non è riuscita: il file non "
            "corrisponde a quello atteso e non è stato installato.",
            f"sha256 {digest} != {PACK_SHA256}",
        )

    destination = pack_dir()
    # Staged beside the destination, never in the system temp: the final step is
    # a rename, and a rename across volumes is not atomic — on Windows it is not
    # even possible.
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="cuda-pack-", dir=destination.parent))
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = _safe_members(archive, staging)
            archive.extractall(staging, members=members)

        if not all((staging / name).exists() for name in REQUIRED):
            raise CudaPackError(
                "Il pacchetto scaricato non contiene le librerie attese.",
                f"missing one of {REQUIRED}",
            )

        # Replace whole, never merge: a half-old, half-new set of libraries is
        # the one state that would be hard to diagnose.
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        staging.replace(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info("NVIDIA support pack installed in %s", destination)
    return destination
