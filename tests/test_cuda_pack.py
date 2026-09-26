"""Downloading executable code, safely.

These tests never touch the network: the download is driven through a fake
opener, because what matters here is what the code does with the bytes it is
given — especially the bytes it should refuse.
"""

from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from app.transcription import cuda_pack


def complete_pack(extra: dict[str, bytes] | None = None) -> bytes:
    """An archive shaped like the real one: both libraries present."""
    entries = dict.fromkeys(cuda_pack.REQUIRED, b"fake dll")
    entries.update(extra or {})
    return make_zip(entries)


def make_zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def pack_home(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda_pack, "pack_dir", lambda: tmp_path / "cuda")
    return tmp_path / "cuda"


def serve(monkeypatch, payload: bytes, digest: str | None = None):
    """Point the downloader at these bytes, pinned to this digest."""
    monkeypatch.setattr(
        cuda_pack, "PACK_SHA256", digest or hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(cuda_pack, "PACK_SIZE", len(payload))
    monkeypatch.setattr(cuda_pack, "_fetch", lambda url, on_chunk, cancel: payload)


# -- installing ------------------------------------------------------------


def test_a_good_pack_is_installed(pack_home, monkeypatch):
    payload = complete_pack()
    serve(monkeypatch, payload)

    cuda_pack.install()

    installed = pack_home / "nvidia" / "cublas" / "bin" / "cublas64_12.dll"
    assert installed.read_bytes() == b"fake dll"
    assert cuda_pack.is_installed()


def test_nothing_is_installed_before_a_download(pack_home):
    assert not cuda_pack.is_installed()


def test_progress_is_reported(pack_home, monkeypatch):
    payload = complete_pack()
    seen: list[tuple[int, int]] = []

    monkeypatch.setattr(cuda_pack, "PACK_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(cuda_pack, "PACK_SIZE", len(payload))

    def fetch(url, on_chunk, cancel):
        on_chunk(len(payload) // 2)
        on_chunk(len(payload))
        return payload

    monkeypatch.setattr(cuda_pack, "_fetch", fetch)
    cuda_pack.install(on_progress=lambda done, total: seen.append((done, total)))

    assert seen and seen[-1][0] == len(payload)
    assert all(total == len(payload) for _, total in seen)


# -- refusing --------------------------------------------------------------


def test_a_tampered_pack_is_refused(pack_home, monkeypatch):
    """The whole reason the hash is pinned in the source.

    Anything that is not exactly the archive this build was made against is
    rejected before a single byte of it reaches the disk as a DLL.
    """
    payload = make_zip({"nvidia/cublas/bin/cublas64_12.dll": b"malicious"})
    serve(monkeypatch, payload, digest="0" * 64)

    with pytest.raises(cuda_pack.CudaPackError) as exc:
        cuda_pack.install()

    assert "verifica" in exc.value.user_message.lower()
    assert not pack_home.exists() or not cuda_pack.is_installed()


def test_a_pack_escaping_its_directory_is_refused(pack_home, monkeypatch):
    """Zip entries are attacker-controlled paths until they are checked."""
    payload = complete_pack({"../../evil.dll": b"nope"})
    serve(monkeypatch, payload)

    with pytest.raises(cuda_pack.CudaPackError):
        cuda_pack.install()

    assert not (pack_home.parent.parent / "evil.dll").exists()


def test_an_absolute_path_is_refused(pack_home, monkeypatch):
    payload = complete_pack({"C:/Windows/System32/evil.dll": b"nope"})
    serve(monkeypatch, payload)

    with pytest.raises(cuda_pack.CudaPackError):
        cuda_pack.install()


def test_a_pack_without_the_expected_library_is_refused(pack_home, monkeypatch):
    """A valid archive of the wrong thing is still the wrong thing."""
    payload = make_zip({"nvidia/cublas/bin/readme.txt": b"hello"})
    serve(monkeypatch, payload)

    with pytest.raises(cuda_pack.CudaPackError):
        cuda_pack.install()

    assert not cuda_pack.is_installed()


def test_cancelling_leaves_nothing_behind(pack_home, monkeypatch):
    monkeypatch.setattr(
        cuda_pack, "_fetch",
        lambda url, on_chunk, cancel: (_ for _ in ()).throw(cuda_pack.Cancelled()),
    )
    with pytest.raises(cuda_pack.Cancelled):
        cuda_pack.install()
    assert not cuda_pack.is_installed()


# -- removing --------------------------------------------------------------


def test_the_pack_can_be_removed(pack_home, monkeypatch):
    payload = complete_pack()
    serve(monkeypatch, payload)
    cuda_pack.install()
    assert cuda_pack.is_installed()

    cuda_pack.remove()
    assert not cuda_pack.is_installed()
