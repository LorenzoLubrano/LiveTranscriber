# LiveTranscriber — Implementation Plan

Real-time, fully local transcription of PC audio and microphone on Windows 10/11.
No cloud STT, no API keys, no telemetry, no account. Offline after model download.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done & verified

---

## 0. Target environment (measured 2026-09-18)

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 (64-bit) |
| CPU | AMD Ryzen 7 260 — 8 cores / 16 threads, 3.80 GHz |
| RAM | 15.3 GB |
| GPU | NVIDIA GeForce RTX 5050 Laptop (8 GB, **Blackwell sm_120**), driver 596.13, CUDA 13.2 |
| iGPU | AMD Radeon 780M (not used for inference) |
| Python | 3.13.13 (`C:\Users\lollo\AppData\Local\Programs\Python\Python313`) |
| Git | 2.52.0 |
| FFmpeg | **not installed** — not required (see §2) |
| Disk free | 329 GB |

### Decisions taken with the user

1. **Python 3.13**, not the 3.11 in the spec. Every dependency ships a native
   cp313 wheel — verified by a full `pip install --dry-run` resolution:
   PySide6 6.11.2, ctranslate2 4.8.2, PyAudioWPatch 0.2.12.8, onnxruntime 1.30.0,
   av 18.1.0, soxr 1.1.0, pyinstaller 6.22.3. Nothing needs compiling, so the
   reason to install a second interpreter disappeared.
2. **GPU runtime libraries installed now** (`nvidia-cublas-cu12`,
   `nvidia-cudnn-cu12`) so the CUDA path is tested on real hardware, not assumed.
3. **Portable folder** is the primary release artifact (PyInstaller one-dir).
   A single `.exe` extracts to `%TEMP%` on every launch and is a known source of
   native-DLL failures with Qt + CTranslate2; the spec explicitly allows a folder.

### Blackwell (sm_120) constraint — drives the whole GPU design

CTranslate2 release history is explicit about this hardware:

- **4.6.2** — *"Disable INT8 for sm120 — Blackwell GPUs."*
- **4.6.3** — adds CUDA 12.8 support; makes cuDNN an optional dependency.
- **4.8.2** — sm_120 runs via PTX JIT.

Consequences, encoded as hard rules in `app/transcription/engine.py`:

- GPU compute type is **`float16`**. `int8` / `int8_float16` must never be
  offered for a CUDA device on sm_120 — it fails with `CUBLAS_STATUS_NOT_SUPPORTED`.
- CPU compute type is **`int8`**.
- First CUDA inference pays a one-off PTX JIT compile cost. Warm up the model
  once at load so the first user-visible segment is not the slow one.
- Any CUDA/cuBLAS/OOM error must degrade to CPU rather than crash.

### Deviations from the spec, and why

| Spec said | Using | Reason |
|---|---|---|
| Python 3.11 | Python 3.13 | All wheels native on 3.13; avoids a second interpreter. |
| "a reliable resampling library" | `soxr` | `libsoxr` bindings: VHQ sinc resampling, ~µs per chunk, no FFmpeg binary. |
| FFmpeg | not installed | `faster-whisper` decodes through PyAV (bundled FFmpeg libs). We feed it NumPy arrays directly, so no decoding happens at all. |
| PyInstaller single `.exe` | one-dir portable | Reliability over artificial single-file packaging. |

---

## Milestone 1 — Environment & scaffold

- [x] Project tree under `%USERPROFILE%\Desktop\LiveTranscriber`
- [x] `.venv` on Python 3.13
- [x] Core dependencies installed and importable
- [x] `pyproject.toml`, `.gitignore`, `LICENSE` (MIT), `README.md`
- [x] `git init` + first commit
- [x] `app/utils/paths.py` — application data directories
- [x] `app/utils/logging_setup.py` — rotating log, no transcript content

**Done when:** `python -c "import PySide6, faster_whisper, pyaudiowpatch, soxr"`
succeeds inside the venv and the repo has its first commit.

**Verified 2026-09-18.** All 9 core imports resolve; WASAPI host API reports 6
devices; repo initialised with 8 logical commits; `ruff check` clean.

---

## Milestone 2 — Audio capture (the foundation; do not rush)

Everything above the audio layer is worthless if this is not solid.

- [x] `app/audio/devices.py` — WASAPI enumeration
  - host-API-aware; loopback devices matched to their render endpoint
  - stable device identity across replug (name + host API + channel signature)
  - de-duplication of the repeated WASAPI entries PortAudio reports
  - default input / default output resolution
- [x] `app/audio/resampler.py` — `soxr` wrapper, any rate → 16 kHz mono float32
  - stateful streaming resampler (no clicks at chunk boundaries)
  - stereo/multi-channel → mono downmix
- [x] `app/audio/ring_buffer.py` — bounded lock-free-ish float32 ring buffer
  - fixed capacity, overwrite-oldest, explicit overflow counter
- [x] `app/audio/capture.py` — one `CaptureStream` per source
  - callback does **only** `bytes → ring buffer` + a level meter update
  - no allocation, no resampling, no I/O in the callback
  - device-lost detection → signal, never an exception in the callback
- [x] `app/audio/loopback.py` — WASAPI loopback for PC audio
- [x] `app/audio/microphone.py` — WASAPI/MME microphone
- [x] `app/audio/recorder.py` — streaming WAV writer
  - writes incrementally to disk, flushes periodically (crash-safe)
  - `pc_audio.wav`, `microphone.wav`, and a mixed track when both are active
- [x] `tests/` — ring buffer, resampler, WAV writer, device de-duplication
- [x] `scripts/audio_test.py` — real-hardware diagnostic (spec §22)

**Done when:** `scripts/audio_test.py` records 10 s of real PC audio and 10 s of
real microphone, writes valid WAVs, and reports non-silent RMS for both.

**Verified 2026-09-18 on the target machine.**

`scripts/audio_test.py --both --seconds 10`:

| | PC (loopback) | Microphone |
|---|---|---|
| device | Speakers (Realtek) 48 kHz 2ch | Mic Array (Realtek) 48 kHz 2ch |
| callbacks | 475 | 470 |
| captured | 10.13 s | 10.03 s |
| **dropped frames** | **0 (0.00%)** | **0 (0.00%)** |
| **overflows** | **0** | **0** |
| RMS | −11.2 dBFS | −32.5 dBFS |

Three valid WAVs written. Frequency check on the output files: a 440 Hz tone
played through the speakers came back as **439.9 Hz** in both `pc_audio.wav` and
`mixed.wav` — the loopback chain reproduces what Windows plays, at the right
rate, with no drift.

Automated coverage: 123 tests, all passing, `ruff` clean.

---

## Milestone 3 — Whisper engine (offline)

- [ ] `app/transcription/models.py` — catalogue, sizes, download + progress, offline check
- [ ] `app/transcription/engine.py` — device/compute selection, warm-up, CPU fallback
- [ ] `scripts/whisper_test.py` — WAV → transcript, CPU and GPU, timed
- [ ] Benchmark: real-time factor for tiny/base/small/medium on this machine

## Milestone 4 — Real-time pipeline

- [ ] `app/transcription/vad.py` — Silero VAD (ONNX), with an energy-gate fallback
- [ ] `app/transcription/streaming.py` — windowing, overlap, confirm/provisional
- [ ] Deduplication / reconciliation across overlapping windows
- [ ] Silence handling (no hallucinated text on quiet audio)
- [ ] `app/sessions/transcript.py` — segment store, search, export source of truth
- [ ] Latency measured end-to-end

## Milestone 5 — GUI (PySide6; `frontend-design` skill here)

## Milestone 6 — Sessions, export, autosave, recovery

## Milestone 7 — Robustness (2–4 h soak test, RAM/handle/VRAM checks)

## Milestone 8 — Packaging (`scripts/build_windows.ps1`, portable folder)

---

## Architecture

```
┌──────────────┐   ┌──────────────┐
│ loopback (PC)│   │ microphone   │      WASAPI callbacks — minimal work
└──────┬───────┘   └──────┬───────┘
       │ int16/float32 native rate, N channels
       ▼                  ▼
┌─────────────────────────────────┐
│ RingBuffer (bounded, per source)│     overflow counted, never unbounded
└──────┬──────────────────┬───────┘
       │                  └────────────► WAV writer (incremental, crash-safe)
       ▼
┌─────────────────────────────────┐
│ Resampler  → 16 kHz mono float32│     soxr, stateful
└──────┬──────────────────────────┘
       ▼
┌─────────────────────────────────┐
│ VAD → segmentation              │     silence never reaches Whisper
└──────┬──────────────────────────┘
       ▼
┌─────────────────────────────────┐
│ Whisper worker thread (per src) │     never on the GUI thread
└──────┬──────────────────────────┘
       ▼
┌─────────────────────────────────┐
│ Transcript manager              │     confirmed / provisional, dedup
└──────┬──────────────────────────┘
       │ Qt signals (queued)
       ▼
┌─────────────────────────────────┐
│ GUI                             │
└─────────────────────────────────┘
```

**Threading:** audio callbacks run on PortAudio threads and only touch ring
buffers. One pipeline thread per source does resample → VAD → segment. One
Whisper worker thread per source consumes segments from a bounded queue. The GUI
thread receives queued Qt signals only. Nothing blocking ever runs on the GUI thread.

**Memory:** audio in RAM is bounded by ring-buffer capacity (seconds, not hours).
Full audio lives on disk in the WAV files. Segment queues are bounded and drop
policy is explicit and logged.
