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

- [x] `app/transcription/hardware.py` — GPU detection, compute-type selection
- [x] `app/transcription/cuda_setup.py` — make pip-installed CUDA DLLs loadable
- [x] `app/transcription/models.py` — catalogue, sizes, download + progress, offline check
- [x] `app/transcription/engine.py` — device/compute selection, warm-up, CPU fallback
- [x] `scripts/whisper_test.py` — WAV → transcript, CPU and GPU, timed
- [x] Benchmark: real-time factor on this machine

**Verified 2026-09-18 on the target machine.**

`scripts/whisper_test.py --benchmark`, on an 11.0 s Italian speech sample:

| model | device | load | inference | RTF | words |
|---|---|---|---|---|---|
| tiny  | cuda/float16 | 0.61 s | 0.26 s | **41.7x** | 75% |
| tiny  | cpu/int8     | 0.44 s | 0.29 s | **38.1x** | 75% |
| small | cuda/float16 | 0.79 s | 0.39 s | **28.4x** | 85% |
| small | cpu/int8     | 1.90 s | 1.47 s | **7.5x**  | 85% |

Real-time needs RTF > 1. Even **CPU + small reaches 7.5x**, so this machine
runs the recommended model live without a GPU; the GPU gives ~4x more headroom.

The test script synthesises its own speech through the Windows speech engine
(Italian and English voices are installed), so it needs no recording.

### Two problems found and fixed here

1. **`cublas64_12.dll is not found`.** The pip CUDA wheels put their DLLs in
   `site-packages/nvidia/*/bin`, and since Python 3.8 Windows ignores `PATH`
   for extension-module dependencies. `cuda_setup.py` registers those
   directories with `os.add_dll_directory()` before any CUDA work. Without it
   the GPU path cannot work at all on Windows.
2. **A broken GPU stalled instead of failing.** `WhisperModel(device="cuda")`
   constructs without touching cuBLAS, so a missing DLL only surfaced later.
   Warm-up failures on GPU now re-raise and trigger the CPU fallback at load
   time, rather than hanging once the user has pressed Start.

### Useful discovery for Milestone 4

faster-whisper **bundles `silero_vad_v6.onnx` (1.2 MB)** and exposes
`get_speech_timestamps`, `VadOptions` and `collect_chunks`. Milestone 4 reuses
it instead of downloading a separate VAD model — one less download, and it
works offline out of the box.

## Milestone 4 — Real-time pipeline

- [x] `app/transcription/vad.py` — Silero VAD (reuses faster-whisper's bundled model)
- [x] `app/transcription/streaming.py` — LocalAgreement-2, confirm/provisional
- [x] Deduplication / reconciliation across overlapping windows
- [x] Silence handling (no hallucinated text on quiet audio)
- [x] `app/sessions/transcript.py` — segment store, search, export source of truth
- [x] Latency measured end-to-end
- [x] `scripts/realtime_test.py` — paced diagnostic

### How streaming works

**LocalAgreement-2**: transcribe a growing buffer repeatedly, confirm only what
two consecutive runs agreed on. Observed doing its job on a real run:

```
provisional: Nell'imite adiabatico la variazione e'       <- wrong, not confirmed
provisional: Nel limite a diabatico la variazione e' ...   <- next run disagrees
CONFIRMED  : Nel limite a diabatico la variazione e' ...   <- only now committed
```

This addresses each failure the spec lists: a word at the buffer edge is never
confirmed while truncated; confirmed audio is dropped from the buffer so it
cannot be emitted twice; the last confirmed words are passed back as
`initial_prompt` for context; and VAD means Whisper is never called on silence.

**Verified 2026-09-18**, fed at wall-clock speed, capture and inference on
separate threads exactly as the app is built:

| model | device | inference | headroom | avg latency | max | words | duplicates |
|---|---|---|---|---|---|---|---|
| tiny | GPU | 6% | 16.2x | 2.40 s | 4.73 s | 83% | none |
| tiny | CPU | 14% | 7.4x | 3.02 s | 5.40 s | 84% | none |
| **small** | **GPU** | **13%** | **7.7x** | **2.38 s** | **3.03 s** | **94%** | none |
| small | CPU | 70% | 1.4x | 4.12 s | 9.63 s | 94% | none |
| medium | GPU | 28% | 3.6x | 2.45 s | 2.93 s | 95% | none |
| medium | CPU | 115% | 0.9x | 14.6 s | 23.8 s | — | cannot keep up |

Streaming output is a **100% word match with transcribing the same audio in one
go** — the windowing loses nothing. Silence: **0 inferences run**, no text.

### The correction this milestone forced

The Milestone 3 benchmark measured *one-shot* transcription and reported
small/CPU at RTF 7.5x. That number does not carry over to live use:
LocalAgreement re-transcribes a growing buffer, so each second of audio passes
through Whisper roughly **four or five times**. The same model streaming uses
70% of the audio time — 1.4x headroom, not 7.5x — and medium/CPU at 115% cannot
keep up at all.

So the quality presets are hardware-aware (`models.recommend_model`), and a
one-shot benchmark must never be used to choose a live model.

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
