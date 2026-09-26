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
| Python | 3.13.13 (per-user install) |
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

## Milestone 5 — GUI (PySide6)

- [x] `app/ui/theme.py` — design tokens, light/dark/system, Qt stylesheet
- [x] `app/ui/widgets/level_meter.py` — segmented meter, peak hold, record dot
- [x] `app/ui/widgets/transcript_view.py` — the reading surface
- [x] `app/ui/main_window.py` — idle/recording shape change, transport, search
- [x] `app/ui/settings_window.py` — general / advanced / privacy
- [x] `app/ui/download_dialog.py` — model download off the GUI thread
- [x] `app/config/settings.py` — QSettings persistence
- [x] `app/sessions/session.py` — the controller wiring audio to transcription
- [x] `app/main.py` — entry point

### The design in one line each

* **Two materials.** The control strip is an instrument: compact, dense, glanced
  at from across a desk. The transcript is a page: a reading serif at a generous
  line height, because a two-hour lecture is a document, not a log.
* **The window changes shape.** Idle is setup-shaped; recording collapses the
  pickers to one summary line and gives everything to the transcript.
* **One saturated colour.** The level meter, green through amber to red as on any
  hardware meter, plus the red record dot. Nothing else is coloured, so the two
  things that mean "it is working" are the only things competing for attention.
* **Provisional text is visibly unsettled** — dimmed and italic, resolving into
  full-contrast upright text when confirmed. This is the one place boldness is
  spent, because watching it settle explains the product better than any label.

### Verified 2026-09-18, end to end through the real GUI

Press Start, speak to the speakers, watch text appear, pause, resume, stop:

```
premo Avvia registrazione
sessione attiva -> .../Recordings/2026-09-18_10-29_Lezione QCED
testo dopo la prima frase: 14 parole      <- text during recording
metto in PAUSA / stato = paused / riprendo
premo Termina
  audio.wav  828.7 KB   26.5s @ 16000 Hz
  segmenti: 6   parole: 29
```

### Three bugs found only by running it

1. **The GUI showed nothing while recording worked perfectly.**
   `RecordingSession.__init__` did `self.transcript = transcript or Transcript()`.
   `Transcript.__bool__` means "has segments", so an **empty transcript is
   falsy** and the caller's object was silently replaced. The session filled its
   own copy — the log said "6 segments" while the window sat empty. Now tested
   for explicitly.
2. **Double spaces at every join.** The provisional anchor sat *after* the
   separating space, so clearing provisional text left the space and the next
   confirmed chunk added another.
3. **A band of window colour across every panel**, because Qt's `QWidget` rule
   paints labels too; and the level meter collapsed to a few pixels because an
   Expanding child inside a Preferred parent has no width to claim.

317 tests passing, `ruff` clean.


## Milestone 6 — Sessions, export, autosave, recovery

- [x] `app/export/base.py` — timestamps, segment grouping, cue wrapping
- [x] `app/export/txt.py` / `srt.py` / `vtt.py` / `json_export.py`
- [x] `app/export/__init__.py` — format registry, `export_all`
- [x] `app/sessions/autosave.py` — periodic crash-safe save
- [x] `app/sessions/recovery.py` — find and rescue interrupted recordings
- [x] `app/ui/recovery_dialog.py` — the Recupera / Ignora prompt
- [x] Export button in the main window

### Why exports regroup

Streaming confirms text in whatever fragments two passes agreed on, so one
spoken sentence often lands as three segments. That is the right record of
*what was confirmed when*, and the wrong shape for a document — so each format
regroups with its own rules. Paragraphs merge freely and break on real pauses;
subtitle cues cap at ~84 characters and 7 seconds, because a cue has to be
readable before it disappears.

Cue lines are balanced rather than greedily filled: filling to the full width
left stubs like `Consideriamo adesso la miltoniana del / sistema, dove il`.

`transcript.json` is deliberately **not** grouped. It is the archival form —
TXT, SRT and VTT are all derivable from it, and none of them from each other.

### Crash safety

`session.json` carries `completed: false` for the whole recording and is only
set true on a clean stop. A session that died is therefore exactly a session
whose record still says false — that is the entire recovery signal.

Both files are written to a temporary sibling and moved into place, so an
interrupted save leaves the previous good file rather than half of a new one.
The transcript's revision counter means idle ticks write nothing.

### Verified 2026-09-18, end to end

Real recording, then a simulated crash:

```
1. REGISTRAZIONE REALE CON EXPORT
   durante la registrazione, completed = False
   audio.wav 592 KB · session.json · transcript.{json,srt,txt,vtt}
   dopo lo stop, completed = True

2. CRASH SIMULATO E RECUPERO
   simulato: 4 segmenti salvati, nessun completed
   sessioni interrotte trovate: 1
     -> Lezione interrotta · 00:00:39 · 4 segmenti · 39 KB di audio
   recuperati 4 segmenti; export scritti; non viene piu' offerta
```

A previous completed recording next to the crashed one is provably untouched,
and *Ignora* stops the prompt without deleting anything.

434 tests passing (24 hardware tests deselected), `ruff` clean.


## Milestone 7 — Robustness

- [x] `scripts/soak_test.py` — long-run harness, accelerated or real time
- [x] RAM, handle, thread and VRAM growth measured per component
- [x] `app/sessions/keepup.py` — detect and mitigate a machine that cannot keep up
- [x] `app/transcription/calibration.py` — measure this machine instead of guessing
- [ ] 2 h real-time soak (thermal behaviour) — **not yet completed**

### What the accelerated soak found (4 simulated hours per component)

| | initial | typical | growth per hour |
|---|---|---|---|
| Transcript store | 36 MB | 39 MB | +1.1 MB |
| Qt document | 56 MB | 60 MB | +1.0 MB |
| Full pipeline | 655 MB | 783 MB (plateau) | flat |
| Handles | 304 | 336 | +0.1 |
| VRAM | 1105 MB | 1110 MB | +12 MB |

No leak. The Qt document reached 291k characters with append time *falling* from
0.08 ms to 0.03 ms, so the suspected quadratic relayout does not happen. Autosave
went from 3.2 ms to 12.6 ms, which is inherent to rewriting the whole file and
still irrelevant at that scale.

Two findings about the test itself, both worth recording:

* **The first version was green and measured nothing.** It fed a tone plus noise,
  which sounds speech-shaped but is not speech: Silero rejected every block, so
  Whisper never ran and memory was flat because nothing happened. It now uses real
  synthesised speech, and the report *fails* a run that produced no transcription.
* **A two-point slope lied about memory.** Windows trims working sets under
  pressure — 784 MB to 131 MB in one case — which two points reported as
  "−350 MB/hour of growth". Replaced with a least-squares fit plus plateau and
  peak.

### The failure this milestone was really about

A model too slow for the machine used to degrade silently. Measured with `small`
on this CPU, paced in real time: **21 s mean latency at 60 s of audio, 31 s at
90 s, peaks near 90 s** — and nothing said why. The ring buffer was also only
30 s deep while being drained solely between inference passes, and one pass over
the full 28 s streaming buffer measured 19.3 s with `large-v3`: a margin of 1.5x
that a slower CPU erases, after which recorded audio is lost silently.

Both are now handled — see the keep-up guard and the 60 s ring — and the app
measures the machine so it does not choose an impossible model in the first place.


## Milestone 8 — Packaging

- [x] `scripts/LiveTranscriber.spec` — one-dir bundle, ~45 unused Qt modules excluded
- [x] `scripts/build_windows.ps1` — tests, build, licences, archive
- [x] One download for everybody (329 MB, 128 MB zipped) plus an optional
      NVIDIA pack the app fetches on request (527 MB)
- [x] A pre-packaged NVIDIA variant for offline machines (1065 MB, 655 MB zipped)
- [x] `app/selftest.py` — `--selftest` from the packaged build
- [x] Both packaged builds verified on this machine
- [x] GitHub Actions running lint plus the non-hardware suite on `windows-latest`

Verified from the packaged executables, not from source:

```
LiveTranscriber\LiveTranscriber.exe --selftest
  Elaborazione : CPU (6 thread) - RTX 5050 rilevata, ma questa versione non
                 include le librerie CUDA. Scarica la versione per NVIDIA.
  Velocita'    : small: 1.1x il tempo reale (troppo lento)   <- correctly flagged

LiveTranscriber-GPU\LiveTranscriber.exe --selftest
  Elaborazione : GPU NVIDIA RTX 5050 Laptop, float16
  Velocita'    : small: 6.4x il tempo reale (va bene)
```

### What the GPU actually needs

The NVIDIA build carried 2335 MB, of which 1093 MB was the cuDNN wheel. Listing
the libraries actually mapped during GPU transcription — real speech, three
model architectures — gives three files:

| library | size | from |
|---|---|---|
| `cublasLt64_12.dll` | 638 MB | the nvidia-cublas wheel |
| `cublas64_12.dll` | 98 MB | the nvidia-cublas wheel |
| `cudnn64_9.dll` | 0.3 MB | shipped inside CTranslate2 itself |

`ctranslate2.dll` names no cuDNN library at all in its imports. So cuDNN came
out of the bundle and out of the `gpu` extra, and the 736 MB that remain became
a pack the app downloads on request.

Verified in isolation, with the downloaded pack temporarily moved aside:

| build | pack | result |
|---|---|---|
| universal | installed | GPU NVIDIA RTX 5050, 9.7x real time |
| universal | absent | CPU, with the message offering the download |
| NVIDIA (trimmed) | absent | GPU NVIDIA RTX 5050, 9.8x real time |

The archives fit well under GitHub's 2 GiB per-asset limit; the build script
checks and warns if a future change pushes one over.

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
