# Architecture

How LiveTranscriber is put together, and why.

Audience: developers working on this codebase.

---

## The problem shape

Three constraints drive nearly every decision:

1. **The GUI must never freeze.** Whisper inference takes seconds. Audio
   callbacks run on a realtime-ish thread. Neither may touch the Qt event loop.
2. **Memory must be flat.** A four-hour lecture is ~2.3 GB of raw 48 kHz stereo
   audio. None of it may accumulate in RAM.
3. **Audio must not be lost.** Not to a stalled consumer, not to a crash, not to
   a Bluetooth dropout.

## Data flow

```text
┌──────────────┐   ┌──────────────┐
│ loopback (PC)│   │ microphone   │   PortAudio callback threads
└──────┬───────┘   └──────┬───────┘   (minimal work, cannot raise)
       │ float32, device rate, N channels
       ▼                  ▼
┌─────────────────────────────────┐
│ RingBuffer (bounded, per source)│   fixed allocation; overflow counted
└──────┬──────────────────┬───────┘
       │                  └──────────► WavWriter (incremental, crash-safe)
       ▼
┌─────────────────────────────────┐
│ StreamResampler → 16 kHz mono   │   soxr, stateful across chunks
└──────┬──────────────────────────┘
       ▼
┌─────────────────────────────────┐
│ VAD → segmentation              │   silence never reaches Whisper
└──────┬──────────────────────────┘   [milestone 4]
       ▼
┌─────────────────────────────────┐
│ Whisper worker (thread/source)  │   bounded queue in front
└──────┬──────────────────────────┘   [milestone 3-4]
       ▼
┌─────────────────────────────────┐
│ Transcript manager              │   confirmed / provisional, dedup
└──────┬──────────────────────────┘   [milestone 4]
       │ queued Qt signals
       ▼
┌─────────────────────────────────┐
│ GUI (PySide6)                   │   [milestone 5]
└─────────────────────────────────┘
```

Implemented today: everything down to and including the WAV writers.

## Threading model

| Thread | Owns | Rule |
|---|---|---|
| PortAudio callback (one per source) | `RingBuffer.write` | No allocation, no I/O, no locks held across work, **never raises** |
| Pipeline (one per source) | resample → VAD → segment | May block; drains the ring buffer |
| Whisper worker (one per source) | inference | Bounded input queue; drop policy is explicit and logged |
| `CaptureWatchdog` | stall detection | Polls; reports device loss through a callback |
| Qt main thread | GUI only | Receives queued signals; never runs inference or file I/O |

An exception escaping a PortAudio callback terminates the stream, so
`CaptureStream._callback` wraps its whole body and logs rather than propagating.

## Why each library

| Choice | Alternative rejected | Reason |
|---|---|---|
| **WASAPI loopback** (PyAudioWPatch) | Microphone pointed at speakers; Stereo Mix | Taps the render endpoint directly: full quality, no room noise, works on any endpoint including Bluetooth. Stereo Mix does not exist on most modern hardware. |
| **soxr** | `scipy.signal.resample_poly`; FFmpeg binary | Genuine streaming API carrying filter state between calls, so chunk boundaries produce no clicks. scipy restarts its filter every call. No external binary to bundle. |
| **faster-whisper / CTranslate2** | openai-whisper; whisper.cpp | Several times faster than the reference implementation, INT8 on CPU, mature Python API, MIT licensed. |
| **PySide6** | Electron; tkinter; web UI | Native desktop, LGPL (usable from MIT code), proper high-DPI and dark-mode support. An Electron app would need a bundled browser for no benefit. |
| **PyInstaller one-dir** | one-file `.exe` | One-file extracts to `%TEMP%` on every launch — slow, and a known source of native-DLL failures with Qt + CTranslate2. One-dir also keeps LGPL libraries replaceable. |

## Memory strategy

- **Audio in RAM is bounded by ring-buffer capacity** (30 s per source by
  default), not by recording length. The backing array is allocated once in
  `RingBuffer.__init__` and never reallocated; `test_memory_is_bounded_over_a_long_run`
  asserts exactly that.
- **Full audio lives on disk**, written incrementally as it arrives.
- **Overflow is visible, not silent.** When a consumer falls behind, the oldest
  frames are dropped and counted in `CaptureStats.frames_dropped`.

## Crash safety

`wave` writes correct chunk sizes only on `close()`. A process killed mid-session
would leave a header claiming zero bytes — a multi-hour recording that reads as
empty.

`WavWriter` owns its header instead, rewriting the RIFF and `data` size fields
and flushing every `flush_interval_s` (5 s default). A crash costs at most that
interval. `test_header_is_valid_before_close` reads a complete WAV back from a
writer that was deliberately never closed.

## Device identity

PortAudio indices are positional and shift whenever a device is plugged,
unplugged, or made default. Persisting an index would silently record from the
wrong device after a reboot.

Devices therefore carry a stable `key`:

```text
"Windows WASAPI|microphone|Microphone Array (Realtek(R) Audio)|2"
 host API      |kind      |name                               |channels
```

Settings persist the key; `find_by_key()` resolves it against the current
enumeration at start time, and reports absence rather than guessing.

### Why WASAPI only

Measured on the development machine, the same microphone appears three times:

| Host API | Name | Rate |
|---|---|---|
| MME | `Microphone Array (Realtek(R) Au` | 44100 |
| DirectSound | `Microphone Array (Realtek(R) Audio)` | 44100 |
| **WASAPI** | `Microphone Array (Realtek(R) Audio)` | **48000** |

MME truncates names at 31 characters. Both legacy APIs report 44.1 kHz for
hardware whose actual mix format is 48 kHz. Only WASAPI exposes loopback devices.
So WASAPI is the source of truth and the others are a last-resort fallback.

## Module map

```text
app/
├── audio/
│   ├── devices.py      Enumeration, stable keys, de-duplication
│   ├── ring_buffer.py  Bounded circular float32 buffer
│   ├── resampler.py    dtype → mono → 16 kHz, streaming
│   ├── capture.py      CaptureStream + CaptureWatchdog
│   ├── loopback.py     PC-audio source resolution
│   ├── microphone.py   Microphone source resolution
│   └── recorder.py     WavWriter, MixRecorder
├── transcription/      [milestone 3-4]
├── sessions/           [milestone 6]
├── export/             [milestone 6]
├── ui/                 [milestone 5]
├── config/             [milestone 6]
└── utils/
    ├── paths.py        Application directories, safe names
    └── logging_setup.py Rotating log, no transcript content
```

## GPU handling

Hardware detection picks the device and compute type; the user may override with
Automatic / CPU / GPU.

| Device | Compute type |
|---|---|
| CPU | `int8` |
| NVIDIA (generic) | `float16` |
| **NVIDIA Blackwell, sm_120 (RTX 50xx)** | `float16` **only** |

CTranslate2 4.6.2 disabled INT8 on sm_120; forcing it there fails with
`CUBLAS_STATUS_NOT_SUPPORTED`. sm_120 also runs through PTX JIT, so the model is
warmed up once at load to keep that cost off the first user-visible segment.

Any CUDA failure — driver mismatch, out of memory, missing cuBLAS — degrades to
CPU with a plain-language message rather than crashing.

## Testing

| Layer | How |
|---|---|
| Ring buffer, resampler, WAV, paths | Pure unit tests, no hardware |
| Device logic | Synthetic devices; real enumeration behind `-m hardware` |
| Capture logic, device loss, pause | Callback driven directly by the test |
| Capture end-to-end | Plays a 440 Hz tone and asserts loopback returns 440 Hz |

The end-to-end test needs no human and no audio playing — it generates its own
signal, which is what makes it runnable in a normal test run.
