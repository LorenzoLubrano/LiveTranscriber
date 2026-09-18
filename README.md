# LiveTranscriber

Real-time transcription of anything your PC plays, and of your microphone, on
Windows 10/11. Lectures, meetings, videos, calls.

Everything runs **on your machine**. No cloud speech-to-text, no API keys, no
account, no telemetry. After the first model download the app works with the
network switched off.

> **Status: in development.** Milestones 1–2 (environment, audio capture) are
> complete and verified on real hardware. Transcription, GUI and packaging are
> being built — see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

---

## What it does

- **Audio PC** — records what Windows is playing, through WASAPI loopback. It
  taps the audio stream itself, so there is no room noise and no need to point a
  microphone at your speakers. Works with speakers, headphones and Bluetooth.
- **Microphone** — records the input device you choose.
- **PC + microphone together** — both at once, kept separate in the transcript:

  ```text
  00:13:42

  [PC]  Il sistema può essere descritto attraverso questa Hamiltoniana...
  [MIC] Quindi in questo caso siamo nel limite adiabatico?
  [PC]  Esatto, purché la variazione sia sufficientemente lenta...
  ```

  No speaker-identification AI is involved: the app already knows which stream
  each piece of audio arrived on.

## Requirements

| | |
|---|---|
| OS | Windows 10 or 11 (64-bit) — WASAPI loopback is Windows-only |
| RAM | 8 GB minimum, 16 GB recommended |
| Disk | ~2 GB for the app, plus 75 MB–3 GB per Whisper model |
| GPU | Optional. An NVIDIA GPU makes transcription several times faster |
| Python | 3.13 (developers only — end users get a build with Python included) |

## Install (developers)

```powershell
git clone <repo> LiveTranscriber
cd LiveTranscriber

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

### CPU mode

Works everywhere, no extra steps. Transcription runs with `int8` quantisation
across your CPU cores.

### NVIDIA GPU mode

```powershell
pip install -e ".[gpu]"
```

This installs the CUDA runtime libraries (cuBLAS and cuDNN, ~1.5 GB) as Python
packages, so **you do not need to install the CUDA Toolkit**.

If CUDA is missing or misconfigured the app does not crash — it logs the problem
and falls back to CPU.

#### RTX 50-series (Blackwell) note

Blackwell GPUs (compute capability sm_120) need **CTranslate2 ≥ 4.8.2** and a
**CUDA 12.8+** runtime, and only `float16` works — CTranslate2 disables INT8 on
sm_120 (release 4.6.2). The app detects this and chooses the right compute type
on its own; forcing `int8` on such a GPU would fail with
`CUBLAS_STATUS_NOT_SUPPORTED`.

## Checking your audio setup

Before anything else, confirm Windows is giving us what we need:

```powershell
python scripts/audio_test.py --list        # show every device
python scripts/audio_test.py --both        # record 10 s of PC audio + mic
```

The second command writes `pc_audio.wav`, `microphone.wav` and `mixed.wav` to
`scripts/out/` and reports whether each stream carried real signal. Play
something on your PC while it runs.

## Tests

```powershell
pytest                                  # everything
pytest -m "not hardware"                # skip tests needing a sound device
pytest tests/test_capture_hardware.py -v # real capture, end to end
```

The hardware tests are self-contained: they play a 440 Hz tone through your
default output and assert that WASAPI loopback returns that exact frequency, so
they need no audio playing and no human.

## Where your data lives

```text
%LOCALAPPDATA%\LiveTranscriber\
    models\                      Whisper models
    logs\live-transcriber.log    Rotating technical log
    config.json                  Settings
    cache\

%USERPROFILE%\Documents\LiveTranscriber\Recordings\
    2026-09-18_14-30_Lezione_QCED\
        pc_audio.wav  microphone.wav  mixed.wav
        transcript.txt  transcript.srt  transcript.vtt
        session.json
```

Recordings are never overwritten: a session whose name already exists gets a
numeric suffix.

## Privacy

LiveTranscriber processes audio and transcripts **locally on your device**. No
audio is sent to any external server for transcription.

- No telemetry, no analytics, no account.
- The only network access the app ever makes is downloading a Whisper model
  from Hugging Face, once, when you ask for it.
- The technical log records timings and sizes, never transcript text.

## Troubleshooting

### "Nessun dispositivo di riproduzione disponibile"

Windows exposes a loopback device only for endpoints that exist. Connect
speakers or headphones, or enable a device in *Sound settings → Output*.

### PC audio records silence

- Check you picked the output device Windows is actually playing through — the
  one marked *(predefinito)* is the system default.
- Some apps take exclusive control of the audio device. Turn off *Allow
  applications to take exclusive control* in the device's advanced properties.

### The microphone records silence

Check *Settings → Privacy & security → Microphone* and confirm desktop apps are
allowed to access it. Also check the device is not muted in the Windows volume
mixer.

### Bluetooth headphones sound bad while recording

Windows switches a Bluetooth headset into its low-quality "hands-free" profile
when an app opens its microphone. Record PC audio through loopback only, or use
a separate microphone, to keep the headphones in high-quality playback mode.

### Device names look truncated

Only via the legacy MME host API, which cuts names at 31 characters and
misreports sample rates. LiveTranscriber enumerates through WASAPI precisely to
avoid this.

More detail in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT — see [LICENSE](LICENSE). Third-party components and their licenses are
listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
