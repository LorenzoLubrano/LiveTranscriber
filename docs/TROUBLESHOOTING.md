# Troubleshooting

Audience: people using LiveTranscriber who hit a problem.

First step for anything audio-related:

```powershell
python scripts/audio_test.py --list
python scripts/audio_test.py --both
```

The technical log is at `%LOCALAPPDATA%\LiveTranscriber\logs\live-transcriber.log`.
It contains timings, device names and errors — never your transcripts.

---

## Audio

### "Nessun dispositivo di riproduzione disponibile"

Windows creates a loopback device only for output endpoints that actually exist.
Connect speakers or headphones, or enable an output device in
*Settings → System → Sound*.

### PC audio records nothing / RMS reported as silence

Work through these in order:

1. **Was audio actually playing?** Loopback records what Windows sends to the
   endpoint. A paused video produces digital silence, correctly.
2. **Is it the right endpoint?** If sound comes out of your headphones but you
   selected *Speakers*, the speakers' loopback is silent. Pick the device marked
   *(predefinito)*, or the one *Sound settings* shows the level meter moving on.
3. **Is another app holding the device exclusively?** Some DAWs, games and
   conferencing apps take exclusive control. Open
   *Sound settings → the device → Advanced*, and turn off
   *Allow applications to take exclusive control of this device*.
4. **Per-app output routing.** Windows lets an app be routed to a different
   endpoint in *Sound settings → Volume mixer*. Check the app you are recording
   is not sent somewhere else.

### The microphone records nothing

1. *Settings → Privacy & security → Microphone* — both **Microphone access** and
   **Let desktop apps access your microphone** must be on.
2. Check the device is not muted in the Windows volume mixer, and that it is not
   muted by a hardware switch on a headset.
3. Some laptops route a headset jack to a different endpoint than the internal
   array. Try each microphone listed by `audio_test.py --list`.

### Bluetooth headphones sound terrible while recording

This is Windows, not LiveTranscriber. A Bluetooth headset has two profiles:

- **A2DP** — stereo, high quality, playback only.
- **HFP/HSP** — mono, ~8–16 kHz, but the microphone works.

The moment any app opens the headset's microphone, Windows switches the whole
device to HFP and playback quality collapses.

**Recommended:** record PC audio through loopback, and use a *different*
microphone (your laptop's array) for your own voice. The headphones then stay in
A2DP.

### A device disappears mid-recording

LiveTranscriber detects a stream that has stopped delivering audio for more than
3 seconds and reports it rather than recording silence. Audio captured before
that point is already on disk.

Bluetooth dropouts and USB re-enumeration are the usual causes. Stop, reselect
the device, and start a new session — the previous recording is intact.

### Device names are cut off

Only through the legacy MME interface, which truncates names at 31 characters
(`Microphone Array (Realtek(R) Au`) and misreports sample rates.
LiveTranscriber enumerates through WASAPI specifically to avoid this; if you see
truncated names, WASAPI enumeration failed and the app fell back — check the log.

### Crackling, gaps, or dropped frames reported

`audio_test.py` reports a non-zero **dropped** count when the consumer could not
keep up. Usually one of:

- A heavier Whisper model than the machine can sustain — drop to `small` or `base`.
- Windows power saving throttling the CPU — switch the power mode to *Balanced*
  or *Best performance*.
- Another process saturating the disk or CPU.

---

## GPU / CUDA

### "Accelerazione: CPU" even though I have an NVIDIA GPU

Install the GPU extra:

```powershell
pip install -e ".[gpu]"
```

This brings in cuBLAS and cuDNN as Python packages — **the CUDA Toolkit is not
required**. Then confirm the driver is present:

```powershell
nvidia-smi
```

### `CUBLAS_STATUS_NOT_SUPPORTED` on an RTX 50-series card

Blackwell GPUs (sm_120) do not support CTranslate2's INT8 kernels — CTranslate2
disabled INT8 for sm_120 in release 4.6.2. Use `float16`, which LiveTranscriber
selects automatically for these cards. If you overrode the compute type in
advanced settings, press **Ripristina valori consigliati**.

You also need CTranslate2 ≥ 4.8.2 and a CUDA 12.8+ runtime:

```powershell
pip install -U "ctranslate2>=4.8.2" "nvidia-cublas-cu12>=12.8" "nvidia-cudnn-cu12>=9.7"
```

### "Memoria GPU insufficiente"

The chosen model does not fit in free VRAM. Either pick a smaller model, or
close whatever else is using the GPU. LiveTranscriber falls back to CPU
automatically and keeps transcribing — slower, but nothing is lost.

### The first transcription is slow, then it speeds up

Expected on Blackwell: sm_120 runs through PTX JIT compilation the first time a
kernel is used. The app warms the model up at load to keep this off your first
segment.

---

## Models

### The download is stuck or fails

Model files come from Hugging Face. Check a proxy or firewall is not blocking
`huggingface.co`. Partial downloads are resumed, not restarted.

### Can I use it offline?

Yes — that is the point. Once a model is in
`%LOCALAPPDATA%\LiveTranscriber\models\`, the app never needs the network again.
Downloading a model is the only network access it ever makes.

---

## Reporting a problem

Include:

1. The output of `python scripts/audio_test.py --list`.
2. The last ~50 lines of `live-transcriber.log`.
3. Your Windows version, CPU and GPU.

The log holds no transcript text, so it is safe to share.
