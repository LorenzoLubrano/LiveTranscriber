# Third-party licenses

LiveTranscriber itself is MIT licensed (see [LICENSE](LICENSE)). It depends on
the components below. Versions are the ones verified in this project's
environment on 2026-09-18.

## Summary

Every dependency is compatible with distributing LiveTranscriber as MIT-licensed
open source. Two are copyleft and carry conditions that shape how the app is
packaged — see *Obligations* below.

| Component | Version | License | Used for |
|---|---|---|---|
| PySide6 / shiboken6 | 6.11.2 | **LGPL-3.0** (or GPL-2.0 / GPL-3.0) | GUI toolkit (Qt 6) |
| faster-whisper | 1.2.1 | MIT | Whisper inference wrapper |
| CTranslate2 | 4.8.2 | MIT | Inference engine |
| PyAudioWPatch | 0.2.12.8 | Apache-2.0 (MIT for PortAudio) | WASAPI capture + loopback |
| onnxruntime | 1.30.0 | MIT | Silero VAD execution |
| soxr (python-soxr) | 1.1.0 | **LGPL-2.1-or-later** | Sample-rate conversion |
| NumPy | 2.5.3 | BSD-3-Clause | Buffer maths |
| soundfile | 0.14.0 | BSD-3-Clause (libsndfile: LGPL-2.1) | WAV reading in tools |
| PyAV | 18.1.0 | BSD-3-Clause (FFmpeg: LGPL-2.1) | Audio decoding inside faster-whisper |
| tokenizers | 0.23.2 | Apache-2.0 | Whisper tokenisation |
| huggingface-hub | 1.32.0 | Apache-2.0 | Model download |
| protobuf | 7.36.2 | BSD-3-Clause | onnxruntime dependency |
| flatbuffers | 25.12.19 | Apache-2.0 | onnxruntime dependency |
| psutil | 7.2.2 | BSD-3-Clause | Benchmark / resource metrics |
| cffi | 2.1.1 | MIT-0 | soundfile dependency |
| tqdm | 4.70.1 | MPL-2.0 / MIT | Download progress |
| PyInstaller | 6.22.3 | GPL-2.0 **with bundling exception** | Build tooling only |
| nvidia-cublas-cu12 | 12.9.2.10 | **NVIDIA proprietary** (redistributable) | Optional GPU extra |
| nvidia-cudnn-cu12 | 9.26.0.51 | **NVIDIA proprietary** (redistributable) | Optional GPU extra |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 | **NVIDIA proprietary** (redistributable) | Optional GPU extra |

### Whisper models

Model weights are downloaded by the user, not shipped with the app. The OpenAI
Whisper models are released under the **MIT license**; the CTranslate2
conversions on Hugging Face carry the same terms. No model files are included in
this repository or in a release build.

## Obligations

### LGPL: Qt (PySide6) and libsoxr

The LGPL permits an MIT-licensed application to use these libraries, provided the
user can replace them with their own build. Two things follow:

1. **The release is a portable folder, not a single-file `.exe`.** Qt and soxr
   ship as separate DLLs/`.pyd` files inside `_internal/`, so a user can swap in
   their own build. A one-file bundle would make relinking materially harder.
   (Reliability was the primary reason for this choice; LGPL compliance is a
   second one.)
2. **License texts are shipped with the build** and the About screen names these
   components, their licenses, and where to obtain their sources.

Neither library is modified by this project.

### GPL: PyInstaller

PyInstaller is a **build-time tool** and is not part of the distributed
application. Its license additionally grants an explicit exception permitting
the resulting bundled applications to be distributed under any license.

### NVIDIA CUDA runtime libraries

cuBLAS, cuDNN and NVRTC are **proprietary**, not open source. They are covered
by NVIDIA's redistribution terms, which permit shipping them alongside an
application.

They are deliberately kept out of the default install: CPU transcription works
with no NVIDIA component at all, and the GPU libraries arrive only when a user
opts in with `pip install -e ".[gpu]"`. The default portable build therefore
contains no proprietary code, and a GPU-enabled build must carry NVIDIA's
license text.

### MPL-2.0: tqdm

File-level copyleft. tqdm is used unmodified, so the only obligation is to
retain its license notice.

## Deliberately excluded

- **Cloud speech-to-text APIs** (OpenAI, Google, Azure, AWS) — incompatible with
  the project's offline, privacy-first goal, and would require API keys.
- **GPL-licensed libraries linked into the app** — would force LiveTranscriber
  itself to become GPL.
- **Bundling NVIDIA CUDA libraries in the default release** — the standard
  build stays fully open source; GPU support is an opt-in extra.
- **FFmpeg built with `--enable-gpl` or `--enable-nonfree`** — the LGPL build
  bundled inside PyAV is used instead, and no FFmpeg binary is shipped.

## Regenerating this list

```powershell
python scripts/check_licenses.py
```
