# PyInstaller build definition.
#
# Driven by scripts/build_windows.ps1; the only input from outside is the
# LIVETRANSCRIBER_BUNDLE_CUDA environment variable.
#
# Two things here are worth knowing:
#
# * faster-whisper ships ``assets/silero_vad_v6.onnx`` inside its package, and
#   the app depends on it for voice activity detection. It is data, not an
#   import, so it has to be collected explicitly or VAD silently degrades to the
#   energy fallback in the frozen build.
# * PySide6 is 630 MB installed, almost all of which is Qt modules this app
#   never touches — WebEngine, 3D, Charts, Quick. Excluding them is the single
#   biggest size saving available.

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent

BUNDLE_CUDA = os.environ.get("LIVETRANSCRIBER_BUNDLE_CUDA", "0") == "1"

# Distinct output folders: building the GPU variant must never overwrite a CPU
# build someone is already using.
BUNDLE_NAME = "LiveTranscriber-GPU" if BUNDLE_CUDA else "LiveTranscriber"

binaries = []
datas = []
hiddenimports = [
    "app.selftest",
    "app.ui.main_window",
    "app.ui.settings_window",
    "app.ui.download_dialog",
    "app.ui.recovery_dialog",
    "app.export.txt",
    "app.export.srt",
    "app.export.vtt",
    "app.export.json_export",
]

# -- faster-whisper: the bundled Silero VAD model is data, not an import ----
datas += collect_data_files("faster_whisper", includes=["assets/*"])

# -- the application icon --------------------------------------------------
# On the .exe below for Explorer, and here as a file so the running window and
# its taskbar button can use it too.
for icon_name in ("icon.ico", "icon-512.png"):
    icon_path = ROOT / "assets" / icon_name
    if icon_path.exists():
        datas.append((str(icon_path), "assets"))

# -- native extensions -----------------------------------------------------
for package in ("ctranslate2", "onnxruntime", "pyaudiowpatch", "soxr", "av"):
    try:
        binaries += collect_dynamic_libs(package)
    except Exception:
        pass

# -- tokenizers and huggingface need their metadata to import --------------
for package in ("tokenizers", "huggingface_hub"):
    try:
        datas += collect_data_files(package)
    except Exception:
        pass

# -- CUDA runtime, only when asked -----------------------------------------
if BUNDLE_CUDA:
    import nvidia

    # `nvidia` is a namespace package, so __file__ is None and only __path__
    # exists. Reaching for __file__ here failed the whole GPU build with a
    # TypeError deep inside pathlib.
    nvidia_root = Path(list(nvidia.__path__)[0])
    for dll in nvidia_root.rglob("*.dll"):
        # Keep the nvidia/<pkg>/bin layout: cuda_setup.py looks for exactly
        # that shape next to the executable when frozen.
        binaries.append((str(dll), str(Path("nvidia") / dll.relative_to(nvidia_root).parent)))

# -- Qt modules this app never uses ----------------------------------------
excluded_qt = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets", "PySide6.QtQuick", "PySide6.QtQuick3D",
    "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras", "PySide6.Qt3DInput", "PySide6.Qt3DLogic", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning", "PySide6.QtLocation",
    "PySide6.QtSerialPort", "PySide6.QtSensors", "PySide6.QtTest", "PySide6.QtSql",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtUiTools", "PySide6.QtPdf",
    "PySide6.QtPdfWidgets", "PySide6.QtSpatialAudio", "PySide6.QtRemoteObjects",
    "PySide6.QtScxml", "PySide6.QtStateMachine", "PySide6.QtTextToSpeech",
    "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtSvgWidgets",
    "PySide6.QtHttpServer", "PySide6.QtGraphs", "PySide6.QtQuickWidgets",
]

excludes = [
    *excluded_qt,
    # Build- and test-time only.
    "pytest", "_pytest", "pytest_qt", "ruff", "PyInstaller",
    # Pulled in transitively, never used at runtime.
    "tkinter", "matplotlib", "pandas", "scipy", "IPython", "notebook",
    "setuptools", "pip", "wheel",
]

if not BUNDLE_CUDA:
    excludes.append("nvidia")


a = Analysis(
    [str(ROOT / "app" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LiveTranscriber",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX corrupts some Qt and CUDA DLLs
    console=False,      # a GUI app must not open a console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "icon.ico")
    if (ROOT / "assets" / "icon.ico").exists()
    else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=BUNDLE_NAME,
)
