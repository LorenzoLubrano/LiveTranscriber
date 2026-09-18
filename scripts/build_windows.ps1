<#
.SYNOPSIS
    Builds the LiveTranscriber Windows release.

.DESCRIPTION
    Produces a portable folder under dist\LiveTranscriber containing
    LiveTranscriber.exe and everything it needs. The end user installs nothing:
    no Python, no pip, no Visual Studio, no CUDA Toolkit.

    A folder rather than a single .exe, deliberately. One-file builds extract
    themselves to %TEMP% on every launch, which is slow and a known source of
    native-DLL failures with Qt and CTranslate2 — and it would also make the
    LGPL relinking requirement for Qt and libsoxr much harder to satisfy.

    Whisper models are never bundled. They are downloaded on demand into
    %LOCALAPPDATA%\LiveTranscriber\models, which keeps the release small and
    lets the user choose what to spend disk on.

.PARAMETER Gpu
    Include the NVIDIA CUDA runtime (~2 GB). Without it the build runs on CPU
    only, on any machine. With it, an NVIDIA GPU is used when present.

.PARAMETER Clean
    Remove build\ and dist\ before building.

.PARAMETER SkipTests
    Skip the test run. Not recommended: the suite is fast.

.EXAMPLE
    .\scripts\build_windows.ps1
    .\scripts\build_windows.ps1 -Gpu -Clean
#>
[CmdletBinding()]
param(
    [switch]$Gpu,
    [switch]$Clean,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root '.venv'
$Python = Join-Path $Venv 'Scripts\python.exe'
$DistDir = Join-Path $Root 'dist'
$BuildDir = Join-Path $Root 'build'
# The spec names the folder after the variant, so a GPU build can never
# overwrite a CPU build that someone is already using.
$BundleName = if ($Gpu) { 'LiveTranscriber-GPU' } else { 'LiveTranscriber' }
$AppDir = Join-Path $DistDir $BundleName

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Detail([string]$Message) {
    Write-Host "    $Message" -ForegroundColor DarkGray
}

# --------------------------------------------------------------- environment

Write-Step 'Checking the build environment'

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found at $Venv. Create it with: py -3.13 -m venv .venv"
}
Write-Detail "python:  $(& $Python --version)"

& $Python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller is not installed. Run: pip install -e ".[build]"'
}
Write-Detail "pyinstaller: $(& $Python -m PyInstaller --version)"

if ($Gpu) {
    & $Python -c "import nvidia" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'CUDA libraries are not installed. Run: pip install -e ".[gpu]"  (or build without -Gpu)'
    }
    Write-Detail 'CUDA runtime: will be bundled (~2 GB)'
} else {
    Write-Detail 'CUDA runtime: not bundled (CPU-only build)'
}

# --------------------------------------------------------------------- tests

if (-not $SkipTests) {
    Write-Step 'Running the test suite'
    & $Python -m pytest $Root\tests -q -m "not hardware"
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed; not building a release.' }
} else {
    Write-Detail 'Tests skipped by request'
}

# --------------------------------------------------------------------- clean

if ($Clean) {
    Write-Step 'Cleaning previous build output'
    # Only this variant's output, so cleaning a GPU build leaves the CPU one.
    foreach ($dir in @((Join-Path $BuildDir $BundleName), $AppDir)) {
        if (Test-Path $dir) {
            Remove-Item -Recurse -Force $dir
            Write-Detail "removed $dir"
        }
    }
}

# --------------------------------------------------------------------- build

Write-Step 'Building with PyInstaller'

$env:LIVETRANSCRIBER_BUNDLE_CUDA = if ($Gpu) { '1' } else { '0' }
$specPath = Join-Path $Root 'scripts\LiveTranscriber.spec'

Push-Location $Root
try {
    & $Python -m PyInstaller $specPath --noconfirm --distpath $DistDir --workpath $BuildDir
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }
} finally {
    Pop-Location
}

if (-not (Test-Path (Join-Path $AppDir 'LiveTranscriber.exe'))) {
    throw "Build finished but LiveTranscriber.exe is missing from $AppDir"
}

# ------------------------------------------------------------------ licences

Write-Step 'Adding licence files'

$licenceDir = Join-Path $AppDir 'licenses'
New-Item -ItemType Directory -Force -Path $licenceDir | Out-Null
foreach ($file in @('LICENSE', 'THIRD_PARTY_LICENSES.md', 'README.md')) {
    $source = Join-Path $Root $file
    if (Test-Path $source) {
        Copy-Item $source $licenceDir
        Write-Detail "copied $file"
    }
}

# LGPL requires users be able to replace Qt and libsoxr with their own builds.
# The one-dir layout already allows that; this note says where they are.
@"
LiveTranscriber is MIT licensed. See LICENSE.

This build includes Qt (via PySide6) and libsoxr, both LGPL. Their binaries are
the separate files under _internal\, and may be replaced with your own builds.
Full details, including where to obtain their sources, are in
THIRD_PARTY_LICENSES.md.
"@ | Set-Content -Path (Join-Path $licenceDir 'README-LICENSES.txt') -Encoding utf8

# ------------------------------------------------------------------- summary

Write-Step 'Measuring the result'

$size = (Get-ChildItem $AppDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
$files = (Get-ChildItem $AppDir -Recurse -File | Measure-Object).Count
$exe = Get-Item (Join-Path $AppDir 'LiveTranscriber.exe')

Write-Host ''
Write-Host 'Build complete' -ForegroundColor Green
Write-Host "  folder    : $AppDir"
Write-Host "  executable: $($exe.Name)  ($([math]::Round($exe.Length / 1MB, 1)) MB)"
Write-Host "  total size: $([math]::Round($size / 1MB, 0)) MB in $files files"
Write-Host "  GPU       : $(if ($Gpu) { 'CUDA runtime bundled' } else { 'CPU only' })"
Write-Host ''
Write-Host '  Whisper models are not bundled; the app downloads them on first use.'
Write-Host "  Run it with: $AppDir\LiveTranscriber.exe"
Write-Host ''
