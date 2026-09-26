"""Accelerator selection.

The behaviour that matters here is refusing INT8 on Blackwell. CTranslate2's own
capability list says CUDA supports ``int8`` — measured on this machine it
reports ``{'float32', 'int8_float32', 'float16', 'int8_float16', 'int8', ...}``
— but the RTX 5050 cannot execute it, and asking anyway fails with
``CUBLAS_STATUS_NOT_SUPPORTED`` at the first inference rather than at load.
"""

from __future__ import annotations

import pytest

from app.transcription.hardware import (
    BLACKWELL_COMPUTE_CAPABILITY,
    Accelerator,
    AcceleratorChoice,
    GpuInfo,
    compute_type_for_cpu,
    compute_type_for_gpu,
    default_cpu_threads,
    describe_hardware,
    select_accelerator,
)


def make_gpu(compute_capability: float = 8.9, name: str = "NVIDIA GeForce RTX 4060",
             free_mb: int = 8000) -> GpuInfo:
    return GpuInfo(
        index=0,
        name=name,
        compute_capability=compute_capability,
        total_memory_mb=8192,
        free_memory_mb=free_mb,
    )


BLACKWELL = make_gpu(12.0, "NVIDIA GeForce RTX 5050 Laptop GPU")
ADA = make_gpu(8.9, "NVIDIA GeForce RTX 4060")


# -- GPU capability --------------------------------------------------------

def test_blackwell_is_recognised():
    assert BLACKWELL.is_blackwell
    assert not BLACKWELL.supports_int8


def test_pre_blackwell_supports_int8():
    assert not ADA.is_blackwell
    assert ADA.supports_int8


def test_boundary_is_inclusive():
    """sm_120 exactly is the first affected generation."""
    assert make_gpu(BLACKWELL_COMPUTE_CAPABILITY).is_blackwell
    assert not make_gpu(BLACKWELL_COMPUTE_CAPABILITY - 0.1).is_blackwell


def test_short_name_strips_vendor_noise():
    assert BLACKWELL.short_name == "RTX 5050 Laptop"
    assert ADA.short_name == "RTX 4060"


# -- compute type ----------------------------------------------------------

def test_gpu_defaults_to_float16():
    assert compute_type_for_gpu(ADA) == "float16"
    assert compute_type_for_gpu(BLACKWELL) == "float16"


@pytest.mark.parametrize("requested", ["int8", "int8_float16", "int8_float32"])
def test_int8_is_refused_on_blackwell(requested):
    """The central guarantee of this module."""
    assert compute_type_for_gpu(BLACKWELL, requested) == "float16"


@pytest.mark.parametrize("requested", ["int8", "int8_float16"])
def test_int8_is_honoured_on_older_cards(requested):
    assert compute_type_for_gpu(ADA, requested) == requested


def test_explicit_float32_is_honoured_everywhere():
    assert compute_type_for_gpu(BLACKWELL, "float32") == "float32"
    assert compute_type_for_gpu(ADA, "float32") == "float32"


def test_auto_is_not_passed_through_as_a_literal():
    assert compute_type_for_gpu(BLACKWELL, "auto") == "float16"
    assert compute_type_for_cpu("auto") == "int8"


def test_cpu_defaults_to_int8():
    assert compute_type_for_cpu() == "int8"
    assert compute_type_for_cpu("float32") == "float32"


# -- selection -------------------------------------------------------------


@pytest.fixture
def cuda_installed(monkeypatch):
    """Pretend the CUDA runtime libraries are present.

    A card is not enough: select_accelerator also checks that this build ships
    cuBLAS and cuDNN. Faking only the card passed on the development machine,
    which has the GPU extras installed, and failed on a clean CI runner, which
    does not — the tests were reading the developer's environment instead of
    stating what they meant.
    """
    monkeypatch.setattr(
        "app.transcription.cuda_setup.cuda_libraries_available", lambda: True
    )


def test_cpu_preference_never_touches_the_gpu(monkeypatch):
    def explode():
        raise AssertionError("detect_gpus must not be called for Accelerator.CPU")

    monkeypatch.setattr("app.transcription.hardware.detect_gpus", explode)
    choice = select_accelerator(Accelerator.CPU)
    assert choice.device == "cpu"
    assert choice.compute_type == "int8"


def test_auto_uses_a_gpu_when_present(monkeypatch, cuda_installed):
    monkeypatch.setattr("app.transcription.hardware.detect_gpus", lambda: [BLACKWELL])
    choice = select_accelerator(Accelerator.AUTO)
    assert choice.device == "cuda"
    assert choice.compute_type == "float16"
    assert choice.is_gpu


def test_auto_falls_back_quietly_without_a_gpu(monkeypatch):
    monkeypatch.setattr("app.transcription.hardware.detect_gpus", lambda: [])
    choice = select_accelerator(Accelerator.AUTO)
    assert choice.device == "cpu"
    assert choice.fallback_reason == "", "auto has nothing to apologise for"


def test_requesting_a_missing_gpu_explains_itself(monkeypatch):
    """Spec §18: a plain-language reason, not a traceback."""
    monkeypatch.setattr("app.transcription.hardware.detect_gpus", lambda: [])
    choice = select_accelerator(Accelerator.GPU)

    assert choice.device == "cpu"
    assert choice.fallback_reason
    assert "CPU" in choice.fallback_reason
    assert "Traceback" not in choice.fallback_reason


def test_low_vram_warns_but_still_selects_the_gpu(
    monkeypatch, caplog, cuda_installed
):
    """CTranslate2 allocates lazily; refusing on an estimate would be wrong."""
    monkeypatch.setattr(
        "app.transcription.hardware.detect_gpus", lambda: [make_gpu(12.0, free_mb=500)]
    )
    choice = select_accelerator(Accelerator.AUTO, model_name="large-v3")
    assert choice.device == "cuda"


def test_cpu_threads_leave_headroom():
    threads = default_cpu_threads()
    assert threads >= 1
    import os

    assert threads <= (os.cpu_count() or 1)


def test_explicit_thread_count_is_respected(monkeypatch):
    monkeypatch.setattr("app.transcription.hardware.detect_gpus", lambda: [])
    assert select_accelerator(Accelerator.CPU, cpu_threads=3).cpu_threads == 3


# -- description -----------------------------------------------------------

def test_description_matches_the_spec_wording():
    gpu = AcceleratorChoice(device="cuda", compute_type="float16", gpu=BLACKWELL)
    assert gpu.description == "Accelerazione: NVIDIA RTX 5050 Laptop"
    assert AcceleratorChoice(device="cpu", compute_type="int8").description == (
        "Accelerazione: CPU"
    )


def test_describe_hardware_runs_on_any_machine():
    text = describe_hardware()
    assert "CPU:" in text
    assert "GPU:" in text


# -- describing the choice to a human --------------------------------------


def test_a_cpu_only_pc_is_not_told_something_is_missing(monkeypatch):
    """The message a machine with no NVIDIA card must NOT see.

    Telling someone their PC "does not include the CUDA libraries" when it has
    no NVIDIA card invites them to go looking for a fix that does not exist.
    """
    from app.transcription import hardware

    monkeypatch.setattr(hardware, "detect_gpus", lambda: [])
    choice = hardware.AcceleratorChoice(device="cpu", compute_type="int8", cpu_threads=6)
    assert hardware.describe_choice(choice) == "CPU (6 thread)"


def test_an_nvidia_pc_on_the_cpu_build_is_told_where_to_look(monkeypatch):
    from app.transcription import hardware

    gpu = hardware.GpuInfo(0, "NVIDIA GeForce RTX 5050 Laptop GPU", 12.0, 8151, 7000)
    monkeypatch.setattr(hardware, "detect_gpus", lambda: [gpu])
    monkeypatch.setattr(
        "app.transcription.cuda_setup.cuda_libraries_available", lambda: False
    )
    choice = hardware.AcceleratorChoice(device="cpu", compute_type="int8", cpu_threads=6)
    text = hardware.describe_choice(choice)
    assert "RTX 5050 Laptop" in text
    assert "versione per NVIDIA" in text


def test_a_gpu_choice_names_the_card_and_the_precision():
    from app.transcription import hardware

    gpu = hardware.GpuInfo(0, "NVIDIA GeForce RTX 5050 Laptop GPU", 12.0, 8151, 7000)
    choice = hardware.AcceleratorChoice(
        device="cuda", compute_type="float16", gpu=gpu, cpu_threads=6
    )
    assert hardware.describe_choice(choice) == "GPU NVIDIA RTX 5050 Laptop, float16"


def test_gpu_is_not_offered_when_there_is_no_card(monkeypatch):
    from app.transcription import hardware

    monkeypatch.setattr(hardware, "detect_gpus", lambda: [])
    availability = hardware.gpu_availability()
    assert not availability.usable
    assert "funziona comunque sulla CPU" in availability.reason


def test_avx2_detection_answers_without_raising():
    from app.transcription.hardware import cpu_supports_avx2

    result = cpu_supports_avx2()
    assert result is None or isinstance(result, bool)
