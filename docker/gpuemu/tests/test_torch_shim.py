"""Tests for the PyTorch bridge. Skipped where torch is not installed.

The image build runs these after installing torch, so they gate the release
even though a bare source checkout skips them.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")


@pytest.fixture(scope="module", autouse=True)
def shim(tmp_path_factory):
    import os

    d = tmp_path_factory.mktemp("gpuemu")
    os.environ["GPUEMU_STATE_FILE"] = str(d / "state.bin")
    os.environ["GPUEMU_CLAIMS_DIR"] = str(d / "claims")
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    from gpuemu.daemon import Daemon
    import gpuemu.torch_shim as ts

    Daemon().tick(0.2)
    ts.install()
    yield ts


@pytest.fixture(autouse=True)
def clean_ledger():
    ledger = torch.cuda._gpuemu_ledger
    ledger.allocated = 0
    ledger.peak = 0
    ledger.claim.set_memory(0)
    yield


def test_cuda_reports_available():
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1


def test_device_identifies_as_the_emulated_card():
    from gpuemu import spec

    dev = spec.selected_device()
    assert torch.cuda.get_device_name(0) == dev.name
    assert torch.cuda.get_device_capability(0) == (dev.cc_major, dev.cc_minor)
    assert torch.cuda.get_device_properties(0).total_memory == dev.mem_total_bytes


def test_tensors_can_be_created_on_cuda():
    x = torch.randn(256, 256, device="cuda")
    assert x.shape == (256, 256)
    assert torch.cuda.memory_allocated() == 256 * 256 * 4


def test_arithmetic_works():
    a = torch.ones(64, 64, device=torch.device("cuda"))
    b = torch.ones(64, 64, device=torch.device("cuda"))
    assert float((a @ b).sum()) == pytest.approx(64 * 64 * 64)


def test_dot_cuda_and_dot_to_are_accounted():
    x = torch.zeros(1024, 1024)  # 4 MiB
    x.cuda()
    assert torch.cuda.memory_allocated() == 4 * 1024**2

    y = torch.zeros(1024, 1024)
    y.to("cuda")
    assert torch.cuda.memory_allocated() == 8 * 1024**2


def test_module_cuda_accounts_parameters():
    model = torch.nn.Linear(1000, 1000)  # 1000*1000 + 1000 floats
    model.cuda()
    expected = (1000 * 1000 + 1000) * 4
    assert torch.cuda.memory_allocated() == expected


def test_oversized_allocation_raises_before_allocating_host_memory():
    """The limit must be checked first.

    Allocating and then accounting would mean this request really asks the host
    for hundreds of gigabytes, and the container is OOM-killed instead of
    raising the error the exercise is about.
    """
    import resource

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with pytest.raises(RuntimeError) as exc:
        torch.zeros(100_000, 100_000, device="cuda")  # ~37 GiB
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert "out of memory" in str(exc.value).lower()
    # Peak RSS must not have jumped by anything like the requested size.
    assert after - before < 2 * 1024**2  # ru_maxrss is KiB on Linux, MiB on mac


def test_oom_is_the_error_type_pytorch_raises():
    expected = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    with pytest.raises(expected):
        torch.zeros(100_000, 100_000, device="cuda")


def test_failed_allocation_does_not_leak_accounting():
    baseline = torch.cuda.memory_allocated()
    with pytest.raises(RuntimeError):
        torch.zeros(100_000, 100_000, device="cuda")
    assert torch.cuda.memory_allocated() == baseline


def test_memory_is_reported_to_nvml():
    """What torch thinks it holds must be what nvidia-smi shows."""
    from gpuemu.daemon import Daemon
    from gpuemu.shm import StateReader

    torch.randn(2048, 2048, device="cuda")  # 16 MiB
    d = Daemon()
    d.tick(0.2)
    with StateReader(d.writer.path) as r:
        gpu = r.read().gpus[0]
    assert gpu.mem_used >= 16 * 1024**2
    assert any(p.used_mem >= 16 * 1024**2 for p in gpu.processes)


def test_mem_get_info_reflects_allocation():
    free_before, total = torch.cuda.mem_get_info()
    torch.randn(2048, 2048, device="cuda")  # 16 MiB
    free_after, _ = torch.cuda.mem_get_info()
    assert total == torch.cuda.get_device_properties(0).total_memory
    assert free_before - free_after == 16 * 1024**2


def test_empty_cache_and_reset_peak_are_harmless():
    torch.randn(512, 512, device="cuda")
    peak = torch.cuda.max_memory_allocated()
    assert peak > 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    assert torch.cuda.max_memory_allocated() == torch.cuda.memory_allocated()


def test_synchronize_and_events_work():
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.randn(128, 128, device="cuda")
    end.record()
    torch.cuda.synchronize()
    assert end.elapsed_time(start) <= 0 or start.elapsed_time(end) >= 0


def test_oom_message_reports_this_process_allocation():
    """'already allocated' must mean what PyTorch means by it: ours."""
    torch.randn(4096, 4096, device="cuda")  # 64 MiB
    with pytest.raises(RuntimeError) as exc:
        torch.zeros(100_000, 100_000, device="cuda")
    message = str(exc.value)
    assert "64.00 MiB already allocated" in message, message
    assert "total capacity" in message
    assert "free" in message


def test_torch_reports_no_cuda_when_no_gpu_allocated():
    """The 'forgot --gpus-per-node' case must fall back to the CPU."""
    import os
    import pathlib
    import subprocess
    import sys

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    # A source checkout is not on the subprocess's path the way pytest's
    # pythonpath setting puts it on ours.
    src = str(pathlib.Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    # A fresh interpreter, because the shim installs once per process.
    out = subprocess.run(
        [sys.executable, "-c",
         "import gpuemu.torch_shim, torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True, env=env,
    )
    assert out.stdout.strip() == "False", f"stdout={out.stdout!r} stderr={out.stderr!r}"
