"""The C header and the Python struct layout must describe the same bytes.

If they drift, nothing raises: the shim simply reads fields at the wrong offsets
and reports plausible-looking nonsense. This test compiles the header's view of
itself and compares it against Python's, so drift fails loudly instead.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from gpuemu import shm

NVML_DIR = Path(__file__).resolve().parents[1] / "nvml"


def _c_layout(build_dir: Path) -> dict[str, int]:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler available")
    out_bin = build_dir / "abi_dump"
    subprocess.run(
        [cc, "-O0", "-std=c11", "-o", str(out_bin), str(NVML_DIR / "abi_dump.c")],
        check=True,
        cwd=NVML_DIR,
    )
    text = subprocess.run([str(out_bin)], check=True, capture_output=True, text=True).stdout
    return {k: int(v) for k, v in (line.split() for line in text.strip().splitlines())}


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    return _c_layout(tmp_path_factory.mktemp("abi"))


def test_struct_sizes_match(layout):
    assert layout["proc_size"] == shm.PROC_SIZE
    assert layout["gpu_size"] == shm.GPU_SIZE
    assert layout["shm_size"] == shm.SHM_SIZE


def test_array_bounds_match(layout):
    assert layout["max_gpus"] == shm.MAX_GPUS
    assert layout["max_procs"] == shm.MAX_PROCS


def test_offsets_match(layout):
    assert layout["off_seq"] == shm.SEQ_OFFSET
    assert layout["off_gpus"] == shm.SHM_HEAD_SIZE
    assert layout["off_gpu_procs"] == shm.GPU_HEAD_SIZE


def test_roundtrip_through_shared_memory(tmp_path):
    """What the writer packs, a reader gets back unchanged."""
    path = tmp_path / "state.bin"
    gpu = shm.GPUState(
        name="NVIDIA L4",
        uuid="GPU-deadbeef",
        bus_id="00000000:00:04.0",
        mem_total=23034 * 1024 * 1024,
        mem_used=4096 * 1024 * 1024,
        util_gpu=87,
        temp=64,
        power_mw=58500,
        fan_speed=-1,
        processes=[shm.Process(pid=4242, name="python", used_mem=4096 * 1024 * 1024, sm_util=87)],
    )
    with shm.StateWriter(path) as w:
        w.write([gpu], "550.54.15", "12.4", "12.550.54.15")

        with shm.StateReader(path) as r:
            snap = r.read()

    assert snap.driver_version == "550.54.15"
    assert snap.cuda_version == "12.4"
    assert len(snap.gpus) == 1

    got = snap.gpus[0]
    assert got.name == "NVIDIA L4"
    assert got.mem_total == 23034 * 1024 * 1024
    assert got.util_gpu == 87
    assert got.temp == 64
    assert got.power_mw == 58500
    assert got.fan_speed == -1  # passive card, must survive as a negative
    assert len(got.processes) == 1
    assert got.processes[0].pid == 4242
    assert got.processes[0].name == "python"
