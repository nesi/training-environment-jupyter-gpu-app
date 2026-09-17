"""gpuemu - an emulated NVIDIA GPU for teaching, on machines that have none.

There is no way to run real CUDA kernels without a real GPU, and this package
does not pretend otherwise. What it does is make everything *around* the GPU
behave correctly: ``nvidia-smi`` and ``nvtop`` show a device with believable
telemetry, memory is genuinely accounted so oversubscribing it raises an
out-of-memory error, jobs queue and run through a Slurm-shaped scheduler, and
PyTorch code written for CUDA runs unmodified. The arithmetic happens on the
CPU.

That covers most of what a GPU workshop actually teaches - how to request a
GPU, how to tell whether you are using it, how to read utilisation and memory,
what OOM looks like and how to respond - without needing hardware.

Typical use::

    import gpuemu

    with gpuemu.gpu(memory="4GiB"):
        train()            # nvidia-smi and nvtop show the load while this runs

or, for framework code::

    import gpuemu.torch_shim   # before importing torch
    import torch
    torch.cuda.is_available()  # True
"""

from __future__ import annotations

from .client import (
    Claim,
    GPUEmuError,
    OutOfMemoryError,
    device_capacity,
    gpu,
    parse_size,
    process_claim,
    visible_devices,
)
from .spec import CUDA_VERSION, DEVICES, DRIVER_VERSION, DeviceSpec, selected_device

__all__ = [
    "Claim",
    "GPUEmuError",
    "OutOfMemoryError",
    "CUDA_VERSION",
    "DRIVER_VERSION",
    "DEVICES",
    "DeviceSpec",
    "device_capacity",
    "gpu",
    "parse_size",
    "process_claim",
    "selected_device",
    "visible_devices",
    "is_available",
    "device_count",
    "snapshot",
]

__version__ = "0.1.0"


def is_available() -> bool:
    """True when the emulator daemon is running and a device is visible."""
    from .shm import StateReader

    vis = visible_devices()
    if vis is not None and not vis:
        return False
    reader = StateReader.try_open()
    if reader is None:
        return False
    try:
        return bool(reader.read().gpus)
    except ValueError:
        return False
    finally:
        reader.close()


def device_count() -> int:
    """Number of devices visible to this process."""
    from .shm import StateReader

    reader = StateReader.try_open()
    if reader is None:
        return 0
    try:
        total = len(reader.read().gpus)
    except ValueError:
        return 0
    finally:
        reader.close()
    vis = visible_devices()
    if vis is None:
        return total
    return len([d for d in vis if d < total])


def snapshot():
    """Current state of every emulated device, for scripting and notebooks."""
    from .shm import StateReader

    reader = StateReader.try_open()
    if reader is None:
        raise GPUEmuError(
            "the gpuemu daemon is not running; start it with 'gpuemu-ctl start'"
        )
    try:
        return reader.read()
    finally:
        reader.close()
