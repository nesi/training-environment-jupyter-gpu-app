"""How a process tells the emulator it is using the GPU.

A process that wants to appear on the device drops a small JSON "claim" file in
the claims directory saying who it is, how much memory it holds and how busy it
is. The daemon reads every claim each tick and folds them into the device state
that NVML then reports.

Claims are files rather than messages to a server for one reason: a process that
crashes cannot leave the GPU allocated. The daemon checks whether each claim's
pid is still alive and reaps the ones that are not, so a killed notebook kernel
frees its memory the same way a real driver would clean up after a dead context.

Memory is genuinely accounted, so asking for more than the card has raises
:class:`OutOfMemoryError`. That is deliberate: hitting a CUDA OOM and learning to
read it is a large part of what a GPU workshop is for, and an emulator that had
infinite memory would quietly teach the wrong lesson.
"""

from __future__ import annotations

import atexit
import errno
import fcntl
import json
import os
import threading
import time
from pathlib import Path
from typing import Literal

from . import spec
from .shm import StateReader, default_state_file

UtilSpec = Literal["auto"] | int | float


class GPUEmuError(RuntimeError):
    """Base class for emulator failures."""


class OutOfMemoryError(GPUEmuError):
    """Raised when a claim would exceed the emulated device's memory."""


def claims_dir() -> Path:
    env = os.environ.get("GPUEMU_CLAIMS_DIR")
    d = Path(env) if env else default_state_file().parent / "claims"
    d.mkdir(parents=True, exist_ok=True)
    try:
        # Any user in the session container may claim the GPU; the sticky bit
        # keeps them from deleting each other's claims.
        os.chmod(d, 0o1777)
    except OSError:
        pass
    return d


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def visible_devices() -> list[int] | None:
    """Device indices this process may use, honouring ``CUDA_VISIBLE_DEVICES``.

    Returns None when the variable is unset (all devices visible). An empty
    list means the variable is set but empty, which is how a real scheduler
    hides the GPU from a job that did not request one.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            # Real CUDA accepts UUIDs here and stops at the first bad entry.
            break
    return out


def parse_size(value: str | int | float) -> int:
    """Turn ``"2GiB"``, ``"512M"`` or a plain number of bytes into bytes."""
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty size")
    units = {
        "": 1,
        "b": 1,
        "k": 1000, "kb": 1000, "kib": 1024,
        "m": 1000**2, "mb": 1000**2, "mib": 1024**2,
        "g": 1000**3, "gb": 1000**3, "gib": 1024**3,
        "t": 1000**4, "tb": 1000**4, "tib": 1024**4,
    }
    num = text
    suffix = ""
    for i, ch in enumerate(text):
        if not (ch.isdigit() or ch in ".-+"):
            num, suffix = text[:i], text[i:]
            break
    key = suffix.strip().lower()
    if key not in units:
        raise ValueError(f"unrecognised size {value!r}")
    return int(float(num) * units[key])


class Claim:
    """One process's hold on one emulated device.

    Normally created through :func:`process_claim` or used as a context
    manager via :func:`gpu`.
    """

    def __init__(
        self,
        device: int = 0,
        name: str | None = None,
        memory: str | int = 0,
        util: UtilSpec = "auto",
        kind: str = "compute",
        pid: int | None = None,
    ):
        self.device = int(device)
        self.pid = pid if pid is not None else os.getpid()
        self.name = name or _default_process_name()
        self.util = util
        self.kind = kind
        self._memory = parse_size(memory)
        self._lock = threading.Lock()
        self._path = claims_dir() / f"{self.pid}.{self.device}.{id(self):x}.json"
        self._open = False

    # -- lifecycle ---------------------------------------------------

    def open(self) -> "Claim":
        if self._open:
            return self
        if self._memory:
            _check_capacity(self.device, self._memory, exclude=self._path)
        self._open = True
        self._flush()
        atexit.register(self.close)
        return self

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "Claim":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- memory accounting -------------------------------------------

    @property
    def memory(self) -> int:
        return self._memory

    def set_memory(self, nbytes: str | int) -> None:
        """Set this claim's memory, raising if the device cannot fit it."""
        target = parse_size(nbytes)
        with self._lock:
            if target > self._memory:
                _check_capacity(
                    self.device, target, exclude=self._path, own_current=self._memory
                )
            self._memory = target
            if self._open:
                self._flush()

    def allocate(self, nbytes: str | int) -> None:
        """Grow this claim by ``nbytes``."""
        self.set_memory(self._memory + parse_size(nbytes))

    def free(self, nbytes: str | int) -> None:
        """Shrink this claim by ``nbytes``, never below zero."""
        self.set_memory(max(0, self._memory - parse_size(nbytes)))

    def set_util(self, util: UtilSpec) -> None:
        with self._lock:
            self.util = util
            if self._open:
                self._flush()

    # -- internals ---------------------------------------------------

    def _flush(self) -> None:
        payload = {
            "pid": self.pid,
            "device": self.device,
            "name": self.name,
            "memory": self._memory,
            "util": self.util,
            "type": self.kind,
            "created": time.time(),
        }
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, self._path)  # atomic, so the daemon never reads a partial claim
        try:
            os.chmod(self._path, 0o644)
        except OSError:
            pass


def _default_process_name() -> str:
    try:
        with open(f"/proc/{os.getpid()}/comm", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        import sys

        return Path(sys.argv[0]).name or "python"


def _capacity_lock():
    """Serialise capacity checks so two processes cannot both pass the same check."""
    path = claims_dir() / ".capacity.lock"
    fh = open(path, "a+b")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    return fh


def total_claimed(device: int, exclude: Path | None = None) -> int:
    total = 0
    for f in claims_dir().glob("*.json"):
        if exclude is not None and f == exclude:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if int(data.get("device", 0)) != device:
            continue
        if not _pid_alive(int(data.get("pid", -1))):
            continue
        total += int(data.get("memory", 0))
    return total


def device_capacity(device: int) -> int:
    """Usable memory on ``device``, i.e. total less the driver's own reservation."""
    reader = StateReader.try_open()
    if reader is not None:
        try:
            snap = reader.read()
            if device < len(snap.gpus):
                g = snap.gpus[device]
                return max(0, g.mem_total - g.mem_reserved)
        except ValueError:
            pass
        finally:
            reader.close()
    dev = spec.selected_device()
    return dev.mem_total_bytes - dev.mem_reserved_bytes


def _check_capacity(
    device: int, want: int, exclude: Path | None, own_current: int = 0
) -> None:
    """Refuse a claim that would not fit on the device.

    ``exclude`` is the caller's own claim file, left out of the total because
    ``want`` replaces it rather than adding to it. ``own_current`` is what that
    claim currently holds, which the message reports as "already allocated" -
    the same thing PyTorch's own OOM message means by the phrase.
    """
    lock = _capacity_lock()
    try:
        capacity = device_capacity(device)
        others = total_claimed(device, exclude=exclude)
        if others + want > capacity:
            free = max(0, capacity - others - own_current)
            raise OutOfMemoryError(
                f"CUDA out of memory. Tried to allocate {_fmt(want)} "
                f"(GPU {device}; {_fmt(capacity)} total capacity; "
                f"{_fmt(own_current)} already allocated; {_fmt(free)} free"
                + (f"; {_fmt(others)} used by other processes)." if others else ").")
            )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _fmt(nbytes: int) -> str:
    val = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if val < 1024 or unit == "GiB":
            return f"{val:.2f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= 1024
    return f"{val:.2f} GiB"


# -- module-level convenience ----------------------------------------

_process_claim: Claim | None = None
_process_claim_lock = threading.Lock()


def process_claim(
    device: int = 0,
    name: str | None = None,
    util: UtilSpec = "auto",
) -> Claim:
    """The single claim representing this process, created on first use.

    The torch shim and the Numba hook both route through this so that one
    Python process shows up as one entry in nvidia-smi, the way a real CUDA
    context does, rather than one per library.
    """
    global _process_claim
    with _process_claim_lock:
        if _process_claim is None:
            _process_claim = Claim(device=device, name=name, util=util).open()
        return _process_claim


def gpu(
    device: int = 0,
    memory: str | int = 0,
    util: UtilSpec = "auto",
    name: str | None = None,
) -> Claim:
    """Claim the GPU for the duration of a ``with`` block.

    >>> with gpuemu.gpu(memory="4GiB", util=90):
    ...     train()
    """
    return Claim(device=device, name=name, memory=memory, util=util)
