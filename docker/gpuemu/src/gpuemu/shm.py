"""Reader and writer for the shared GPU state file.

This is the Python half of the layout declared in ``nvml/gpuemu_shm.h``. The C
shim mmaps the same bytes, so the two have to agree exactly; ``tests/test_abi.py``
compares the sizes and offsets computed here against the ones the C compiler
produces, which is what stops the two drifting apart silently.

Everything is little-endian and unaligned (``<`` in :mod:`struct`), which is what
``#pragma pack(1)`` gives on the C side.

Writers use a seqlock: bump ``seq`` to odd, write, bump to even. Readers retry
while it is odd or changes under them. One writer only (the daemon); any number
of readers.
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"GPUEMU1\x00"
ABI_VERSION = 1

MAX_GPUS = 8
MAX_PROCS = 64

PROC_COMPUTE = 1
PROC_GRAPHICS = 2
PROC_MPS = 4

# gpuemu_proc_t
_PROC_FMT = "<2IQ6I96s"
PROC_SIZE = struct.calcsize(_PROC_FMT)

# gpuemu_gpu_t, scalar part (everything before the process array). Spelled out
# one field per line so it lines up with the header; order must match exactly.
_GPU_FMT = (
    "<"
    "96s"   # name
    "80s"   # uuid
    "32s"   # serial
    "32s"   # bus_id
    "16s"   # bus_id_legacy
    "5I"    # pci: domain, bus, device, device_id, subsys_id
    "3Q"    # mem: total, used, reserved
    "4I"    # clocks: gr, sm, mem, video
    "4I"    # max clocks: gr, sm, mem, video
    "4I"    # util: gpu, mem, enc, dec
    "2I"    # enc/dec sampling period (us)
    "5I"    # temps: cur, shutdown, slowdown, gpu_max, mem_max
    "i"     # fan speed, -1 => not supported
    "5I"    # power: cur, limit, min_limit, max_limit, default_limit (mW)
    "4I"    # pcie: gen, width, max_gen, max_width
    "2I"    # pcie tx/rx (KB/s)
    "I"     # pstate
    "6I"    # persistence, compute, mig, ecc, display_active, display_mode
    "2Q"    # ecc corrected / uncorrected
    "2I"    # compute capability major, minor
    "I"     # n_procs
    "I"     # _pad
)
GPU_HEAD_SIZE = struct.calcsize(_GPU_FMT)
GPU_SIZE = GPU_HEAD_SIZE + MAX_PROCS * PROC_SIZE

# gpuemu_shm_t header
_SHM_FMT = (
    "<"
    "8s"    # magic
    "I"     # version
    "I"     # n_gpus
    "Q"     # seq
    "d"     # timestamp
    "32s"   # driver_version
    "16s"   # cuda_version
    "32s"   # nvml_version
    "I"     # _pad2
)
SHM_HEAD_SIZE = struct.calcsize(_SHM_FMT)
SHM_SIZE = SHM_HEAD_SIZE + MAX_GPUS * GPU_SIZE

SEQ_OFFSET = struct.calcsize("<8sII")


def default_state_file() -> Path:
    """Where the state file lives.

    ``/run/gpuemu`` when it is writable, which is the case inside the session
    container, otherwise a per-user path so the emulator still works when
    someone runs it on a laptop for development.
    """
    env = os.environ.get("GPUEMU_STATE_FILE")
    if env:
        return Path(env)
    run = Path("/run/gpuemu")
    try:
        run.mkdir(parents=True, exist_ok=True)
        if os.access(run, os.W_OK):
            return run / "state.bin"
    except OSError:
        pass
    base = Path(os.environ.get("XDG_RUNTIME_DIR") or Path.home() / ".cache")
    d = base / "gpuemu"
    d.mkdir(parents=True, exist_ok=True)
    return d / "state.bin"


def _cstr(value: str, size: int) -> bytes:
    """Encode to a fixed-width NUL-terminated field, truncating if needed."""
    raw = value.encode("utf-8", "replace")[: size - 1]
    return raw + b"\x00" * (size - len(raw))


def _pystr(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")


@dataclass
class Process:
    """One process holding memory on the emulated device."""

    pid: int
    name: str = ""
    type: int = PROC_COMPUTE
    used_mem: int = 0
    sm_util: int = 0
    mem_util: int = 0
    enc_util: int = 0
    dec_util: int = 0
    gi_id: int = 0xFFFFFFFF
    ci_id: int = 0xFFFFFFFF

    def pack(self) -> bytes:
        return struct.pack(
            _PROC_FMT,
            self.pid,
            self.type,
            self.used_mem,
            self.sm_util,
            self.mem_util,
            self.enc_util,
            self.dec_util,
            self.gi_id,
            self.ci_id,
            _cstr(self.name, 96),
        )

    @classmethod
    def unpack(cls, raw: bytes) -> "Process":
        (
            pid,
            type_,
            used_mem,
            sm,
            mem,
            enc,
            dec,
            gi,
            ci,
            name,
        ) = struct.unpack(_PROC_FMT, raw)
        return cls(
            pid=pid,
            type=type_,
            used_mem=used_mem,
            sm_util=sm,
            mem_util=mem,
            enc_util=enc,
            dec_util=dec,
            gi_id=gi,
            ci_id=ci,
            name=_pystr(name),
        )


@dataclass
class GPUState:
    """Everything the shim can report about one emulated device."""

    name: str = "NVIDIA L4"
    uuid: str = ""
    serial: str = ""
    bus_id: str = ""
    bus_id_legacy: str = ""

    pci_domain: int = 0
    pci_bus: int = 0
    pci_device: int = 0
    pci_device_id: int = 0
    pci_subsys_id: int = 0

    mem_total: int = 0
    mem_used: int = 0
    mem_reserved: int = 0

    clock_gr: int = 0
    clock_sm: int = 0
    clock_mem: int = 0
    clock_video: int = 0
    max_clock_gr: int = 0
    max_clock_sm: int = 0
    max_clock_mem: int = 0
    max_clock_video: int = 0

    util_gpu: int = 0
    util_mem: int = 0
    util_enc: int = 0
    util_dec: int = 0
    enc_sampling_us: int = 167000
    dec_sampling_us: int = 167000

    temp: int = 0
    temp_shutdown: int = 95
    temp_slowdown: int = 92
    temp_gpu_max: int = 90
    temp_mem_max: int = 95

    fan_speed: int = -1

    power_mw: int = 0
    power_limit_mw: int = 0
    power_min_limit_mw: int = 0
    power_max_limit_mw: int = 0
    power_default_limit_mw: int = 0

    pcie_gen: int = 1
    pcie_width: int = 8
    pcie_max_gen: int = 4
    pcie_max_width: int = 16
    pcie_tx_kbps: int = 0
    pcie_rx_kbps: int = 0

    pstate: int = 8

    persistence_mode: int = 0
    compute_mode: int = 0
    mig_mode: int = 0
    ecc_mode: int = 1
    display_active: int = 0
    display_mode: int = 0

    ecc_corrected: int = 0
    ecc_uncorrected: int = 0

    cc_major: int = 8
    cc_minor: int = 9

    processes: list[Process] = field(default_factory=list)

    @property
    def mem_free(self) -> int:
        return max(0, self.mem_total - self.mem_used)

    def pack(self) -> bytes:
        procs = self.processes[:MAX_PROCS]
        head = struct.pack(
            _GPU_FMT,
            _cstr(self.name, 96),
            _cstr(self.uuid, 80),
            _cstr(self.serial, 32),
            _cstr(self.bus_id, 32),
            _cstr(self.bus_id_legacy, 16),
            self.pci_domain,
            self.pci_bus,
            self.pci_device,
            self.pci_device_id,
            self.pci_subsys_id,
            self.mem_total,
            self.mem_used,
            self.mem_reserved,
            self.clock_gr,
            self.clock_sm,
            self.clock_mem,
            self.clock_video,
            self.max_clock_gr,
            self.max_clock_sm,
            self.max_clock_mem,
            self.max_clock_video,
            self.util_gpu,
            self.util_mem,
            self.util_enc,
            self.util_dec,
            self.enc_sampling_us,
            self.dec_sampling_us,
            self.temp,
            self.temp_shutdown,
            self.temp_slowdown,
            self.temp_gpu_max,
            self.temp_mem_max,
            self.fan_speed,
            self.power_mw,
            self.power_limit_mw,
            self.power_min_limit_mw,
            self.power_max_limit_mw,
            self.power_default_limit_mw,
            self.pcie_gen,
            self.pcie_width,
            self.pcie_max_gen,
            self.pcie_max_width,
            self.pcie_tx_kbps,
            self.pcie_rx_kbps,
            self.pstate,
            self.persistence_mode,
            self.compute_mode,
            self.mig_mode,
            self.ecc_mode,
            self.display_active,
            self.display_mode,
            self.ecc_corrected,
            self.ecc_uncorrected,
            self.cc_major,
            self.cc_minor,
            len(procs),
            0,
        )
        body = b"".join(p.pack() for p in procs)
        body += b"\x00" * ((MAX_PROCS - len(procs)) * PROC_SIZE)
        return head + body

    @classmethod
    def unpack(cls, raw: bytes) -> "GPUState":
        vals = struct.unpack(_GPU_FMT, raw[:GPU_HEAD_SIZE])
        s = cls(
            name=_pystr(vals[0]),
            uuid=_pystr(vals[1]),
            serial=_pystr(vals[2]),
            bus_id=_pystr(vals[3]),
            bus_id_legacy=_pystr(vals[4]),
            pci_domain=vals[5],
            pci_bus=vals[6],
            pci_device=vals[7],
            pci_device_id=vals[8],
            pci_subsys_id=vals[9],
            mem_total=vals[10],
            mem_used=vals[11],
            mem_reserved=vals[12],
            clock_gr=vals[13],
            clock_sm=vals[14],
            clock_mem=vals[15],
            clock_video=vals[16],
            max_clock_gr=vals[17],
            max_clock_sm=vals[18],
            max_clock_mem=vals[19],
            max_clock_video=vals[20],
            util_gpu=vals[21],
            util_mem=vals[22],
            util_enc=vals[23],
            util_dec=vals[24],
            enc_sampling_us=vals[25],
            dec_sampling_us=vals[26],
            temp=vals[27],
            temp_shutdown=vals[28],
            temp_slowdown=vals[29],
            temp_gpu_max=vals[30],
            temp_mem_max=vals[31],
            fan_speed=vals[32],
            power_mw=vals[33],
            power_limit_mw=vals[34],
            power_min_limit_mw=vals[35],
            power_max_limit_mw=vals[36],
            power_default_limit_mw=vals[37],
            pcie_gen=vals[38],
            pcie_width=vals[39],
            pcie_max_gen=vals[40],
            pcie_max_width=vals[41],
            pcie_tx_kbps=vals[42],
            pcie_rx_kbps=vals[43],
            pstate=vals[44],
            persistence_mode=vals[45],
            compute_mode=vals[46],
            mig_mode=vals[47],
            ecc_mode=vals[48],
            display_active=vals[49],
            display_mode=vals[50],
            ecc_corrected=vals[51],
            ecc_uncorrected=vals[52],
            cc_major=vals[53],
            cc_minor=vals[54],
        )
        n_procs = min(vals[55], MAX_PROCS)
        for i in range(n_procs):
            off = GPU_HEAD_SIZE + i * PROC_SIZE
            s.processes.append(Process.unpack(raw[off : off + PROC_SIZE]))
        return s


@dataclass
class Snapshot:
    """A consistent read of the whole state file."""

    driver_version: str
    cuda_version: str
    nvml_version: str
    timestamp: float
    gpus: list[GPUState]


class StateWriter:
    """The daemon's handle on the state file. Single writer, seqlock protected."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_state_file()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create at full size first; mmap cannot grow a file under us.
        with open(self.path, "r+b" if self.path.exists() else "w+b") as fh:
            fh.truncate(SHM_SIZE)
        self._fh = open(self.path, "r+b")
        self._mm = mmap.mmap(self._fh.fileno(), SHM_SIZE, access=mmap.ACCESS_WRITE)
        self._seq = 0
        # Readable by every user in the container, not just the daemon's owner.
        try:
            os.chmod(self.path, 0o644)
        except OSError:
            pass

    def write(
        self,
        gpus: list[GPUState],
        driver_version: str,
        cuda_version: str,
        nvml_version: str,
    ) -> None:
        payload = b"".join(g.pack() for g in gpus[:MAX_GPUS])
        payload += b"\x00" * ((MAX_GPUS - len(gpus)) * GPU_SIZE)

        # Odd seq marks the update in progress, so a reader that catches us
        # mid-write knows to retry instead of trusting a torn frame.
        self._seq += 1
        self._write_seq(self._seq)
        header = struct.pack(
            _SHM_FMT,
            MAGIC,
            ABI_VERSION,
            len(gpus),
            self._seq,
            time.time(),
            _cstr(driver_version, 32),
            _cstr(cuda_version, 16),
            _cstr(nvml_version, 32),
            0,
        )
        self._mm[0:SHM_HEAD_SIZE] = header
        self._mm[SHM_HEAD_SIZE : SHM_HEAD_SIZE + len(payload)] = payload
        self._seq += 1
        self._write_seq(self._seq)
        self._mm.flush()

    def _write_seq(self, value: int) -> None:
        self._mm[SEQ_OFFSET : SEQ_OFFSET + 8] = struct.pack("<Q", value)

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fh.close()

    def __enter__(self) -> "StateWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class StateReader:
    """Read-only view for nvidia-smi and the Python client."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_state_file()
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), SHM_SIZE, access=mmap.ACCESS_READ)

    @classmethod
    def try_open(cls, path: Path | str | None = None) -> "StateReader | None":
        """Open the state file, or return None if no daemon has made one yet."""
        try:
            return cls(path)
        except (OSError, ValueError):
            return None

    def read(self, retries: int = 16) -> Snapshot:
        for _ in range(retries):
            seq1 = struct.unpack("<Q", self._mm[SEQ_OFFSET : SEQ_OFFSET + 8])[0]
            if seq1 & 1:
                time.sleep(0.001)
                continue
            raw = self._mm[0:SHM_SIZE]
            seq2 = struct.unpack("<Q", raw[SEQ_OFFSET : SEQ_OFFSET + 8])[0]
            if seq1 != seq2:
                continue
            return self._parse(raw)
        # Consistently busy writer: take what we have rather than hanging. At
        # worst one frame mixes two updates, on data that refreshes anyway.
        return self._parse(self._mm[0:SHM_SIZE])

    @staticmethod
    def _parse(raw: bytes) -> Snapshot:
        magic, version, n_gpus, _seq, ts, driver, cuda, nvml, _pad = struct.unpack(
            _SHM_FMT, raw[:SHM_HEAD_SIZE]
        )
        if magic != MAGIC:
            raise ValueError(f"not a gpuemu state file (magic {magic!r})")
        if version != ABI_VERSION:
            raise ValueError(f"state file ABI v{version}, expected v{ABI_VERSION}")
        gpus = []
        for i in range(min(n_gpus, MAX_GPUS)):
            off = SHM_HEAD_SIZE + i * GPU_SIZE
            gpus.append(GPUState.unpack(raw[off : off + GPU_SIZE]))
        return Snapshot(
            driver_version=_pystr(driver),
            cuda_version=_pystr(cuda),
            nvml_version=_pystr(nvml),
            timestamp=ts,
            gpus=gpus,
        )

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fh.close()

    def __enter__(self) -> "StateReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
