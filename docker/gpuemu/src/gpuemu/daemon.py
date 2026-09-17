"""gpuemud - keeps the emulated device's state file up to date.

Every tick this reads the claims directory, works out how hard each claiming
process is actually working, and moves the device's utilisation, clocks, power
and temperature towards where that says they should be. NVML readers see the
result.

The point of modelling any of this rather than just reporting the numbers a
claim asks for is that the readings have to behave like hardware to be worth
watching. Utilisation that snaps between 0 and 100 teaches nothing; a card that
ramps its clocks, draws more power as it works and warms up slowly afterwards
shows the learner the relationships that actually matter when they profile a
real job.

Where a claim says ``util: "auto"`` - the default - utilisation is derived from
the CPU time the claiming process and its children actually burn. That is what
makes the display feel connected to the learner's code: when their training
loop runs, the GPU is busy, and when it finishes, it idles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import spec
from .client import _pid_alive, claims_dir
from .spec import MIB
from .shm import PROC_COMPUTE, PROC_GRAPHICS, PROC_MPS, GPUState, Process, StateWriter

TICK_SECONDS = 0.2

_CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100

_KIND_BITS = {
    "compute": PROC_COMPUTE,
    "graphics": PROC_GRAPHICS,
    "mps": PROC_MPS,
}


def container_cpus() -> float:
    """How many CPUs this container may actually use.

    ``os.cpu_count()`` reports the host's CPUs, which inside a 2-core pod on a
    32-core node would make every process look idle. The cgroup quota is the
    number that matters, so prefer it and fall back only if it is absent.
    """
    for path, parse in (
        ("/sys/fs/cgroup/cpu.max", "v2"),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "v1"),
    ):
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        try:
            if parse == "v2":
                quota_s, period_s = text.split()
                if quota_s == "max":
                    break
                return max(0.1, int(quota_s) / int(period_s))
            quota = int(text)
            if quota <= 0:
                break
            period = int(
                Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8").strip()
            )
            return max(0.1, quota / period)
        except (ValueError, OSError):
            continue
    return float(os.cpu_count() or 1)


def _read_proc_tree_cpu(pids: set[int]) -> float:
    """Total CPU seconds used by ``pids``, in seconds."""
    total_ticks = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        # comm can contain spaces and parens, so split after the last ')'.
        close = data.rfind(b")")
        if close < 0:
            continue
        fields = data[close + 2 :].split()
        if len(fields) < 15:
            continue
        try:
            total_ticks += int(fields[11]) + int(fields[12])  # utime, stime
        except ValueError:
            continue
    return total_ticks / _CLOCK_TICKS


def _descendants(root: int) -> set[int]:
    """``root`` plus every process descended from it.

    Data loader workers and subprocesses spawned by a job do the work on its
    behalf, so their CPU time counts towards the job's utilisation.
    """
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return {root}
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        close = data.rfind(b")")
        if close < 0:
            continue
        fields = data[close + 2 :].split()
        if len(fields) < 2:
            continue
        try:
            children.setdefault(int(fields[1]), []).append(int(entry))
        except ValueError:
            continue

    out = {root}
    stack = [root]
    while stack:
        cur = stack.pop()
        for child in children.get(cur, ()):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


@dataclass
class _ClaimSample:
    """Bookkeeping for one claim between ticks."""

    cpu_seconds: float = 0.0
    wall: float = 0.0
    util: float = 0.0


@dataclass
class _DeviceSim:
    """Smoothed, continuously-varying state for one emulated device."""

    index: int
    dev: spec.DeviceSpec
    util: float = 0.0
    mem_util: float = 0.0
    power_w: float = 0.0
    temp_c: float = 0.0
    clock_gr: float = 0.0
    clock_mem: float = 0.0
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        self.power_w = self.dev.power_idle_w
        self.temp_c = self.dev.temp_idle_c
        self.clock_gr = self.dev.idle_clock_gr_mhz
        self.clock_mem = self.dev.idle_clock_mem_mhz
        self.rng.seed(1234 + self.index)

    def step(self, target_util: float, mem_fraction: float, dt: float) -> None:
        d = self.dev

        # Utilisation follows load quickly on the way up and decays a little
        # more slowly, which is what a real card's sampled counter looks like.
        alpha = 1.0 - math.exp(-dt / (0.25 if target_util > self.util else 0.6))
        self.util += (target_util - self.util) * alpha
        self.util = max(0.0, min(100.0, self.util))

        busy = self.util > 1.0

        # Memory controller activity tracks SM activity, scaled by how much of
        # the framebuffer is actually in play.
        mem_target = self.util * (0.35 + 0.45 * min(1.0, mem_fraction * 4)) if busy else 0.0
        self.mem_util += (mem_target - self.mem_util) * (1.0 - math.exp(-dt / 0.4))
        self.mem_util = max(0.0, min(100.0, self.mem_util))

        # Clocks boost under load and drop back to idle when there is nothing
        # to do. Real cards also trim clocks near the thermal limit.
        if busy:
            headroom = 1.0
            if self.temp_c > d.temp_slowdown_c - 8:
                headroom = max(0.7, 1.0 - (self.temp_c - (d.temp_slowdown_c - 8)) / 20.0)
            gr_target = d.max_clock_gr_mhz * headroom
            mem_target_clk = float(d.max_clock_mem_mhz)
        else:
            gr_target = float(d.idle_clock_gr_mhz)
            mem_target_clk = float(d.idle_clock_mem_mhz)
        clk_alpha = 1.0 - math.exp(-dt / 0.3)
        self.clock_gr += (gr_target - self.clock_gr) * clk_alpha
        self.clock_mem += (mem_target_clk - self.clock_mem) * clk_alpha

        # Power rises faster than linearly with utilisation, the way real
        # boards behave once clocks and voltage both climb.
        frac = self.util / 100.0
        p_target = d.power_idle_w + (d.power_limit_w - d.power_idle_w) * (frac**1.15)
        p_target += (self.mem_util / 100.0) * 0.08 * d.power_limit_w
        p_target = min(p_target, d.power_limit_w)
        self.power_w += (p_target - self.power_w) * (1.0 - math.exp(-dt / 0.5))
        if busy:
            self.power_w += self.rng.uniform(-0.015, 0.015) * d.power_limit_w
        self.power_w = max(d.power_idle_w * 0.8, min(d.power_limit_w, self.power_w))

        # Temperature lags power by a long time constant; this is the reading
        # that makes a finished job still look warm for a while afterwards.
        load = (self.power_w - d.power_idle_w) / max(1e-6, d.power_limit_w - d.power_idle_w)
        t_target = d.temp_idle_c + (d.temp_max_load_c - d.temp_idle_c) * max(0.0, load)
        self.temp_c += (t_target - self.temp_c) * (1.0 - math.exp(-dt / d.thermal_tau_s))

    def pstate(self) -> int:
        # P0 flat out, P2 working, P8 idle: the states an L4 actually reports.
        if self.util >= 30:
            return 0
        if self.util >= 2:
            return 2
        return 8


class Daemon:
    def __init__(self, state_file: Path | None = None, verbose: bool = False):
        self.dev = spec.selected_device()
        self.n_gpus = spec.device_count()
        self.sims = [_DeviceSim(i, self.dev) for i in range(self.n_gpus)]
        self.writer = StateWriter(state_file)
        self.samples: dict[str, _ClaimSample] = {}
        self.verbose = verbose
        self.gain = _float_env("GPUEMU_UTIL_GAIN", 1.0)
        self.cpus = container_cpus()
        self._running = True

    def stop(self, *_):
        self._running = False

    def run(self, once: bool = False) -> None:
        last = time.monotonic()
        # Publish an idle device immediately so nvidia-smi works right away
        # rather than only after the first full tick.
        self.tick(TICK_SECONDS)
        if once:
            return
        while self._running:
            time.sleep(TICK_SECONDS)
            now = time.monotonic()
            dt = min(2.0, max(1e-3, now - last))
            last = now
            try:
                self.tick(dt)
            except Exception as exc:  # keep the device alive through a bad claim
                if self.verbose:
                    print(f"gpuemud: tick failed: {exc}", file=sys.stderr)
        self.writer.close()

    # -- one simulation step -----------------------------------------

    def tick(self, dt: float) -> None:
        claims = self._load_claims()
        per_device: dict[int, list[tuple[dict, float]]] = {i: [] for i in range(self.n_gpus)}

        seen: set[str] = set()
        for key, claim in claims.items():
            seen.add(key)
            util = self._claim_util(key, claim, dt)
            dev_idx = min(self.n_gpus - 1, max(0, int(claim.get("device", 0))))
            per_device[dev_idx].append((claim, util))
        for stale in set(self.samples) - seen:
            del self.samples[stale]

        gpus: list[GPUState] = []
        for i in range(self.n_gpus):
            gpus.append(self._build_state(i, per_device[i], dt))

        self.writer.write(gpus, spec.DRIVER_VERSION, spec.CUDA_VERSION, spec.NVML_VERSION)

    def _load_claims(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        d = claims_dir()
        try:
            files = list(d.glob("*.json"))
        except OSError:
            return out
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            pid = int(data.get("pid", -1))
            if pid < 0 or not _pid_alive(pid):
                # The owning process is gone, so the memory it held is gone
                # too. Removing the file is what a driver cleaning up a dead
                # context amounts to here.
                try:
                    f.unlink()
                except OSError:
                    pass
                continue
            out[f.name] = data
        return out

    def _claim_util(self, key: str, claim: dict, dt: float) -> float:
        """Utilisation this claim contributes, 0-100."""
        want = claim.get("util", "auto")
        if isinstance(want, (int, float)):
            return max(0.0, min(100.0, float(want)))

        pid = int(claim["pid"])
        now = time.monotonic()
        try:
            cpu = _read_proc_tree_cpu(_descendants(pid))
        except OSError:
            return 0.0

        prev = self.samples.get(key)
        self.samples[key] = _ClaimSample(cpu_seconds=cpu, wall=now)
        if prev is None or now <= prev.wall:
            return 0.0

        busy_cores = (cpu - prev.cpu_seconds) / (now - prev.wall)
        # A process saturating its CPU allocation stands in for a saturated
        # GPU. Anything less shows up proportionally, which is the honest
        # reading: work that does not keep the device fed leaves it idle.
        util = 100.0 * busy_cores / max(0.5, self.cpus) * self.gain
        smoothed = 0.6 * util + 0.4 * (prev.util if prev else 0.0)
        self.samples[key].util = smoothed
        return max(0.0, min(100.0, smoothed))

    def _build_state(self, index: int, claims: list[tuple[dict, float]], dt: float) -> GPUState:
        d = self.dev
        sim = self.sims[index]

        claimed_mem = sum(int(c.get("memory", 0)) for c, _ in claims)
        raw_util = sum(u for _, u in claims)
        target_util = min(100.0, raw_util)

        mem_fraction = claimed_mem / max(1, d.mem_total_bytes)
        sim.step(target_util, mem_fraction, dt)

        bus_id, bus_id_legacy, domain, bus, dev_num = spec.make_bus_ids(index)

        # An idle card still reports a megabyte or so in use.
        mem_used = claimed_mem + 1 * MIB

        procs: list[Process] = []
        for claim, util in claims:
            # When claims together ask for more than the card can do, each gets
            # the share it would get from real time-slicing.
            share = (util / raw_util) * sim.util if raw_util > 0 else 0.0
            procs.append(
                Process(
                    pid=int(claim["pid"]),
                    name=str(claim.get("name", "python"))[:95],
                    type=_KIND_BITS.get(str(claim.get("type", "compute")), PROC_COMPUTE),
                    used_mem=int(claim.get("memory", 0)),
                    sm_util=int(round(share)),
                    mem_util=int(round(share * 0.6)),
                )
            )
        procs.sort(key=lambda p: p.pid)

        busy = sim.util > 1.0
        return GPUState(
            name=d.name,
            uuid=spec.make_uuid(index),
            serial=f"{1320923000000 + index:013d}",
            bus_id=bus_id,
            bus_id_legacy=bus_id_legacy,
            pci_domain=domain,
            pci_bus=bus,
            pci_device=dev_num,
            pci_device_id=d.pci_device_id,
            pci_subsys_id=d.pci_subsys_id,
            mem_total=d.mem_total_bytes,
            mem_used=mem_used,
            mem_reserved=d.mem_reserved_bytes,
            clock_gr=int(sim.clock_gr),
            clock_sm=int(sim.clock_gr),
            clock_mem=int(sim.clock_mem),
            clock_video=int(d.max_clock_video_mhz if busy else d.max_clock_video_mhz * 0.3),
            max_clock_gr=d.max_clock_gr_mhz,
            max_clock_sm=d.max_clock_sm_mhz,
            max_clock_mem=d.max_clock_mem_mhz,
            max_clock_video=d.max_clock_video_mhz,
            util_gpu=int(round(sim.util)),
            util_mem=int(round(sim.mem_util)),
            util_enc=0,
            util_dec=0,
            temp=int(round(sim.temp_c)),
            temp_shutdown=d.temp_shutdown_c,
            temp_slowdown=d.temp_slowdown_c,
            temp_gpu_max=d.temp_gpu_max_c,
            temp_mem_max=d.temp_mem_max_c,
            fan_speed=-1 if not d.has_fan else int(min(100, 30 + sim.util * 0.5)),
            power_mw=int(sim.power_w * 1000),
            power_limit_mw=int(d.power_limit_w * 1000),
            power_min_limit_mw=int(d.power_min_limit_w * 1000),
            power_max_limit_mw=int(d.power_limit_w * 1000),
            power_default_limit_mw=int(d.power_limit_w * 1000),
            # Link training drops an idle card to a narrower, slower link.
            pcie_gen=d.pcie_max_gen if busy else 1,
            pcie_width=d.pcie_max_width if busy else 8,
            pcie_max_gen=d.pcie_max_gen,
            pcie_max_width=d.pcie_max_width,
            pcie_tx_kbps=int(sim.util * 3000) if busy else 0,
            pcie_rx_kbps=int(sim.util * 9000) if busy else 0,
            pstate=sim.pstate(),
            persistence_mode=1 if _bool_env("GPUEMU_PERSISTENCE_MODE", True) else 0,
            compute_mode=0,
            mig_mode=0,
            ecc_mode=1,
            display_active=0,
            display_mode=0,
            ecc_corrected=0,
            ecc_uncorrected=0,
            cc_major=d.cc_major,
            cc_minor=d.cc_minor,
            processes=procs,
        )


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gpuemud",
        description="Maintain the emulated GPU's state file.",
    )
    ap.add_argument("--state-file", type=Path, default=None)
    ap.add_argument("--once", action="store_true", help="write one frame and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    daemon = Daemon(state_file=args.state_file, verbose=args.verbose)
    signal.signal(signal.SIGTERM, daemon.stop)
    signal.signal(signal.SIGINT, daemon.stop)

    if args.verbose:
        print(
            f"gpuemud: {daemon.n_gpus}x {daemon.dev.name}, "
            f"state={daemon.writer.path}, cpus={daemon.cpus:.1f}",
            file=sys.stderr,
        )
    daemon.run(once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
