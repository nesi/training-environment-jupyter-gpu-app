"""nvidia-smi, reimplemented against the emulated device.

The layout here is copied from the real tool's 550-series output, column for
column. That precision is the point: learners are told to run nvidia-smi and
read particular boxes, workshop notes contain screenshots, and scripts parse
``--query-gpu`` output. Anything that looked approximately right would break all
three.

Supported: the default table, ``-L``, ``-q``, ``--query-gpu``/``--query-compute-apps``
with ``--format=csv``, ``pmon``, ``dmon``, ``-l``, ``-i``. Anything that would
change device state (``-pm``, ``-pl``, ``-r``, ``-c``) is accepted and reported
as unsupported rather than silently ignored, so a learner following a tutorial
gets a clear answer instead of a lie.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from . import spec
from .shm import GPUState, Snapshot, StateReader

MIB = 1024 * 1024

# The 550-series table is 91 characters wide between the outer '+'s.
_W = 91


def _load(retries: int = 1) -> Snapshot:
    reader = StateReader.try_open()
    if reader is None:
        raise SystemExit(
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n"
            "The gpuemu daemon does not appear to be running; start it with 'gpuemud &' "
            "or run 'gpuemu-ctl status' to check."
        )
    try:
        return reader.read()
    finally:
        reader.close()


def _visible_pairs(snap: Snapshot) -> list[tuple[int, GPUState]]:
    """The devices this process is allowed to see.

    Real nvidia-smi ignores CUDA_VISIBLE_DEVICES, but on a cluster it is run
    inside a cgroup that does not contain the device nodes, so a job that did
    not request a GPU gets "No devices were found" anyway. We reproduce that
    outcome from the variable, because the outcome is the thing worth teaching
    and we have no cgroups to do it with.
    """
    from .client import visible_devices

    vis = visible_devices()
    if vis is None:
        return list(enumerate(snap.gpus))
    return [(i, g) for i, g in enumerate(snap.gpus) if i in vis]


def _select(snap: Snapshot, spec_ids: str | None) -> list[tuple[int, GPUState]]:
    pairs = _visible_pairs(snap)
    if not pairs:
        raise SystemExit("No devices were found")
    if not spec_ids:
        return pairs
    wanted: list[tuple[int, GPUState]] = []
    for token in spec_ids.split(","):
        token = token.strip()
        if not token:
            continue
        match = None
        if token.isdigit():
            idx = int(token)
            # Only among visible devices: -i 0 must fail if 0 is not ours.
            match = next((p for p in pairs if p[0] == idx), None)
        else:
            for idx, g in pairs:
                if token in (g.uuid, g.bus_id, g.bus_id_legacy):
                    match = (idx, g)
                    break
        if match is None:
            raise SystemExit(f"No devices were found with the id {token}")
        wanted.append(match)
    return wanted


# ---------------------------------------------------------------- tables


def _fmt_mem(nbytes: int) -> str:
    return f"{nbytes // MIB}MiB"


# Column widths of the three panels in the device table. They have to be these
# exact numbers: the header strings below are copied verbatim from nvidia-smi,
# and everything else is aligned against them.
_C1, _C2, _C3 = 41, 24, 22


def render_table(snap: Snapshot, gpus: list[tuple[int, GPUState]]) -> str:
    # Real nvidia-smi pads the date to a fixed width, trailing blanks and all.
    stamp = datetime.now().strftime("%a %b %e %H:%M:%S %Y")
    out: list[str] = [f"{stamp}       ", ""]

    out.append("+" + "-" * (_W - 2) + "+")
    # Field widths chosen so "Driver Version:" starts at column 35 and
    # "CUDA Version:" at column 80, which is where the real tool puts them.
    title = (
        f" NVIDIA-SMI {snap.driver_version:<23}"
        f"Driver Version: {snap.driver_version:<15}"
        f"CUDA Version: {snap.cuda_version:<9}"
    )
    out.append("|" + title[: _W - 2].ljust(_W - 2) + "|")
    # Note the asymmetry: this rule ends with '+' while the '=' rule below ends
    # with '|'. That is what nvidia-smi prints, odd as it looks.
    out.append("|" + "-" * _C1 + "+" + "-" * _C2 + "+" + "-" * _C3 + "+")
    out.append(
        "| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |"
    )
    out.append(
        "| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |"
    )
    out.append(
        "|                                         |                        |               MIG M. |"
    )
    out.append("|" + "=" * _C1 + "+" + "=" * _C2 + "+" + "=" * _C3 + "|")

    for idx, g in gpus:
        persistence = "On" if g.persistence_mode else "Off"
        ecc = str(g.ecc_uncorrected) if g.ecc_mode else "N/A"
        disp = "On" if g.display_active else "Off"

        # Row 1: index and model, PCI address, ECC error count.
        c1 = f"{idx:>4}  {g.name:<30}{persistence:>3}  "
        c2 = f"   {g.bus_id:<16} {disp:>3} "
        c3 = f"{ecc:>21} "
        out.append("|" + c1[:_C1].ljust(_C1) + "|" + c2[:_C2].ljust(_C2) + "|" + c3[:_C3].ljust(_C3) + "|")

        # Row 2: cooling and thermals, power, memory, utilisation.
        fan = "N/A" if g.fan_speed < 0 else f"{g.fan_speed}%"
        power = f"{g.power_mw / 1000:.0f}W / {g.power_limit_mw / 1000:>3.0f}W"
        r1 = f" {fan:>3}{str(g.temp) + 'C':>6}{'P' + str(g.pstate):>6}{power:>23}  "
        r2 = f" {_fmt_mem(g.mem_used):>10} / {_fmt_mem(g.mem_total):>9} "
        r3 = f"{str(g.util_gpu) + '%':>8}{'Default':>13} "
        out.append("|" + r1[:_C1].ljust(_C1) + "|" + r2[:_C2].ljust(_C2) + "|" + r3[:_C3].ljust(_C3) + "|")

        # Row 3: MIG mode only, the rest blank.
        mig = "Disabled" if g.mig_mode else "N/A"
        out.append("|" + " " * _C1 + "|" + " " * _C2 + "|" + f"{mig:>21} "[:_C3].ljust(_C3) + "|")
        out.append("+" + "-" * _C1 + "+" + "-" * _C2 + "+" + "-" * _C3 + "+")

    out.append("")
    out.append("+" + "-" * (_W - 2) + "+")
    out.append("| Processes:".ljust(_W - 1) + "|")
    out.append(
        "|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |"
    )
    out.append(
        "|        ID   ID                                                               Usage      |"
    )
    out.append("|" + "=" * (_W - 2) + "|")

    any_proc = False
    for idx, g in gpus:
        for p in g.processes:
            any_proc = True
            gi = "N/A" if p.gi_id == 0xFFFFFFFF else str(p.gi_id)
            ci = "N/A" if p.ci_id == 0xFFFFFFFF else str(p.ci_id)
            kind = "C" if p.type & 1 else "G"
            row = (
                f"{idx:>5}{gi:>6}{ci:>5}{p.pid:>10}{kind:>7}   "
                f"{p.name[:43]:<43}{_fmt_mem(p.used_mem):>9} "
            )
            out.append("|" + row[: _W - 2].ljust(_W - 2) + "|")
    if not any_proc:
        out.append("|  No running processes found".ljust(_W - 1) + "|")
    out.append("+" + "-" * (_W - 2) + "+")
    return "\n".join(out)


def render_list(gpus: list[tuple[int, GPUState]]) -> str:
    return "\n".join(f"GPU {i}: {g.name} (UUID: {g.uuid})" for i, g in gpus)


# ---------------------------------------------------------------- queries

# Maps --query-gpu field names to (value, unit). Units are stripped when
# --format=nounits is given, which is how scripts consume this.
def _gpu_fields(idx: int, g: GPUState, snap: Snapshot) -> dict[str, tuple[str, str]]:
    return {
        "index": (str(idx), ""),
        "name": (g.name, ""),
        "gpu_name": (g.name, ""),
        "uuid": (g.uuid, ""),
        "gpu_uuid": (g.uuid, ""),
        "serial": (g.serial, ""),
        "pci.bus_id": (g.bus_id, ""),
        "gpu_bus_id": (g.bus_id, ""),
        "driver_version": (snap.driver_version, ""),
        "compute_cap": (f"{g.cc_major}.{g.cc_minor}", ""),
        "pstate": (f"P{g.pstate}", ""),
        "persistence_mode": ("Enabled" if g.persistence_mode else "Disabled", ""),
        "compute_mode": ("Default", ""),
        "mig.mode.current": ("Disabled" if g.mig_mode else "N/A", ""),
        "utilization.gpu": (str(g.util_gpu), "%"),
        "utilization.memory": (str(g.util_mem), "%"),
        "memory.total": (str(g.mem_total // MIB), "MiB"),
        "memory.used": (str(g.mem_used // MIB), "MiB"),
        "memory.free": (str(g.mem_free // MIB), "MiB"),
        "memory.reserved": (str(g.mem_reserved // MIB), "MiB"),
        "temperature.gpu": (str(g.temp), ""),
        "temperature.memory": ("N/A", ""),
        "power.draw": (f"{g.power_mw / 1000:.2f}", "W"),
        "power.limit": (f"{g.power_limit_mw / 1000:.2f}", "W"),
        "power.max_limit": (f"{g.power_max_limit_mw / 1000:.2f}", "W"),
        "power.min_limit": (f"{g.power_min_limit_mw / 1000:.2f}", "W"),
        "enforced.power.limit": (f"{g.power_limit_mw / 1000:.2f}", "W"),
        "clocks.sm": (str(g.clock_sm), "MHz"),
        "clocks.current.sm": (str(g.clock_sm), "MHz"),
        "clocks.gr": (str(g.clock_gr), "MHz"),
        "clocks.current.graphics": (str(g.clock_gr), "MHz"),
        "clocks.mem": (str(g.clock_mem), "MHz"),
        "clocks.current.memory": (str(g.clock_mem), "MHz"),
        "clocks.max.sm": (str(g.max_clock_sm), "MHz"),
        "clocks.max.graphics": (str(g.max_clock_gr), "MHz"),
        "clocks.max.memory": (str(g.max_clock_mem), "MHz"),
        "fan.speed": ("N/A" if g.fan_speed < 0 else str(g.fan_speed), "%"),
        "pcie.link.gen.current": (str(g.pcie_gen), ""),
        "pcie.link.gen.max": (str(g.pcie_max_gen), ""),
        "pcie.link.width.current": (str(g.pcie_width), ""),
        "pcie.link.width.max": (str(g.pcie_max_width), ""),
        "ecc.errors.corrected.volatile.total": (str(g.ecc_corrected), ""),
        "ecc.errors.uncorrected.volatile.total": (str(g.ecc_uncorrected), ""),
        "count": (str(len(snap.gpus)), ""),
        "display_active": ("Disabled", ""),
        "display_mode": ("Disabled", ""),
    }


def render_query_gpu(
    snap: Snapshot,
    gpus: list[tuple[int, GPUState]],
    fields: list[str],
    header: bool,
    units: bool,
) -> str:
    lines: list[str] = []
    if header:
        cols = []
        for f in fields:
            sample = _gpu_fields(0, snap.gpus[0], snap).get(f)
            unit = sample[1] if sample else ""
            cols.append(f"{f} [{unit}]" if (units and unit) else f)
        lines.append(", ".join(cols))
    for idx, g in gpus:
        table = _gpu_fields(idx, g, snap)
        row = []
        for f in fields:
            if f not in table:
                raise SystemExit(f'Field "{f}" is not supported by this emulator')
            value, unit = table[f]
            row.append(f"{value} {unit}" if (units and unit and value != "N/A") else value)
        lines.append(", ".join(row))
    return "\n".join(lines)


def render_query_apps(
    gpus: list[tuple[int, GPUState]], fields: list[str], header: bool, units: bool
) -> str:
    lines: list[str] = []
    if header:
        cols = [f"{f} [MiB]" if (units and "memory" in f) else f for f in fields]
        lines.append(", ".join(cols))
    for idx, g in gpus:
        for p in g.processes:
            table = {
                "pid": str(p.pid),
                "process_name": p.name,
                "gpu_uuid": g.uuid,
                "gpu_name": g.name,
                "gpu_bus_id": g.bus_id,
                "gpu_serial": g.serial,
                "used_gpu_memory": str(p.used_mem // MIB),
                "used_memory": str(p.used_mem // MIB),
            }
            row = []
            for f in fields:
                if f not in table:
                    raise SystemExit(f'Field "{f}" is not supported by this emulator')
                v = table[f]
                row.append(f"{v} MiB" if (units and "memory" in f) else v)
            lines.append(", ".join(row))
    return "\n".join(lines)


def _architecture_for(name: str) -> str:
    """Look the architecture up from the reported model name.

    Matching on the name rather than reading GPUEMU_DEVICE means this stays
    right even if a caller's environment differs from the daemon's.
    """
    for dev in spec.DEVICES.values():
        if dev.name == name:
            return dev.architecture
    return "Unknown"


def render_query_full(snap: Snapshot, gpus: list[tuple[int, GPUState]]) -> str:
    stamp = datetime.now().strftime("%a %b %e %H:%M:%S %Y")
    out = [
        "=============NVSMI LOG=============",
        "",
        f"Timestamp                                 : {stamp}",
        f"Driver Version                            : {snap.driver_version}",
        f"CUDA Version                              : {snap.cuda_version}",
        "",
        f"Attached GPUs                             : {len(snap.gpus)}",
    ]
    for idx, g in gpus:
        fan = "N/A" if g.fan_speed < 0 else f"{g.fan_speed} %"
        out += [
            f"GPU {g.bus_id}",
            f"    Product Name                          : {g.name}",
            f"    Product Brand                         : NVIDIA",
            f"    Product Architecture                  : {_architecture_for(g.name)}",
            f"    Persistence Mode                      : {'Enabled' if g.persistence_mode else 'Disabled'}",
            f"    MIG Mode",
            f"        Current                           : {'Enabled' if g.mig_mode else 'N/A'}",
            f"        Pending                           : {'Enabled' if g.mig_mode else 'N/A'}",
            f"    Serial Number                         : {g.serial}",
            f"    GPU UUID                              : {g.uuid}",
            f"    Minor Number                          : {idx}",
            f"    GPU Virtualization Mode",
            f"        Virtualization Mode               : None",
            f"    GPU Reset Status",
            f"        Reset Required                    : No",
            f"    PCI",
            f"        Bus                               : 0x{g.pci_bus:02X}",
            f"        Device                            : 0x{g.pci_device:02X}",
            f"        Domain                            : 0x{g.pci_domain:04X}",
            f"        Device Id                         : 0x{g.pci_device_id:08X}",
            f"        Bus Id                            : {g.bus_id}",
            f"        GPU Link Info",
            f"            PCIe Generation",
            f"                Max                       : {g.pcie_max_gen}",
            f"                Current                   : {g.pcie_gen}",
            f"            Link Width",
            f"                Max                       : {g.pcie_max_width}x",
            f"                Current                   : {g.pcie_width}x",
            f"    Fan Speed                             : {fan}",
            f"    Performance State                     : P{g.pstate}",
            f"    FB Memory Usage",
            f"        Total                             : {g.mem_total // MIB} MiB",
            f"        Reserved                          : {g.mem_reserved // MIB} MiB",
            f"        Used                              : {g.mem_used // MIB} MiB",
            f"        Free                              : {g.mem_free // MIB} MiB",
            f"    Compute Mode                          : Default",
            f"    Utilization",
            f"        Gpu                               : {g.util_gpu} %",
            f"        Memory                            : {g.util_mem} %",
            f"        Encoder                           : {g.util_enc} %",
            f"        Decoder                           : {g.util_dec} %",
            f"    ECC Mode",
            f"        Current                           : {'Enabled' if g.ecc_mode else 'Disabled'}",
            f"        Pending                           : {'Enabled' if g.ecc_mode else 'Disabled'}",
            f"    Temperature",
            f"        GPU Current Temp                  : {g.temp} C",
            f"        GPU Shutdown Temp                 : {g.temp_shutdown} C",
            f"        GPU Slowdown Temp                 : {g.temp_slowdown} C",
            f"        GPU Max Operating Temp            : {g.temp_gpu_max} C",
            f"    GPU Power Readings",
            f"        Power Draw                        : {g.power_mw / 1000:.2f} W",
            f"        Current Power Limit               : {g.power_limit_mw / 1000:.2f} W",
            f"        Default Power Limit               : {g.power_default_limit_mw / 1000:.2f} W",
            f"        Min Power Limit                   : {g.power_min_limit_mw / 1000:.2f} W",
            f"        Max Power Limit                   : {g.power_max_limit_mw / 1000:.2f} W",
            f"    Clocks",
            f"        Graphics                          : {g.clock_gr} MHz",
            f"        SM                                : {g.clock_sm} MHz",
            f"        Memory                            : {g.clock_mem} MHz",
            f"        Video                             : {g.clock_video} MHz",
            f"    Max Clocks",
            f"        Graphics                          : {g.max_clock_gr} MHz",
            f"        SM                                : {g.max_clock_sm} MHz",
            f"        Memory                            : {g.max_clock_mem} MHz",
            f"        Video                             : {g.max_clock_video} MHz",
            f"    Compute Capability                    : {g.cc_major}.{g.cc_minor}",
            f"    Processes                             : {'None' if not g.processes else ''}",
        ]
        for p in g.processes:
            out += [
                f"        GPU instance ID                   : N/A",
                f"        Compute instance ID               : N/A",
                f"        Process ID                        : {p.pid}",
                f"            Type                          : C",
                f"            Name                          : {p.name}",
                f"            Used GPU Memory               : {p.used_mem // MIB} MiB",
            ]
        out.append("")
    return "\n".join(out)


def render_pmon(snap: Snapshot, gpus: list[tuple[int, GPUState]]) -> str:
    out = [
        "# gpu        pid  type    sm   mem   enc   dec   command",
        "# Idx          #   C/G     %     %     %     %   name",
    ]
    for idx, g in gpus:
        if not g.processes:
            out.append(f"{idx:>5} {'-':>10} {'-':>5} {'-':>5} {'-':>5} {'-':>5} {'-':>5}   -")
        for p in g.processes:
            kind = "C" if p.type & 1 else "G"
            out.append(
                f"{idx:>5} {p.pid:>10} {kind:>5} {p.sm_util:>5} {p.mem_util:>5} "
                f"{p.enc_util:>5} {p.dec_util:>5}   {p.name[:16]}"
            )
    return "\n".join(out)


def render_dmon(gpus: list[tuple[int, GPUState]]) -> str:
    out = [
        "# gpu   pwr gtemp mtemp    sm   mem   enc   dec  mclk  pclk",
        "# Idx     W     C     C     %     %     %     %   MHz   MHz",
    ]
    for idx, g in gpus:
        out.append(
            f"{idx:>5} {g.power_mw // 1000:>5} {g.temp:>5} {'-':>5} "
            f"{g.util_gpu:>5} {g.util_mem:>5} {g.util_enc:>5} {g.util_dec:>5} "
            f"{g.clock_mem:>5} {g.clock_sm:>5}"
        )
    return "\n".join(out)


# ---------------------------------------------------------------- entry


def _split_fields(raw: str) -> list[str]:
    return [f.strip() for f in raw.split(",") if f.strip()]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Subcommands come before the flags, matching the real tool.
    sub = None
    if argv and argv[0] in {"pmon", "dmon", "topo", "nvlink", "mig"}:
        sub = argv.pop(0)

    ap = argparse.ArgumentParser(prog="nvidia-smi", add_help=False)
    ap.add_argument("-h", "--help", action="store_true")
    ap.add_argument("-L", "--list-gpus", action="store_true")
    ap.add_argument("-q", "--query", action="store_true")
    ap.add_argument("-i", "--id", default=None)
    ap.add_argument("-l", "--loop", nargs="?", const=5, type=int, default=None)
    ap.add_argument("-c", "--count", type=int, default=None)
    ap.add_argument("--query-gpu", default=None)
    ap.add_argument("--query-compute-apps", default=None)
    ap.add_argument("--format", default=None)
    ap.add_argument("--version", action="store_true")
    # Accepted so the failure is an honest message rather than a silent no-op.
    ap.add_argument("-pm", "--persistence-mode", default=None)
    ap.add_argument("-pl", "--power-limit", default=None)
    ap.add_argument("-r", "--gpu-reset", action="store_true")
    ap.add_argument("-ac", "--applications-clocks", default=None)
    args, unknown = ap.parse_known_args(argv)

    if args.help:
        print(__doc__.strip())
        return 0

    if args.version:
        snap = _load()
        print(f"NVIDIA-SMI version  : {snap.driver_version}")
        print(f"NVML version        : {snap.nvml_version}")
        print(f"DRIVER version      : {snap.driver_version}")
        print(f"CUDA Version        : {snap.cuda_version}")
        return 0

    for flag, value in (
        ("-pm", args.persistence_mode),
        ("-pl", args.power_limit),
        ("-ac", args.applications_clocks),
    ):
        if value is not None:
            print(
                f"{flag}: changing device settings is not supported by the GPU emulator.\n"
                "The device is simulated, so there is nothing to configure.",
                file=sys.stderr,
            )
            return 3
    if args.gpu_reset:
        print(
            "-r: resetting is not supported by the GPU emulator. "
            "Restart the daemon with 'gpuemu-ctl restart' instead.",
            file=sys.stderr,
        )
        return 3

    if unknown:
        print(f"Unsupported option(s): {' '.join(unknown)}", file=sys.stderr)
        return 2

    fmt = _split_fields(args.format or "")
    header = "noheader" not in fmt
    units = "nounits" not in fmt

    def once() -> str:
        snap = _load()
        gpus = _select(snap, args.id)
        if sub == "pmon":
            return render_pmon(snap, gpus)
        if sub == "dmon":
            return render_dmon(gpus)
        if sub in {"topo", "nvlink", "mig"}:
            raise SystemExit(f"nvidia-smi {sub} is not supported by the GPU emulator")
        if args.list_gpus:
            return render_list(gpus)
        if args.query_gpu is not None:
            if not fmt:
                raise SystemExit("--query-gpu requires --format=csv")
            return render_query_gpu(snap, gpus, _split_fields(args.query_gpu), header, units)
        if args.query_compute_apps is not None:
            if not fmt:
                raise SystemExit("--query-compute-apps requires --format=csv")
            return render_query_apps(gpus, _split_fields(args.query_compute_apps), header, units)
        if args.query:
            return render_query_full(snap, gpus)
        return render_table(snap, gpus)

    if args.loop:
        try:
            while True:
                print(once(), flush=True)
                time.sleep(args.loop)
        except KeyboardInterrupt:
            return 0
    print(once())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
