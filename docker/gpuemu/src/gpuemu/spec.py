"""The device we pretend to be, and the physical behaviour we pretend it has.

Defaults describe an NVIDIA L4: Ada Lovelace AD104, 24 GB GDDR6, 72 W, PCIe
Gen4 x16, passively cooled. The numbers are the ones a real L4 reports, because
learners will compare what they see here against documentation and against a
real cluster, and round numbers would give the game away for no benefit.

Set ``GPUEMU_DEVICE`` to one of the keys in ``DEVICES`` to emulate something
else, or ``GPUEMU_GPUS`` to a count to pretend there are several.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

MIB = 1024 * 1024


@dataclass(frozen=True)
class DeviceSpec:
    """Static properties of an emulated card."""

    name: str
    architecture: str
    # nvidia-smi reports the board's usable framebuffer, which is a little less
    # than the marketing capacity once ECC and driver reservations are taken out.
    mem_total_mib: int
    mem_reserved_mib: int

    sm_count: int
    cuda_cores: int
    cc_major: int
    cc_minor: int

    max_clock_gr_mhz: int
    max_clock_sm_mhz: int
    max_clock_mem_mhz: int
    max_clock_video_mhz: int
    idle_clock_gr_mhz: int
    idle_clock_mem_mhz: int

    power_limit_w: float
    power_idle_w: float
    power_min_limit_w: float

    temp_idle_c: float
    temp_max_load_c: float
    temp_slowdown_c: int
    temp_shutdown_c: int
    temp_gpu_max_c: int
    temp_mem_max_c: int
    # Seconds for temperature to cover ~63% of the gap to its target. The L4 is
    # a passive card in a server airflow path, so it heats and cools slowly
    # compared with a fan-cooled desktop part.
    thermal_tau_s: float

    pci_device_id: int
    pci_subsys_id: int
    pcie_max_gen: int
    pcie_max_width: int

    # Passive cards have no fan of their own and report N/A rather than 0.
    has_fan: bool

    # Rough device memory bandwidth, used to turn a memory-traffic estimate into
    # the "memory utilisation" percentage NVML reports.
    mem_bandwidth_gbps: float

    # ---- published specifications, not measurements -----------------------
    #
    # Nothing below is emulated: no arithmetic in this environment runs at
    # these rates, and none of it is measured here. They are the vendor's
    # figures, carried so that a learner can ask "what would this card be good
    # at?" against the card they are currently pretending to have. The
    # fp64:fp32 ratio is the number that actually decides which cards suit
    # which software, and it varies by a factor of sixty across this fleet.
    fp32_tflops: float = 0.0
    fp64_tflops: float = 0.0
    tf32_tflops: float = 0.0  # tensor core, dense
    fp16_tflops: float = 0.0  # tensor core, dense

    # The capacity this board is sold and documented as, in GB. Distinct from
    # mem_total_mib, which is the usable framebuffer nvidia-smi reports once
    # ECC and the driver have taken their share - a 24 GB L4 shows 23034 MiB.
    # Comparison tables should use this one, because it is the number in the
    # cluster's documentation and the number a researcher will quote.
    vram_gb: int = 0

    # How this card is spelled in a Slurm request, and how many sit in one
    # node, on the cluster this workshop teaches against.
    gres_name: str = ""
    max_per_node: int = 1
    host_cores_per_node: int = 0
    host_mem_gb_per_node: int = 0

    @property
    def mem_total_bytes(self) -> int:
        return self.mem_total_mib * MIB

    @property
    def mem_reserved_bytes(self) -> int:
        return self.mem_reserved_mib * MIB

    @property
    def fp64_ratio(self) -> float:
        """FP64 throughput as a fraction of FP32 — the 'is it a 1/64 card?' number."""
        if not self.fp32_tflops:
            return 0.0
        return self.fp64_tflops / self.fp32_tflops


L4 = DeviceSpec(
    name="NVIDIA L4",
    architecture="Ada Lovelace",
    mem_total_mib=23034,
    mem_reserved_mib=533,
    sm_count=60,
    cuda_cores=7680,
    cc_major=8,
    cc_minor=9,
    max_clock_gr_mhz=2040,
    max_clock_sm_mhz=2040,
    max_clock_mem_mhz=6251,
    max_clock_video_mhz=1950,
    idle_clock_gr_mhz=210,
    idle_clock_mem_mhz=405,
    power_limit_w=72.0,
    power_idle_w=12.0,
    power_min_limit_w=40.0,
    temp_idle_c=38.0,
    temp_max_load_c=76.0,
    temp_slowdown_c=92,
    temp_shutdown_c=95,
    temp_gpu_max_c=90,
    temp_mem_max_c=95,
    thermal_tau_s=45.0,
    pci_device_id=0x27B810DE,  # AD104GL [L4]
    pci_subsys_id=0x16CA10DE,
    pcie_max_gen=4,
    pcie_max_width=16,
    has_fan=False,
    mem_bandwidth_gbps=300.0,
    fp32_tflops=30.3,
    fp64_tflops=0.489,
    tf32_tflops=60.0,
    fp16_tflops=121.0,
    vram_gb=24,
    gres_name="l4",
    max_per_node=4,
    host_cores_per_node=168,
    host_mem_gb_per_node=768,
)

# The rest of the cluster's fleet, so the same image can host a workshop about
# any card a researcher might request. Values follow each board's public specs
# and the node layout published at docs.nesi.org.nz/Batch_Computing/Hardware/.
A100_80GB = replace(
    L4,
    name="NVIDIA A100-SXM4-80GB",
    architecture="Ampere",
    mem_total_mib=81920,
    mem_reserved_mib=1024,
    sm_count=108,
    cuda_cores=6912,
    cc_major=8,
    cc_minor=0,
    max_clock_gr_mhz=1410,
    max_clock_sm_mhz=1410,
    max_clock_mem_mhz=1593,
    idle_clock_gr_mhz=210,
    idle_clock_mem_mhz=405,
    power_limit_w=400.0,
    power_idle_w=50.0,
    power_min_limit_w=100.0,
    temp_idle_c=32.0,
    temp_max_load_c=72.0,
    pci_device_id=0x20B210DE,
    pci_subsys_id=0x147F10DE,
    mem_bandwidth_gbps=2039.0,
    # 1:2 FP64. The only card in this fleet built for double precision.
    fp32_tflops=19.5,
    fp64_tflops=9.7,
    tf32_tflops=156.0,
    fp16_tflops=312.0,
    vram_gb=80,
    gres_name="a100",
    max_per_node=4,
    host_cores_per_node=64,
    host_mem_gb_per_node=512,
)

H100_NVL = replace(
    L4,
    name="NVIDIA H100 NVL",
    architecture="Hopper",
    mem_total_mib=95830,
    mem_reserved_mib=1100,
    sm_count=132,
    cuda_cores=16896,
    cc_major=9,
    cc_minor=0,
    max_clock_gr_mhz=1785,
    max_clock_sm_mhz=1785,
    max_clock_mem_mhz=2619,
    power_limit_w=400.0,
    power_idle_w=55.0,
    power_min_limit_w=200.0,
    temp_idle_c=33.0,
    temp_max_load_c=74.0,
    pci_device_id=0x232110DE,
    pci_subsys_id=0x167410DE,
    mem_bandwidth_gbps=3938.0,
    fp32_tflops=60.0,
    fp64_tflops=30.0,
    tf32_tflops=835.0,
    fp16_tflops=1671.0,
    vram_gb=94,
    gres_name="h100",
    max_per_node=2,
    host_cores_per_node=168,
    host_mem_gb_per_node=768,
)

RTX_PRO_6000 = replace(
    L4,
    name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
    architecture="Blackwell",
    # 96 GB nominal; the reported figure is that less the driver's framebuffer
    # overhead, in the same proportion the H100 shows.
    mem_total_mib=97887,
    mem_reserved_mib=1100,
    sm_count=188,
    cuda_cores=24064,
    cc_major=12,
    cc_minor=0,
    # NVIDIA quotes 120 TFLOPS FP32, and FP32 = cores x 2 x clock, which puts
    # the boost clock at 24064 x 2 x 2.49 GHz = 119.8 TFLOPS.
    max_clock_gr_mhz=2490,
    max_clock_sm_mhz=2490,
    # 1597 GB/s over a 512-bit bus is ~25 Gbps; nvidia-smi reports GDDR7 at half
    # the effective rate, the same convention it uses for GDDR6X.
    max_clock_mem_mhz=12501,
    max_clock_video_mhz=2100,
    power_limit_w=600.0,
    power_idle_w=30.0,
    # NVIDIA document this board as capable of being power capped to 450 W.
    power_min_limit_w=450.0,
    temp_idle_c=35.0,
    temp_max_load_c=80.0,
    temp_slowdown_c=93,
    temp_shutdown_c=98,
    # GB202. Published as 10de:2bb1; the Server Edition SKU may carry its own
    # device ID, which we have not been able to confirm.
    pci_device_id=0x2BB110DE,
    pci_subsys_id=0x204B10DE,
    pcie_max_gen=5,
    pcie_max_width=16,
    # The Server Edition is a dual-slot passive card, like the L4.
    has_fan=False,
    mem_bandwidth_gbps=1597.0,
    # A consumer-lineage die: enormous FP32, 1/64 FP64. Fast at everything
    # except the one thing a lot of scientific code needs.
    fp32_tflops=120.0,
    fp64_tflops=1.9,
    tf32_tflops=240.0,
    fp16_tflops=480.0,
    vram_gb=96,
    gres_name="pro_6000",
    max_per_node=2,
    host_cores_per_node=168,
    host_mem_gb_per_node=768,
)

DEVICES = {
    "l4": L4,
    "a100": A100_80GB,
    "h100": H100_NVL,
    "rtxpro6000": RTX_PRO_6000,
}

# Fleet order for anything that prints a comparison: cheapest and smallest
# first, which is also the order a researcher should try them in.
FLEET = ("l4", "a100", "h100", "rtxpro6000")

# Back-compatible aliases. The A100 and H100 definitions were originally the
# PCIe parts; they now describe the SXM4-80GB and NVL boards this cluster
# actually has, because a workshop that teaches the wrong VRAM figure teaches
# the wrong request.
A100_40GB = A100_80GB
H100_PCIE = H100_NVL

# Reported by nvidia-smi and NVML. Pinned to a real driver/CUDA pairing so that
# version checks in learners' code behave the way they would on the cluster.
DRIVER_VERSION = "550.54.15"
CUDA_VERSION = "12.4"
NVML_VERSION = "12.550.54.15"


def _parse_mib(text: str) -> int:
    """Parse ``1GiB`` / ``512M`` / a bare number of MiB into MiB."""
    raw = text.strip().upper().removesuffix("B")
    multiplier = 1
    for suffix, factor in (("GI", 1024), ("G", 1024), ("MI", 1), ("M", 1)):
        if raw.endswith(suffix):
            raw, multiplier = raw[: -len(suffix)], factor
            break
    return max(1, int(float(raw) * multiplier))


def selected_device() -> DeviceSpec:
    key = os.environ.get("GPUEMU_DEVICE", "l4").strip().lower()
    try:
        dev = DEVICES[key]
    except KeyError:
        known = ", ".join(sorted(DEVICES))
        raise SystemExit(f"GPUEMU_DEVICE={key!r} is not one of: {known}") from None

    # A deliberately small card is the cheapest way to teach memory pressure:
    # on a 1 GB device a learner hits a real out-of-memory error with a tensor
    # that costs the host almost nothing, so the exercise works without the
    # session needing 24 GB of RAM to fill.
    override = os.environ.get("GPUEMU_MEM_TOTAL", "").strip()
    if override:
        try:
            total_mib = _parse_mib(override)
        except ValueError:
            raise SystemExit(
                f"GPUEMU_MEM_TOTAL={override!r} is not a size like '1GiB' or '512MiB'"
            ) from None
        # Keep the driver's reservation at the same proportion the real board
        # has (~2.3% on an L4), so "total" and "free" stay plausible.
        ratio = dev.mem_reserved_mib / dev.mem_total_mib
        dev = replace(
            dev,
            mem_total_mib=total_mib,
            mem_reserved_mib=max(1, round(total_mib * ratio)),
        )
    return dev


def device_count() -> int:
    try:
        n = int(os.environ.get("GPUEMU_GPUS", "1"))
    except ValueError:
        return 1
    return max(1, min(n, 8))


def make_uuid(index: int) -> str:
    """A stable, obviously-synthetic UUID.

    Real GPU UUIDs are random, but a deterministic one means a learner can
    restart the daemon without every tool that cached the ID losing track of
    the device. The leading zeros make it clear this is not real hardware to
    anyone who looks closely.
    """
    return f"GPU-e0000000-0000-4000-8000-{index:012d}"


def make_bus_ids(index: int) -> tuple[str, str, int, int, int]:
    """PCI addressing for device ``index``: (busId, busIdLegacy, domain, bus, dev).

    Cloud instances typically present GPUs at 00:04.0, 00:05.0 and so on, which
    is what we imitate.
    """
    domain, bus, dev = 0, 0, 4 + index
    return (
        f"{domain:08X}:{bus:02X}:{dev:02X}.0",
        f"{domain:04X}:{bus:02X}:{dev:02X}.0",
        domain,
        bus,
        dev,
    )
