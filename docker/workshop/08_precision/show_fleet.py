#!/usr/bin/env python3
"""The GPUs available on the cluster, side by side, with the numbers that matter.

None of these figures are measured here - this environment has no GPU and
nothing in it runs at these speeds. They are the manufacturer's published
specifications, printed together so the trade-offs are visible in one place.

The column to look at first is fp64:fp32. It is the ratio of double-precision
to single-precision throughput, and it splits this fleet cleanly in two.
"""

from gpuemu.spec import DEVICES, FLEET

rows = [DEVICES[key] for key in FLEET]

print()
print("GPUs on this cluster")
print("=" * 94)
print(
    f"{'Card':<30}{'VRAM':>7}{'Per node':>10}{'fp32':>9}{'fp64':>9}"
    f"{'fp64:fp32':>11}{'Slurm':>18}"
)
print("-" * 94)
for dev in rows:
    ratio = dev.fp64_ratio
    ratio_text = f"1:{round(1 / ratio)}" if ratio else "-"
    print(
        f"{(dev.name if len(dev.name) <= 29 else dev.name[:26] + chr(8230)):<30}"
        f"{dev.vram_gb:>6}G"
        f"{dev.max_per_node:>10}"
        f"{dev.fp32_tflops:>8.0f}T"
        f"{dev.fp64_tflops:>8.1f}T"
        f"{ratio_text:>11}"
        f"{dev.gres_name + ':1':>18}"
    )
print("-" * 94)
print("fp32 / fp64 are TFLOPS: trillions of floating point operations per second.")

print()
print("Reading this table")
print("-" * 94)
print("  A100 and H100 are 1:2 cards. They do double precision at half their")
print("  single precision rate, which is as good as it gets. If your software")
print("  needs fp64, these are your only sensible options.")
print()
print("  L4 and RTX PRO 6000 are around 1:60. The RTX PRO 6000 has the highest")
print("  fp32 number here by a wide margin and one of the worst fp64 numbers.")
print("  It is a superb card for machine learning and a poor one for a")
print("  quantum chemistry code - the same card, for the same money.")
print()
print("  The L4 is the small, low-power card. It is the right answer more often")
print("  than people expect: if your work fits in 24 GB and does not need")
print("  fp64, an L4 you can have now beats an A100 you have to queue for.")

print()
print("How to request each of them")
print("-" * 94)
for dev in rows:
    print(f"  #SBATCH --gpus-per-node {dev.gres_name}:1")
print()
print("  Ask for more than one only if your software says it can use more than")
print("  one. Most cannot, and a second idle GPU helps nobody.")
print()
