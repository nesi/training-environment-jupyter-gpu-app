#!/usr/bin/env python3
"""What "precision" means, and why it decides which GPU you should ask for.

A computer stores a real number in a fixed number of bits, so most real numbers
are stored slightly wrong. How wrong depends on the format:

    float64  "double precision"   ~16 significant digits
    float32  "single precision"   ~7  significant digits
    float16  "half precision"     ~3  significant digits

Every GPU can do all three. What differs - enormously - is how fast. A GPU is
built out of hardware that does single-precision arithmetic, and double
precision is either supported properly or bolted on:

    A100          1 double for every  2 singles   built for fp64
    H100 NVL      1 double for every  2 singles   built for fp64
    RTX PRO 6000  1 double for every 64 singles   avoid for fp64
    L4            1 double for every 64 singles   avoid for fp64

A factor of 64 is the difference between a run that takes an hour and a run
that takes two and a half days. So the question "which GPU?" usually reduces
to "does my software need double precision?" - and the answer is a property of
your software, not something you get to choose.

    Usually needs fp64      molecular dynamics, quantum chemistry, CFD,
                            climate models, linear solvers, anything where
                            small errors accumulate over millions of steps
    Usually fine in fp32    machine learning, image processing, most
                            Monte Carlo work, visualisation
    Often fine in fp16      neural network training and inference, which is
                            what the newer cards are optimised for

This script shows why that split exists, using arithmetic you can check.
"""

import numpy as np

print(__doc__.split("This script shows")[0].rstrip())
print("=" * 70)


# ------------------------------------------------- 1. accumulating error

print()
print("1. Adding up a million numbers")
print()

values = np.random.default_rng(0).random(1_000_000) * 100

exact = float(np.sum(values, dtype=np.float128)) if hasattr(np, "float128") else None
sum64 = float(np.sum(values.astype(np.float64), dtype=np.float64))
sum32 = float(np.sum(values.astype(np.float32), dtype=np.float32))
reference = exact if exact is not None else sum64

print(f"   float64 total: {sum64:.6f}")
print(f"   float32 total: {sum32:.6f}")
print(f"   difference:    {abs(sum64 - sum32):.6f}")
print()
print("   Each individual addition is only very slightly wrong. There are a")
print("   million of them, and the errors do not cancel. This is what people")
print("   mean by 'the error accumulates'.")


# ------------------------------------------------ 2. catastrophic failure

print()
print("2. Subtracting two nearly equal numbers")
print()

a, b = 1.0000001, 1.0000000
print(f"   ({a} - {b})")
print(f"   float64: {np.float64(a) - np.float64(b):.10e}")
print(f"   float32: {np.float32(a) - np.float32(b):.10e}")
print(f"   correct: {1e-7:.10e}")
print()
print("   float32 does not have enough digits to tell these two numbers")
print("   apart properly, so the answer loses most of its meaning. Iterative")
print("   solvers do this millions of times.")


# --------------------------------------------- 3. an ill-conditioned solve

print()
print("3. Solving a system of equations that is sensitive to error")
print()

n = 12
hilbert = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
truth = np.ones(n)
rhs = hilbert @ truth

x64 = np.linalg.solve(hilbert.astype(np.float64), rhs.astype(np.float64))
x32 = np.linalg.solve(hilbert.astype(np.float32), rhs.astype(np.float32))

print(f"   worst error, float64: {np.max(np.abs(x64 - truth)):.2e}")
print(f"   worst error, float32: {np.max(np.abs(x32 - truth)):.2e}")
print()
print("   The correct answer is all ones. In float32 the answer is wrong by")
print("   more than the answer itself. No amount of extra compute fixes this;")
print("   only more precision does.")


# ----------------------------------------------------------- what to do

print()
print("=" * 70)
print()
print("How to find out what YOUR software needs, without guessing:")
print()
print("  * Read its documentation. Software that needs fp64 almost always")
print("    says so, and often refuses to build without it.")
print("  * Look for a precision setting in your input files. Many packages")
print("    (GROMACS, LAMMPS, VASP) ship both single and double builds.")
print("  * If you are using machine learning libraries, you are in fp32 or")
print("    fp16 already unless you went out of your way not to be.")
print()
print("Then choose:")
print()
print("  needs fp64        -> A100 or H100. Do not use L4 or RTX PRO 6000.")
print("  fp32 is fine      -> any of them; pick on VRAM and availability.")
print("  fp16 / training   -> the newest card you can get.")
