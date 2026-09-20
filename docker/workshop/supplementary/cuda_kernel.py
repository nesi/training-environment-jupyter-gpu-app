#!/usr/bin/env python3
"""Real CUDA kernels, really executed - just not on a GPU.

Numba ships a CUDA simulator that runs ``@cuda.jit`` kernels in the Python
interpreter, with each thread as a real Python thread. It is slow, and it is
also completely faithful to the programming model: ``threadIdx``, ``blockIdx``,
grid-stride loops, shared memory, ``syncthreads`` and race conditions all behave
as they would on hardware.

That makes it the one part of this environment where learners write GPU code and
get genuinely correct GPU semantics back, including the bugs. A missing
``syncthreads`` produces a wrong answer here for the same reason it would on an
L4.

The image sets NUMBA_ENABLE_CUDASIM=1, so this runs as-is.
"""

from __future__ import annotations

import numpy as np
from numba import cuda


@cuda.jit
def vector_add(a, b, out):
    """One thread per element, the canonical first CUDA kernel."""
    i = cuda.grid(1)
    if i < out.size:  # guard: the grid is usually larger than the data
        out[i] = a[i] + b[i]


@cuda.jit
def row_sums_shared(matrix, out):
    """Sum each row with a shared-memory reduction.

    Deliberately written the way a workshop would teach it, so the
    ``syncthreads`` calls are load-bearing. Remove either one and the answers
    go wrong - which is the point of running it.
    """
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block = cuda.blockDim.x

    partial = cuda.shared.array(shape=128, dtype=np.float32)

    total = np.float32(0.0)
    for col in range(tid, matrix.shape[1], block):
        total += matrix[row, col]
    partial[tid] = total
    cuda.syncthreads()

    stride = block // 2
    while stride > 0:
        if tid < stride:
            partial[tid] += partial[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        out[row] = partial[0]


def main() -> int:
    print(f"CUDA available to numba: {cuda.is_available()}")
    print("(running under the CUDA simulator - correct semantics, no hardware)\n")

    # -- vector add ---------------------------------------------------
    n = 1024
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 2

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.device_array_like(a)

    threads_per_block = 128
    blocks = (n + threads_per_block - 1) // threads_per_block
    vector_add[blocks, threads_per_block](d_a, d_b, d_out)

    out = d_out.copy_to_host()
    assert np.allclose(out, a + b), "vector_add disagreed with numpy"
    print(f"vector_add: {blocks} blocks x {threads_per_block} threads -> correct")

    # -- shared-memory reduction --------------------------------------
    rows, cols = 16, 500
    matrix = np.random.default_rng(0).random((rows, cols), dtype=np.float32)
    d_matrix = cuda.to_device(matrix)
    d_sums = cuda.device_array(rows, dtype=np.float32)

    row_sums_shared[rows, threads_per_block](d_matrix, d_sums)
    sums = d_sums.copy_to_host()

    expected = matrix.sum(axis=1)
    assert np.allclose(sums, expected, rtol=1e-4), "reduction disagreed with numpy"
    print(f"row_sums_shared: {rows} blocks, shared-memory reduction -> correct")

    print("\nTry breaking it: delete one of the syncthreads() calls in")
    print("row_sums_shared and run again. The simulator will give you the")
    print("same wrong answers a real GPU would.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
