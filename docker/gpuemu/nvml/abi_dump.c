/*
 * Prints the sizes and offsets the C side of the shared state expects.
 *
 * tests/test_abi.py runs this and compares against gpuemu/shm.py. Without it,
 * a field added on one side and forgotten on the other produces garbage
 * readings rather than an error, which is a miserable thing to debug.
 */

#include <stddef.h>
#include <stdio.h>

#include "gpuemu_shm.h"

int main(void) {
  printf("proc_size %zu\n", sizeof(gpuemu_proc_t));
  printf("gpu_size %zu\n", sizeof(gpuemu_gpu_t));
  printf("shm_size %zu\n", sizeof(gpuemu_shm_t));
  printf("max_gpus %d\n", GPUEMU_MAX_GPUS);
  printf("max_procs %d\n", GPUEMU_MAX_PROCS);
  printf("off_seq %zu\n", offsetof(gpuemu_shm_t, seq));
  printf("off_gpus %zu\n", offsetof(gpuemu_shm_t, gpus));
  printf("off_gpu_procs %zu\n", offsetof(gpuemu_gpu_t, procs));
  printf("off_gpu_mem_total %zu\n", offsetof(gpuemu_gpu_t, mem_total));
  printf("off_gpu_fan %zu\n", offsetof(gpuemu_gpu_t, fan_speed));
  printf("off_gpu_nprocs %zu\n", offsetof(gpuemu_gpu_t, n_procs));
  return 0;
}
