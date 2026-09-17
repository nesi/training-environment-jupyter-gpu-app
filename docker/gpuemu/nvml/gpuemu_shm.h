/*
 * gpuemu_shm.h - shared-memory layout for the emulated GPU.
 *
 * One writer (the gpuemud daemon, in Python) and many readers (this NVML shim,
 * nvidia-smi, anything else that wants the device state). The file is mmap'd by
 * both sides, so the layout below has to agree byte-for-byte with the format
 * strings in gpuemu/shm.py. Every struct is packed and every field is a fixed
 * width integer or a char array, which makes the C layout identical to what
 * Python's struct module produces in '<' (little-endian, no alignment) mode.
 *
 * tests/test_abi.py asserts the two agree, so if you add a field here, add it
 * there too and the test will tell you when they drift.
 *
 * Concurrency is a seqlock. The writer bumps `seq` to an odd value before it
 * touches anything and to the next even value when it is done. A reader takes a
 * copy, then checks that `seq` is even and unchanged; if not it retries. That
 * keeps readers from seeing a half-written frame without needing a lock that a
 * crashing writer could leave held.
 */

#ifndef GPUEMU_SHM_H
#define GPUEMU_SHM_H

#include <stdint.h>

#define GPUEMU_MAGIC "GPUEMU1"
#define GPUEMU_ABI_VERSION 1u

#define GPUEMU_MAX_GPUS 8
#define GPUEMU_MAX_PROCS 64

/* process `type` is a bitmask so one entry can be both compute and graphics */
#define GPUEMU_PROC_COMPUTE 1u
#define GPUEMU_PROC_GRAPHICS 2u
#define GPUEMU_PROC_MPS 4u

#pragma pack(push, 1)

typedef struct {
  uint32_t pid;
  uint32_t type;
  uint64_t used_mem; /* bytes */
  uint32_t sm_util;
  uint32_t mem_util;
  uint32_t enc_util;
  uint32_t dec_util;
  uint32_t gi_id; /* MIG GPU instance, 0xFFFFFFFF when not in MIG mode */
  uint32_t ci_id; /* MIG compute instance, likewise */
  char name[96];
} gpuemu_proc_t; /* 136 bytes */

typedef struct {
  char name[96];
  char uuid[80];
  char serial[32];
  char bus_id[32];
  char bus_id_legacy[16];

  uint32_t pci_domain;
  uint32_t pci_bus;
  uint32_t pci_device;
  uint32_t pci_device_id;
  uint32_t pci_subsys_id;

  uint64_t mem_total; /* bytes */
  uint64_t mem_used;
  uint64_t mem_reserved;

  uint32_t clock_gr;
  uint32_t clock_sm;
  uint32_t clock_mem;
  uint32_t clock_video;
  uint32_t max_clock_gr;
  uint32_t max_clock_sm;
  uint32_t max_clock_mem;
  uint32_t max_clock_video;

  uint32_t util_gpu;
  uint32_t util_mem;
  uint32_t util_enc;
  uint32_t util_dec;
  uint32_t enc_sampling_us;
  uint32_t dec_sampling_us;

  uint32_t temp;
  uint32_t temp_shutdown;
  uint32_t temp_slowdown;
  uint32_t temp_gpu_max;
  uint32_t temp_mem_max;

  /* The L4 is a passive, single-slot card with no fan of its own, so real
   * hardware reports N/A here. -1 means "tell the caller NOT_SUPPORTED". */
  int32_t fan_speed;

  uint32_t power_mw;
  uint32_t power_limit_mw;
  uint32_t power_min_limit_mw;
  uint32_t power_max_limit_mw;
  uint32_t power_default_limit_mw;

  uint32_t pcie_gen;
  uint32_t pcie_width;
  uint32_t pcie_max_gen;
  uint32_t pcie_max_width;
  uint32_t pcie_tx_kbps;
  uint32_t pcie_rx_kbps;

  uint32_t pstate; /* 0 == P0 */

  uint32_t persistence_mode;
  uint32_t compute_mode;
  uint32_t mig_mode;
  uint32_t ecc_mode;
  uint32_t display_active;
  uint32_t display_mode;

  uint64_t ecc_corrected;
  uint64_t ecc_uncorrected;

  uint32_t cc_major;
  uint32_t cc_minor;

  uint32_t n_procs;
  uint32_t _pad;

  gpuemu_proc_t procs[GPUEMU_MAX_PROCS];
} gpuemu_gpu_t; /* 9188 bytes */

typedef struct {
  char magic[8];
  uint32_t version;
  uint32_t n_gpus;
  uint64_t seq;
  double timestamp;
  char driver_version[32];
  char cuda_version[16];
  char nvml_version[32];
  uint32_t _pad2;
  gpuemu_gpu_t gpus[GPUEMU_MAX_GPUS];
} gpuemu_shm_t; /* 73620 bytes */

#pragma pack(pop)

#endif /* GPUEMU_SHM_H */
