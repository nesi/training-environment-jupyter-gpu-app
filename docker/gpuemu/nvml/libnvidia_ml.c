/*
 * libnvidia-ml.so.1 - a stand-in NVML implementation backed by the gpuemu
 * simulator instead of by a real driver.
 *
 * Tools like nvtop never link NVML at build time; they dlopen
 * "libnvidia-ml.so.1" and pull every entry point out with dlsym. That means a
 * shared object exporting the right symbol names with the right signatures is
 * indistinguishable from the real thing, as far as they can tell, and no kernel
 * driver or device node has to exist. This file is that shared object.
 *
 * Everything it reports comes from the mmap'd state file described in
 * gpuemu_shm.h, which the gpuemud daemon keeps up to date. If the daemon is not
 * running, we fall back to a static idle L4 so that nvidia-smi and nvtop still
 * show a plausible device rather than failing to initialise.
 *
 * Scope: the subset that nvtop, nvidia-smi and pynvml actually call. Anything
 * outside that returns NVML_ERROR_NOT_SUPPORTED, which is the same thing real
 * NVML does for features a given card lacks, so callers handle it already.
 */

#define _GNU_SOURCE

#include <fcntl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "gpuemu_shm.h"

/* ------------------------------------------------------------------ */
/* NVML public types (mirrored from nvml.h - we cannot include it)     */
/* ------------------------------------------------------------------ */

typedef enum {
  NVML_SUCCESS = 0,
  NVML_ERROR_UNINITIALIZED = 1,
  NVML_ERROR_INVALID_ARGUMENT = 2,
  NVML_ERROR_NOT_SUPPORTED = 3,
  NVML_ERROR_NO_PERMISSION = 4,
  NVML_ERROR_ALREADY_INITIALIZED = 5,
  NVML_ERROR_NOT_FOUND = 6,
  NVML_ERROR_INSUFFICIENT_SIZE = 7,
  NVML_ERROR_DRIVER_NOT_LOADED = 9,
  NVML_ERROR_TIMEOUT = 10,
  NVML_ERROR_FUNCTION_NOT_FOUND = 13,
  NVML_ERROR_UNKNOWN = 999,
} nvmlReturn_t;

typedef struct nvmlDevice_st *nvmlDevice_t;

typedef struct {
  char busIdLegacy[16];
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pciDeviceId;
  unsigned int pciSubSystemId;
  char busId[32];
} nvmlPciInfo_t;

typedef struct {
  unsigned int gpu;
  unsigned int memory;
} nvmlUtilization_t;

typedef struct {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
} nvmlMemory_t;

typedef struct {
  unsigned int version;
  unsigned long long total;
  unsigned long long reserved;
  unsigned long long free;
  unsigned long long used;
} nvmlMemory_v2_t;

typedef struct {
  unsigned int pid;
  unsigned long long usedGpuMemory;
} nvmlProcessInfo_v1_t;

typedef struct {
  unsigned int pid;
  unsigned long long usedGpuMemory;
  unsigned int gpuInstanceId;
  unsigned int computeInstanceId;
} nvmlProcessInfo_v2_t;

typedef nvmlProcessInfo_v2_t nvmlProcessInfo_v3_t;
typedef nvmlProcessInfo_v2_t nvmlProcessInfo_t;

typedef struct {
  unsigned int pid;
  unsigned long long timeStamp;
  unsigned int smUtil;
  unsigned int memUtil;
  unsigned int encUtil;
  unsigned int decUtil;
} nvmlProcessUtilizationSample_t;

typedef enum {
  NVML_TEMPERATURE_GPU = 0,
} nvmlTemperatureSensors_t;

typedef enum {
  NVML_TEMPERATURE_THRESHOLD_SHUTDOWN = 0,
  NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1,
  NVML_TEMPERATURE_THRESHOLD_MEM_MAX = 2,
  NVML_TEMPERATURE_THRESHOLD_GPU_MAX = 3,
  NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN = 4,
  NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR = 5,
  NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX = 6,
} nvmlTemperatureThresholds_t;

typedef enum {
  NVML_CLOCK_GRAPHICS = 0,
  NVML_CLOCK_SM = 1,
  NVML_CLOCK_MEM = 2,
  NVML_CLOCK_VIDEO = 3,
} nvmlClockType_t;

typedef enum {
  NVML_PCIE_UTIL_TX_BYTES = 0,
  NVML_PCIE_UTIL_RX_BYTES = 1,
} nvmlPcieUtilCounter_t;

typedef enum {
  NVML_MEMORY_ERROR_TYPE_CORRECTED = 0,
  NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1,
} nvmlMemoryErrorType_t;

typedef enum {
  NVML_VOLATILE_ECC = 0,
  NVML_AGGREGATE_ECC = 1,
} nvmlEccCounterType_t;

typedef enum {
  NVML_VALUE_TYPE_DOUBLE = 0,
  NVML_VALUE_TYPE_UNSIGNED_INT = 1,
  NVML_VALUE_TYPE_UNSIGNED_LONG = 2,
  NVML_VALUE_TYPE_UNSIGNED_LONG_LONG = 3,
  NVML_VALUE_TYPE_SIGNED_LONG_LONG = 4,
  NVML_VALUE_TYPE_SIGNED_INT = 5,
  NVML_VALUE_TYPE_UNSIGNED_SHORT = 6,
} nvmlValueType_t;

typedef union {
  double dVal;
  int siVal;
  unsigned int uiVal;
  unsigned long ulVal;
  unsigned long long ullVal;
  signed long long sllVal;
  unsigned short usVal;
} nvmlValue_t;

typedef struct {
  unsigned int fieldId;
  unsigned int scopeId;
  long long timestamp;
  long long latencyUsec;
  nvmlValueType_t valueType;
  nvmlReturn_t nvmlReturn;
  nvmlValue_t value;
} nvmlFieldValue_t;

#define NVML_DEVICE_NAME_BUFFER_SIZE 64
#define NVML_DEVICE_NAME_V2_BUFFER_SIZE 96
#define NVML_DEVICE_UUID_BUFFER_SIZE 80
#define NVML_DEVICE_UUID_V2_BUFFER_SIZE 96
#define NVML_DEVICE_SERIAL_BUFFER_SIZE 30
#define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE 81
#define NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE 80
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 32

#define NVML_FEATURE_DISABLED 0
#define NVML_FEATURE_ENABLED 1
#define NVML_DEVICE_MIG_DISABLE 0
#define NVML_COMPUTEMODE_DEFAULT 0

#define NVML_EXPORT __attribute__((visibility("default")))

/* ------------------------------------------------------------------ */
/* Shared-state access                                                 */
/* ------------------------------------------------------------------ */

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static gpuemu_shm_t *g_shm = NULL;   /* mmap of the live state file */
static int g_shm_fd = -1;
static bool g_initialised = false;
static gpuemu_shm_t g_fallback;      /* static idle L4, used if no daemon */
static bool g_fallback_ready = false;

static const char *state_path(void) {
  const char *p = getenv("GPUEMU_STATE_FILE");
  if (p && *p)
    return p;
  return "/run/gpuemu/state.bin";
}

/* A believable idle L4, so the tooling still works with no daemon running. */
static void build_fallback(void) {
  if (g_fallback_ready)
    return;
  memset(&g_fallback, 0, sizeof(g_fallback));
  memcpy(g_fallback.magic, GPUEMU_MAGIC, sizeof(GPUEMU_MAGIC));
  g_fallback.version = GPUEMU_ABI_VERSION;
  g_fallback.n_gpus = 1;
  g_fallback.seq = 2;
  snprintf(g_fallback.driver_version, sizeof(g_fallback.driver_version), "550.54.15");
  snprintf(g_fallback.cuda_version, sizeof(g_fallback.cuda_version), "12.4");
  snprintf(g_fallback.nvml_version, sizeof(g_fallback.nvml_version), "12.550.54.15");

  gpuemu_gpu_t *g = &g_fallback.gpus[0];
  snprintf(g->name, sizeof(g->name), "NVIDIA L4");
  snprintf(g->uuid, sizeof(g->uuid), "GPU-00000000-0000-0000-0000-000000000000");
  snprintf(g->serial, sizeof(g->serial), "0000000000000");
  snprintf(g->bus_id, sizeof(g->bus_id), "00000000:00:04.0");
  snprintf(g->bus_id_legacy, sizeof(g->bus_id_legacy), "0000:00:04.0");
  g->pci_domain = 0;
  g->pci_bus = 0;
  g->pci_device = 4;
  g->pci_device_id = 0x27B810DEu; /* AD104GL [L4] */
  g->pci_subsys_id = 0x16CA10DEu;
  g->mem_total = 23034ULL * 1024ULL * 1024ULL;
  g->mem_used = 1ULL * 1024ULL * 1024ULL;
  g->mem_reserved = 533ULL * 1024ULL * 1024ULL;
  g->clock_gr = 210;
  g->clock_sm = 210;
  g->clock_mem = 405;
  g->clock_video = 555;
  g->max_clock_gr = 2040;
  g->max_clock_sm = 2040;
  g->max_clock_mem = 6251;
  g->max_clock_video = 1950;
  g->temp = 38;
  g->temp_shutdown = 95;
  g->temp_slowdown = 92;
  g->temp_gpu_max = 90;
  g->temp_mem_max = 95;
  g->fan_speed = -1; /* passive card */
  g->power_mw = 12000;
  g->power_limit_mw = 72000;
  g->power_min_limit_mw = 40000;
  g->power_max_limit_mw = 72000;
  g->power_default_limit_mw = 72000;
  g->pcie_gen = 1;
  g->pcie_width = 8;
  g->pcie_max_gen = 4;
  g->pcie_max_width = 16;
  g->pstate = 8;
  g->compute_mode = NVML_COMPUTEMODE_DEFAULT;
  g->mig_mode = NVML_DEVICE_MIG_DISABLE;
  g->ecc_mode = NVML_FEATURE_ENABLED;
  g->cc_major = 8;
  g->cc_minor = 9;
  g->n_procs = 0;
  g_fallback_ready = true;
}

static void try_map_state(void) {
  struct stat st;
  const char *path = state_path();

  if (g_shm)
    return;

  g_shm_fd = open(path, O_RDONLY);
  if (g_shm_fd < 0)
    return;
  if (fstat(g_shm_fd, &st) != 0 || (size_t)st.st_size < sizeof(gpuemu_shm_t)) {
    close(g_shm_fd);
    g_shm_fd = -1;
    return;
  }
  void *m = mmap(NULL, sizeof(gpuemu_shm_t), PROT_READ, MAP_SHARED, g_shm_fd, 0);
  if (m == MAP_FAILED) {
    close(g_shm_fd);
    g_shm_fd = -1;
    return;
  }
  g_shm = (gpuemu_shm_t *)m;
  if (memcmp(g_shm->magic, GPUEMU_MAGIC, sizeof(GPUEMU_MAGIC)) != 0 ||
      g_shm->version != GPUEMU_ABI_VERSION) {
    munmap(m, sizeof(gpuemu_shm_t));
    close(g_shm_fd);
    g_shm_fd = -1;
    g_shm = NULL;
  }
}

/*
 * Read the seqlock counter.
 *
 * `seq` sits at a naturally 8-byte-aligned offset, but it lives in a packed
 * struct, so taking its address directly would make the compiler assume it is
 * unaligned (and -Waddress-of-packed-member would rightly complain). Going
 * through offsetof and a volatile load sidesteps that while still giving us a
 * single atomic load on every architecture this image targets.
 */
static inline uint64_t read_seq(const gpuemu_shm_t *s) {
  const volatile uint64_t *p =
      (const volatile uint64_t *)((const char *)s + offsetof(gpuemu_shm_t, seq));
  uint64_t v = *p;
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  return v;
}

/*
 * Take a consistent copy of one GPU's state.
 *
 * The daemon marks a write in progress by making `seq` odd, so we read the
 * counter, copy, and read it again; a stable even value means nothing changed
 * underneath us. After a few failed attempts we give up and use the copy we
 * have, which is at worst one frame of slightly mixed numbers on a display that
 * refreshes anyway.
 */
static bool snapshot(unsigned int index, gpuemu_gpu_t *out, gpuemu_shm_t *hdr_out) {
  build_fallback();

  pthread_mutex_lock(&g_lock);
  try_map_state();
  const gpuemu_shm_t *src = g_shm ? g_shm : &g_fallback;

  bool ok = false;
  for (int attempt = 0; attempt < 8; attempt++) {
    uint64_t s1 = read_seq(src);
    if (s1 & 1ULL)
      continue; /* writer mid-update */
    if (index >= src->n_gpus || src->n_gpus > GPUEMU_MAX_GPUS)
      break;
    memcpy(out, &src->gpus[index], sizeof(*out));
    if (hdr_out) {
      memcpy(hdr_out->magic, src->magic, sizeof(src->magic));
      hdr_out->version = src->version;
      hdr_out->n_gpus = src->n_gpus;
      hdr_out->timestamp = src->timestamp;
      memcpy(hdr_out->driver_version, src->driver_version, sizeof(src->driver_version));
      memcpy(hdr_out->cuda_version, src->cuda_version, sizeof(src->cuda_version));
      memcpy(hdr_out->nvml_version, src->nvml_version, sizeof(src->nvml_version));
    }
    uint64_t s2 = read_seq(src);
    if (s1 == s2) {
      ok = true;
      break;
    }
  }
  pthread_mutex_unlock(&g_lock);
  return ok;
}

static unsigned int device_count(void) {
  build_fallback();
  pthread_mutex_lock(&g_lock);
  try_map_state();
  const gpuemu_shm_t *src = g_shm ? g_shm : &g_fallback;
  unsigned int n = src->n_gpus;
  pthread_mutex_unlock(&g_lock);
  if (n > GPUEMU_MAX_GPUS)
    n = GPUEMU_MAX_GPUS;
  return n;
}

/* Handles are just 1-based indices cast to a pointer: never NULL, and cheap to
 * validate, which is all the opacity the API actually requires of us. */
static inline nvmlDevice_t index_to_handle(unsigned int i) {
  return (nvmlDevice_t)(uintptr_t)(i + 1u);
}

static inline bool handle_to_index(nvmlDevice_t d, unsigned int *idx) {
  uintptr_t v = (uintptr_t)d;
  if (v == 0 || v > GPUEMU_MAX_GPUS)
    return false;
  *idx = (unsigned int)(v - 1);
  return true;
}

#define REQUIRE_INIT()                                                                             \
  do {                                                                                             \
    if (!g_initialised)                                                                            \
      return NVML_ERROR_UNINITIALIZED;                                                             \
  } while (0)

#define GET_GPU(dev, gpu)                                                                          \
  gpuemu_gpu_t gpu;                                                                                \
  do {                                                                                             \
    unsigned int _i;                                                                               \
    REQUIRE_INIT();                                                                                \
    if (!handle_to_index((dev), &_i))                                                              \
      return NVML_ERROR_INVALID_ARGUMENT;                                                          \
    if (!snapshot(_i, &gpu, NULL))                                                                 \
      return NVML_ERROR_NOT_FOUND;                                                                 \
  } while (0)

/* Copy into a caller buffer the way NVML does: truncation is an error, not a
 * silent short string. */
static nvmlReturn_t copy_str(char *dst, unsigned int len, const char *src) {
  if (!dst || len == 0)
    return NVML_ERROR_INVALID_ARGUMENT;
  size_t need = strlen(src) + 1;
  if (need > (size_t)len)
    return NVML_ERROR_INSUFFICIENT_SIZE;
  memcpy(dst, src, need);
  return NVML_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Init / shutdown / version                                           */
/* ------------------------------------------------------------------ */

NVML_EXPORT nvmlReturn_t nvmlInit_v2(void) {
  build_fallback();
  pthread_mutex_lock(&g_lock);
  try_map_state();
  g_initialised = true;
  pthread_mutex_unlock(&g_lock);
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlInit(void) { return nvmlInit_v2(); }

NVML_EXPORT nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
  (void)flags;
  return nvmlInit_v2();
}

NVML_EXPORT nvmlReturn_t nvmlShutdown(void) {
  pthread_mutex_lock(&g_lock);
  if (g_shm) {
    munmap((void *)g_shm, sizeof(gpuemu_shm_t));
    g_shm = NULL;
  }
  if (g_shm_fd >= 0) {
    close(g_shm_fd);
    g_shm_fd = -1;
  }
  g_initialised = false;
  pthread_mutex_unlock(&g_lock);
  return NVML_SUCCESS;
}

NVML_EXPORT const char *nvmlErrorString(nvmlReturn_t result) {
  switch (result) {
  case NVML_SUCCESS: return "The operation was successful";
  case NVML_ERROR_UNINITIALIZED: return "Uninitialized";
  case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
  case NVML_ERROR_NOT_SUPPORTED: return "Not Supported";
  case NVML_ERROR_NO_PERMISSION: return "Insufficient Permissions";
  case NVML_ERROR_ALREADY_INITIALIZED: return "Already Initialized";
  case NVML_ERROR_NOT_FOUND: return "Not Found";
  case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient Size";
  case NVML_ERROR_DRIVER_NOT_LOADED: return "Driver Not Loaded";
  case NVML_ERROR_TIMEOUT: return "Timeout";
  case NVML_ERROR_FUNCTION_NOT_FOUND: return "Function Not Found";
  default: return "Unknown Error";
  }
}

static void header_strings(char *driver, size_t dlen, char *cuda, size_t clen, char *nvml,
                           size_t nlen) {
  gpuemu_gpu_t tmp;
  gpuemu_shm_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  if (!snapshot(0, &tmp, &hdr)) {
    build_fallback();
    memcpy(&hdr, &g_fallback, sizeof(hdr) < sizeof(g_fallback) ? sizeof(hdr) : sizeof(g_fallback));
  }
  if (driver)
    snprintf(driver, dlen, "%s", hdr.driver_version);
  if (cuda)
    snprintf(cuda, clen, "%s", hdr.cuda_version);
  if (nvml)
    snprintf(nvml, nlen, "%s", hdr.nvml_version);
}

NVML_EXPORT nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length) {
  char buf[64];
  REQUIRE_INIT();
  header_strings(buf, sizeof(buf), NULL, 0, NULL, 0);
  return copy_str(version, length, buf);
}

NVML_EXPORT nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length) {
  char buf[64];
  REQUIRE_INIT();
  header_strings(NULL, 0, NULL, 0, buf, sizeof(buf));
  return copy_str(version, length, buf);
}

NVML_EXPORT nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion) {
  char buf[32];
  int major = 0, minor = 0;
  if (!cudaDriverVersion)
    return NVML_ERROR_INVALID_ARGUMENT;
  header_strings(NULL, 0, buf, sizeof(buf), NULL, 0);
  sscanf(buf, "%d.%d", &major, &minor);
  *cudaDriverVersion = major * 1000 + minor * 10;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *v) {
  return nvmlSystemGetCudaDriverVersion(v);
}

/* ------------------------------------------------------------------ */
/* Device enumeration and identity                                     */
/* ------------------------------------------------------------------ */

NVML_EXPORT nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
  REQUIRE_INIT();
  if (!deviceCount)
    return NVML_ERROR_INVALID_ARGUMENT;
  *deviceCount = device_count();
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetCount(unsigned int *c) { return nvmlDeviceGetCount_v2(c); }

NVML_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
  REQUIRE_INIT();
  if (!device)
    return NVML_ERROR_INVALID_ARGUMENT;
  if (index >= device_count())
    return NVML_ERROR_INVALID_ARGUMENT;
  *device = index_to_handle(index);
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int i, nvmlDevice_t *d) {
  return nvmlDeviceGetHandleByIndex_v2(i, d);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
  unsigned int i;
  REQUIRE_INIT();
  if (!index || !handle_to_index(device, &i))
    return NVML_ERROR_INVALID_ARGUMENT;
  *index = i;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device) {
  unsigned int n = device_count();
  REQUIRE_INIT();
  if (!uuid || !device)
    return NVML_ERROR_INVALID_ARGUMENT;
  for (unsigned int i = 0; i < n; i++) {
    gpuemu_gpu_t g;
    if (snapshot(i, &g, NULL) && strcmp(g.uuid, uuid) == 0) {
      *device = index_to_handle(i);
      return NVML_SUCCESS;
    }
  }
  return NVML_ERROR_NOT_FOUND;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId,
                                                          nvmlDevice_t *device) {
  unsigned int n = device_count();
  REQUIRE_INIT();
  if (!pciBusId || !device)
    return NVML_ERROR_INVALID_ARGUMENT;
  for (unsigned int i = 0; i < n; i++) {
    gpuemu_gpu_t g;
    if (snapshot(i, &g, NULL) &&
        (strcasecmp(g.bus_id, pciBusId) == 0 || strcasecmp(g.bus_id_legacy, pciBusId) == 0)) {
      *device = index_to_handle(i);
      return NVML_SUCCESS;
    }
  }
  return NVML_ERROR_NOT_FOUND;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *b, nvmlDevice_t *d) {
  return nvmlDeviceGetHandleByPciBusId_v2(b, d);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
  GET_GPU(device, g);
  return copy_str(name, length, g.name);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length) {
  GET_GPU(device, g);
  return copy_str(uuid, length, g.uuid);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial,
                                             unsigned int length) {
  GET_GPU(device, g);
  return copy_str(serial, length, g.serial);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber) {
  unsigned int i;
  REQUIRE_INIT();
  if (!minorNumber || !handle_to_index(device, &i))
    return NVML_ERROR_INVALID_ARGUMENT;
  *minorNumber = i;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId) {
  GET_GPU(device, g);
  if (!boardId)
    return NVML_ERROR_INVALID_ARGUMENT;
  *boardId = (g.pci_bus << 8) | g.pci_device;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
  GET_GPU(device, g);
  if (!pci)
    return NVML_ERROR_INVALID_ARGUMENT;
  memset(pci, 0, sizeof(*pci));
  snprintf(pci->busIdLegacy, sizeof(pci->busIdLegacy), "%s", g.bus_id_legacy);
  snprintf(pci->busId, sizeof(pci->busId), "%s", g.bus_id);
  pci->domain = g.pci_domain;
  pci->bus = g.pci_bus;
  pci->device = g.pci_device;
  pci->pciDeviceId = g.pci_device_id;
  pci->pciSubSystemId = g.pci_subsys_id;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t d, nvmlPciInfo_t *p) {
  return nvmlDeviceGetPciInfo_v3(d, p);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t d, nvmlPciInfo_t *p) {
  return nvmlDeviceGetPciInfo_v3(d, p);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major,
                                                            int *minor) {
  GET_GPU(device, g);
  if (!major || !minor)
    return NVML_ERROR_INVALID_ARGUMENT;
  *major = (int)g.cc_major;
  *minor = (int)g.cc_minor;
  return NVML_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Live telemetry                                                      */
/* ------------------------------------------------------------------ */

NVML_EXPORT nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device,
                                                       nvmlUtilization_t *utilization) {
  GET_GPU(device, g);
  if (!utilization)
    return NVML_ERROR_INVALID_ARGUMENT;
  utilization->gpu = g.util_gpu;
  utilization->memory = g.util_mem;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
  GET_GPU(device, g);
  if (!memory)
    return NVML_ERROR_INVALID_ARGUMENT;
  memory->total = g.mem_total;
  memory->used = g.mem_used;
  memory->free = g.mem_total > g.mem_used ? g.mem_total - g.mem_used : 0;
  return NVML_SUCCESS;
}

/*
 * The v2 form is a versioned struct: the caller stamps `version` before the
 * call and NVML rejects anything it does not recognise. Real NVML returns
 * INVALID_ARGUMENT on a bad version, and nvtop relies on that to decide whether
 * to fall back to v1, so mirror it rather than being lenient.
 */
NVML_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory) {
  GET_GPU(device, g);
  if (!memory)
    return NVML_ERROR_INVALID_ARGUMENT;
  unsigned int expected = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2u << 24));
  if (memory->version != expected)
    return NVML_ERROR_INVALID_ARGUMENT;
  memory->total = g.mem_total;
  memory->reserved = g.mem_reserved;
  memory->used = g.mem_used;
  memory->free = g.mem_total > (g.mem_used + g.mem_reserved)
                     ? g.mem_total - g.mem_used - g.mem_reserved
                     : 0;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device,
                                                  nvmlTemperatureSensors_t sensorType,
                                                  unsigned int *temp) {
  GET_GPU(device, g);
  if (!temp)
    return NVML_ERROR_INVALID_ARGUMENT;
  if (sensorType != NVML_TEMPERATURE_GPU)
    return NVML_ERROR_NOT_SUPPORTED;
  *temp = g.temp;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device,
                                                           nvmlTemperatureThresholds_t type,
                                                           unsigned int *temp) {
  GET_GPU(device, g);
  if (!temp)
    return NVML_ERROR_INVALID_ARGUMENT;
  switch (type) {
  case NVML_TEMPERATURE_THRESHOLD_SHUTDOWN: *temp = g.temp_shutdown; return NVML_SUCCESS;
  case NVML_TEMPERATURE_THRESHOLD_SLOWDOWN: *temp = g.temp_slowdown; return NVML_SUCCESS;
  case NVML_TEMPERATURE_THRESHOLD_GPU_MAX: *temp = g.temp_gpu_max; return NVML_SUCCESS;
  case NVML_TEMPERATURE_THRESHOLD_MEM_MAX: *temp = g.temp_mem_max; return NVML_SUCCESS;
  default: return NVML_ERROR_NOT_SUPPORTED;
  }
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed) {
  GET_GPU(device, g);
  if (!speed)
    return NVML_ERROR_INVALID_ARGUMENT;
  if (g.fan_speed < 0)
    return NVML_ERROR_NOT_SUPPORTED; /* passively cooled, as on a real L4 */
  *speed = (unsigned int)g.fan_speed;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int *numFans) {
  GET_GPU(device, g);
  if (!numFans)
    return NVML_ERROR_INVALID_ARGUMENT;
  *numFans = g.fan_speed < 0 ? 0u : 1u;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
  GET_GPU(device, g);
  if (!power)
    return NVML_ERROR_INVALID_ARGUMENT;
  *power = g.power_mw;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
  GET_GPU(device, g);
  if (!limit)
    return NVML_ERROR_INVALID_ARGUMENT;
  *limit = g.power_limit_mw;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device,
                                                           unsigned int *limit) {
  return nvmlDeviceGetEnforcedPowerLimit(device, limit);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device,
                                                                  unsigned int *limit) {
  GET_GPU(device, g);
  if (!limit)
    return NVML_ERROR_INVALID_ARGUMENT;
  *limit = g.power_default_limit_mw;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device,
                                                                      unsigned int *minLimit,
                                                                      unsigned int *maxLimit) {
  GET_GPU(device, g);
  if (!minLimit || !maxLimit)
    return NVML_ERROR_INVALID_ARGUMENT;
  *minLimit = g.power_min_limit_mw;
  *maxLimit = g.power_max_limit_mw;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type,
                                                unsigned int *clock) {
  GET_GPU(device, g);
  if (!clock)
    return NVML_ERROR_INVALID_ARGUMENT;
  switch (type) {
  case NVML_CLOCK_GRAPHICS: *clock = g.clock_gr; return NVML_SUCCESS;
  case NVML_CLOCK_SM: *clock = g.clock_sm; return NVML_SUCCESS;
  case NVML_CLOCK_MEM: *clock = g.clock_mem; return NVML_SUCCESS;
  case NVML_CLOCK_VIDEO: *clock = g.clock_video; return NVML_SUCCESS;
  default: return NVML_ERROR_NOT_SUPPORTED;
  }
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type,
                                                   unsigned int *clock) {
  GET_GPU(device, g);
  if (!clock)
    return NVML_ERROR_INVALID_ARGUMENT;
  switch (type) {
  case NVML_CLOCK_GRAPHICS: *clock = g.max_clock_gr; return NVML_SUCCESS;
  case NVML_CLOCK_SM: *clock = g.max_clock_sm; return NVML_SUCCESS;
  case NVML_CLOCK_MEM: *clock = g.max_clock_mem; return NVML_SUCCESS;
  case NVML_CLOCK_VIDEO: *clock = g.max_clock_video; return NVML_SUCCESS;
  default: return NVML_ERROR_NOT_SUPPORTED;
  }
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, unsigned int *pState) {
  GET_GPU(device, g);
  if (!pState)
    return NVML_ERROR_INVALID_ARGUMENT;
  *pState = g.pstate;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t d, unsigned int *p) {
  return nvmlDeviceGetPerformanceState(d, p);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device,
                                                         unsigned int *utilization,
                                                         unsigned int *samplingPeriodUs) {
  GET_GPU(device, g);
  if (!utilization || !samplingPeriodUs)
    return NVML_ERROR_INVALID_ARGUMENT;
  *utilization = g.util_enc;
  *samplingPeriodUs = g.enc_sampling_us;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device,
                                                         unsigned int *utilization,
                                                         unsigned int *samplingPeriodUs) {
  GET_GPU(device, g);
  if (!utilization || !samplingPeriodUs)
    return NVML_ERROR_INVALID_ARGUMENT;
  *utilization = g.util_dec;
  *samplingPeriodUs = g.dec_sampling_us;
  return NVML_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* PCIe                                                                */
/* ------------------------------------------------------------------ */

NVML_EXPORT nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device,
                                                             unsigned int *gen) {
  GET_GPU(device, g);
  if (!gen)
    return NVML_ERROR_INVALID_ARGUMENT;
  *gen = g.pcie_gen;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device,
                                                            unsigned int *gen) {
  GET_GPU(device, g);
  if (!gen)
    return NVML_ERROR_INVALID_ARGUMENT;
  *gen = g.pcie_max_gen;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *width) {
  GET_GPU(device, g);
  if (!width)
    return NVML_ERROR_INVALID_ARGUMENT;
  *width = g.pcie_width;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int *width) {
  GET_GPU(device, g);
  if (!width)
    return NVML_ERROR_INVALID_ARGUMENT;
  *width = g.pcie_max_width;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device,
                                                     nvmlPcieUtilCounter_t counter,
                                                     unsigned int *value) {
  GET_GPU(device, g);
  if (!value)
    return NVML_ERROR_INVALID_ARGUMENT;
  switch (counter) {
  case NVML_PCIE_UTIL_TX_BYTES: *value = g.pcie_tx_kbps; return NVML_SUCCESS;
  case NVML_PCIE_UTIL_RX_BYTES: *value = g.pcie_rx_kbps; return NVML_SUCCESS;
  default: return NVML_ERROR_NOT_SUPPORTED;
  }
}

/* ------------------------------------------------------------------ */
/* Modes and ECC                                                       */
/* ------------------------------------------------------------------ */

NVML_EXPORT nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, unsigned int *mode) {
  GET_GPU(device, g);
  if (!mode)
    return NVML_ERROR_INVALID_ARGUMENT;
  *mode = g.persistence_mode;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, unsigned int *mode) {
  GET_GPU(device, g);
  if (!mode)
    return NVML_ERROR_INVALID_ARGUMENT;
  *mode = g.compute_mode;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *current,
                                              unsigned int *pending) {
  GET_GPU(device, g);
  if (!current || !pending)
    return NVML_ERROR_INVALID_ARGUMENT;
  *current = g.mig_mode;
  *pending = g.mig_mode;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, unsigned int *current,
                                              unsigned int *pending) {
  GET_GPU(device, g);
  if (!current || !pending)
    return NVML_ERROR_INVALID_ARGUMENT;
  *current = g.ecc_mode;
  *pending = g.ecc_mode;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device,
                                                     nvmlMemoryErrorType_t errorType,
                                                     nvmlEccCounterType_t counterType,
                                                     unsigned long long *eccCounts) {
  GET_GPU(device, g);
  (void)counterType;
  if (!eccCounts)
    return NVML_ERROR_INVALID_ARGUMENT;
  if (g.ecc_mode == NVML_FEATURE_DISABLED)
    return NVML_ERROR_NOT_SUPPORTED;
  *eccCounts = (errorType == NVML_MEMORY_ERROR_TYPE_CORRECTED) ? g.ecc_corrected
                                                               : g.ecc_uncorrected;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, unsigned int *display) {
  GET_GPU(device, g);
  if (!display)
    return NVML_ERROR_INVALID_ARGUMENT;
  *display = g.display_mode;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, unsigned int *isActive) {
  GET_GPU(device, g);
  if (!isActive)
    return NVML_ERROR_INVALID_ARGUMENT;
  *isActive = g.display_active;
  return NVML_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Running processes                                                   */
/* ------------------------------------------------------------------ */

/*
 * NVML's process calls use the "ask twice" convention: pass infoCount as the
 * capacity of your buffer, and if it is too small you get INSUFFICIENT_SIZE
 * back with infoCount updated to what you need. A capacity of zero is the
 * documented way to query the count without allocating, and callers including
 * nvtop depend on that path returning INSUFFICIENT_SIZE rather than SUCCESS.
 */
static nvmlReturn_t fill_processes(nvmlDevice_t device, unsigned int *infoCount, void *infos,
                                   unsigned int want_type, size_t stride, bool with_mig_ids) {
  GET_GPU(device, g);
  if (!infoCount)
    return NVML_ERROR_INVALID_ARGUMENT;

  unsigned int matched = 0;
  unsigned int n = g.n_procs > GPUEMU_MAX_PROCS ? GPUEMU_MAX_PROCS : g.n_procs;
  for (unsigned int i = 0; i < n; i++)
    if (g.procs[i].type & want_type)
      matched++;

  unsigned int capacity = *infoCount;
  *infoCount = matched;

  if (matched == 0)
    return NVML_SUCCESS;
  if (!infos || capacity < matched)
    return NVML_ERROR_INSUFFICIENT_SIZE;

  unsigned int out = 0;
  for (unsigned int i = 0; i < n; i++) {
    if (!(g.procs[i].type & want_type))
      continue;
    char *slot = (char *)infos + (size_t)out * stride;
    nvmlProcessInfo_v1_t base;
    base.pid = g.procs[i].pid;
    base.usedGpuMemory = g.procs[i].used_mem;
    memcpy(slot, &base, sizeof(base));
    if (with_mig_ids) {
      unsigned int ids[2] = {g.procs[i].gi_id, g.procs[i].ci_id};
      memcpy(slot + sizeof(base), ids, sizeof(ids));
    }
    out++;
  }
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t d, unsigned int *c,
                                                                 nvmlProcessInfo_v3_t *i) {
  return fill_processes(d, c, i, GPUEMU_PROC_COMPUTE, sizeof(nvmlProcessInfo_v3_t), true);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t d, unsigned int *c,
                                                                 nvmlProcessInfo_v2_t *i) {
  return fill_processes(d, c, i, GPUEMU_PROC_COMPUTE, sizeof(nvmlProcessInfo_v2_t), true);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t d, unsigned int *c,
                                                              nvmlProcessInfo_v1_t *i) {
  return fill_processes(d, c, i, GPUEMU_PROC_COMPUTE, sizeof(nvmlProcessInfo_v1_t), false);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t d, unsigned int *c,
                                                                  nvmlProcessInfo_v3_t *i) {
  return fill_processes(d, c, i, GPUEMU_PROC_GRAPHICS, sizeof(nvmlProcessInfo_v3_t), true);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(nvmlDevice_t d, unsigned int *c,
                                                                  nvmlProcessInfo_v2_t *i) {
  return fill_processes(d, c, i, GPUEMU_PROC_GRAPHICS, sizeof(nvmlProcessInfo_v2_t), true);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice_t d, unsigned int *c,
                                                               nvmlProcessInfo_v1_t *i) {
  return fill_processes(d, c, i, GPUEMU_PROC_GRAPHICS, sizeof(nvmlProcessInfo_v1_t), false);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t d, unsigned int *c,
                                                                    nvmlProcessInfo_v3_t *i) {
  return fill_processes(d, c, i, GPUEMU_PROC_MPS, sizeof(nvmlProcessInfo_v3_t), true);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(nvmlDevice_t d, unsigned int *c,
                                                                    nvmlProcessInfo_v2_t *i) {
  return fill_processes(d, c, i, GPUEMU_PROC_MPS, sizeof(nvmlProcessInfo_v2_t), true);
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses(nvmlDevice_t d, unsigned int *c,
                                                                 nvmlProcessInfo_v1_t *i) {
  return fill_processes(d, c, i, GPUEMU_PROC_MPS, sizeof(nvmlProcessInfo_v1_t), false);
}

/*
 * Per-process utilisation. Real NVML returns samples newer than `lastSeenTimeStamp`
 * and reports NOT_FOUND when there is nothing new; nvtop treats either as "no
 * data" and moves on, so returning the current frame each time is enough.
 */
NVML_EXPORT nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device,
                                                         nvmlProcessUtilizationSample_t *utilization,
                                                         unsigned int *processSamplesCount,
                                                         unsigned long long lastSeenTimeStamp) {
  GET_GPU(device, g);
  (void)lastSeenTimeStamp;
  if (!processSamplesCount)
    return NVML_ERROR_INVALID_ARGUMENT;

  unsigned int n = g.n_procs > GPUEMU_MAX_PROCS ? GPUEMU_MAX_PROCS : g.n_procs;
  unsigned int capacity = *processSamplesCount;
  *processSamplesCount = n;

  if (n == 0)
    return NVML_ERROR_NOT_FOUND;
  if (!utilization || capacity < n)
    return NVML_ERROR_INSUFFICIENT_SIZE;

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  unsigned long long now = (unsigned long long)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000ULL;

  for (unsigned int i = 0; i < n; i++) {
    utilization[i].pid = g.procs[i].pid;
    utilization[i].timeStamp = now;
    utilization[i].smUtil = g.procs[i].sm_util;
    utilization[i].memUtil = g.procs[i].mem_util;
    utilization[i].encUtil = g.procs[i].enc_util;
    utilization[i].decUtil = g.procs[i].dec_util;
  }
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name,
                                                  unsigned int length) {
  char path[64];
  snprintf(path, sizeof(path), "/proc/%u/cmdline", pid);
  FILE *f = fopen(path, "r");
  if (!f)
    return NVML_ERROR_NOT_FOUND;
  size_t n = fread(name, 1, length - 1, f);
  fclose(f);
  name[n] = '\0';
  for (size_t i = 0; i + 1 < n; i++) /* cmdline is NUL-separated */
    if (name[i] == '\0')
      name[i] = ' ';
  return NVML_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Deliberately unsupported                                            */
/* ------------------------------------------------------------------ */

/* The L4 has no NVLink, so reporting "not supported" here is not a shortcut:
 * it is what the real card does, and nvtop hides the NVLink panel because of
 * it. Same for MIG instances, which we do not model. */
NVML_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t d, unsigned int l, unsigned int *a) {
  (void)d; (void)l; (void)a;
  return NVML_ERROR_NOT_SUPPORTED;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t d, unsigned int l,
                                                    unsigned int *v) {
  (void)d; (void)l; (void)v;
  return NVML_ERROR_NOT_SUPPORTED;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t d, int valuesCount,
                                                  nvmlFieldValue_t *values) {
  (void)d;
  if (!values || valuesCount <= 0)
    return NVML_ERROR_INVALID_ARGUMENT;
  for (int i = 0; i < valuesCount; i++)
    values[i].nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
  return NVML_SUCCESS;
}

NVML_EXPORT nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t d, unsigned int *count) {
  (void)d;
  if (count)
    *count = 0;
  return NVML_ERROR_NOT_SUPPORTED;
}
