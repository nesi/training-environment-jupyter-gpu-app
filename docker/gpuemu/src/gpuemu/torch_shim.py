"""Make PyTorch's CUDA API work against the emulated device.

Import this before ``torch`` is used and CUDA code runs unchanged: ``.cuda()``,
``.to("cuda")``, ``device="cuda"`` and the ``torch.cuda`` namespace all work,
``torch.cuda.is_available()`` is True, and the device identifies itself as an L4
with the right capability and memory.

What is real and what is not
----------------------------
The tensors live in host memory and the arithmetic runs on the CPU. What the
shim adds is correct *bookkeeping*: every tensor placed on "cuda" is counted
against the emulated card's 24 GB, that total is reported to NVML so nvidia-smi
and nvtop show it, and exceeding it raises ``torch.cuda.OutOfMemoryError`` with
the message PyTorch would really produce. Learners therefore hit real memory
limits and read real OOM messages, which is the part of GPU memory management
worth teaching; they just do not get the speedup.

Consequences worth knowing, and worth telling learners:

* ``tensor.device`` reports ``cpu``, because that is where the data is. Set
  ``GPUEMU_TORCH_SPOOF_DEVICE=1`` to have it report ``cuda:0`` instead, for
  demonstrations where the illusion matters more than the accuracy.
* Anything that compiles or loads real device code - custom CUDA extensions,
  ``torch.compile`` with a CUDA backend, Triton kernels, bitsandbytes - cannot
  work and will fail at the point it tries.
* Timings are CPU timings. Never use this environment to teach that one
  approach is faster than another on a GPU.
"""

from __future__ import annotations

import os
import sys
import warnings

from . import client, spec

_INSTALLED = False


def _spoof_device() -> bool:
    return os.environ.get("GPUEMU_TORCH_SPOOF_DEVICE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def install(torch=None) -> None:
    """Patch ``torch`` in place. Safe to call more than once."""
    global _INSTALLED
    if _INSTALLED:
        return

    if torch is None:
        import torch as torch_mod

        torch = torch_mod

    if torch.cuda.is_available():
        # A real GPU is present, so the emulator has no business here.
        warnings.warn(
            "gpuemu.torch_shim: real CUDA device detected, leaving torch alone",
            stacklevel=2,
        )
        _INSTALLED = True
        return

    # No visible device means no device, and torch.cuda.is_available() must stay
    # False. This is the whole point of the "forgot --gpus-per-node" lesson: a
    # job that did not ask for a GPU has to fall back to the CPU here exactly as
    # it would on the cluster, rather than being quietly handed one.
    visible = client.visible_devices()
    if visible is not None and not visible:
        _INSTALLED = True
        return

    dev = spec.selected_device()
    n_devices = spec.device_count() if visible is None else len(visible)
    claim = client.process_claim(device=0, util="auto")

    _patch_cuda_namespace(torch, dev, n_devices, claim)
    _patch_placement(torch, claim)
    if _spoof_device():
        _patch_device_property(torch)

    _INSTALLED = True


# ------------------------------------------------------------------ memory


class _MemoryLedger:
    """Tracks what this process has 'on the GPU' and mirrors it to the emulator."""

    def __init__(self, claim: client.Claim):
        self.claim = claim
        self.allocated = 0
        self.peak = 0

    def add(self, nbytes: int) -> None:
        if nbytes <= 0:
            return
        try:
            self.claim.set_memory(self.allocated + nbytes)
        except client.OutOfMemoryError as exc:
            raise _oom_error(exc) from None
        self.allocated += nbytes
        self.peak = max(self.peak, self.allocated)

    def remove(self, nbytes: int) -> None:
        self.allocated = max(0, self.allocated - max(0, nbytes))
        try:
            self.claim.set_memory(self.allocated)
        except client.OutOfMemoryError:
            pass

    def reset_peak(self) -> None:
        self.peak = self.allocated


def _oom_error(exc: Exception):
    """Raise the error PyTorch would raise, so tutorials on handling it apply."""
    import torch

    cls = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return cls(str(exc))


def _tensor_bytes(t) -> int:
    try:
        return t.untyped_storage().nbytes()
    except Exception:
        try:
            return t.numel() * t.element_size()
        except Exception:
            return 0


# ------------------------------------------------------------ cuda namespace


def _patch_cuda_namespace(torch, dev: spec.DeviceSpec, n_devices: int, claim) -> None:
    ledger = _MemoryLedger(claim)
    torch.cuda._gpuemu_ledger = ledger
    current = {"index": 0}

    class _DeviceProperties:
        """Stands in for torch.cuda's device properties struct."""

        def __init__(self):
            self.name = dev.name
            self.major = dev.cc_major
            self.minor = dev.cc_minor
            self.total_memory = dev.mem_total_bytes
            self.multi_processor_count = dev.sm_count
            self.is_integrated = False
            self.is_multi_gpu_board = False
            self.max_threads_per_multi_processor = 1536
            self.warp_size = 32
            self.L2_cache_size = 48 * 1024 * 1024
            self.uuid = spec.make_uuid(0)

        def __repr__(self):
            return (
                f"_CudaDeviceProperties(name='{self.name}', "
                f"major={self.major}, minor={self.minor}, "
                f"total_memory={self.total_memory // (1024 * 1024)}MB, "
                f"multi_processor_count={self.multi_processor_count})"
            )

    props = _DeviceProperties()

    def _index(device=None) -> int:
        if device is None:
            return current["index"]
        if isinstance(device, int):
            return device
        idx = getattr(device, "index", None)
        return 0 if idx is None else idx

    class _Stream:
        """A no-op stream. Ordering is trivially satisfied when work is synchronous."""

        def __init__(self, *a, **k):
            self.device = torch.device("cuda", current["index"])

        def synchronize(self):
            return None

        def wait_stream(self, other):
            return None

        def query(self):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Event:
        """Timing events that measure wall-clock time on the CPU."""

        def __init__(self, enable_timing=False, *a, **k):
            self.enable_timing = enable_timing
            self._t = None

        def record(self, stream=None):
            import time

            self._t = time.perf_counter()

        def synchronize(self):
            return None

        def query(self):
            return True

        def elapsed_time(self, other):
            if self._t is None or other._t is None:
                raise RuntimeError("Event was not recorded")
            return (other._t - self._t) * 1000.0

    class _device_ctx:
        def __init__(self, device):
            self.idx = _index(device)
            self.prev = current["index"]

        def __enter__(self):
            current["index"] = self.idx
            return self

        def __exit__(self, *exc):
            current["index"] = self.prev
            return False

    patches = {
        "is_available": lambda: True,
        "is_initialized": lambda: True,
        "init": lambda: None,
        "device_count": lambda: n_devices,
        "current_device": lambda: current["index"],
        "set_device": lambda d: current.__setitem__("index", _index(d)),
        "get_device_name": lambda device=None: dev.name,
        "get_device_capability": lambda device=None: (dev.cc_major, dev.cc_minor),
        "get_device_properties": lambda device=None: props,
        "get_arch_list": lambda: [f"sm_{dev.cc_major}{dev.cc_minor}"],
        "get_device_index": _index,
        "synchronize": lambda device=None: None,
        "empty_cache": lambda: None,
        "ipc_collect": lambda: None,
        "memory_allocated": lambda device=None: ledger.allocated,
        "max_memory_allocated": lambda device=None: ledger.peak,
        "memory_reserved": lambda device=None: ledger.allocated,
        "max_memory_reserved": lambda device=None: ledger.peak,
        "memory_cached": lambda device=None: ledger.allocated,
        "reset_peak_memory_stats": lambda device=None: ledger.reset_peak(),
        "reset_max_memory_allocated": lambda device=None: ledger.reset_peak(),
        "mem_get_info": lambda device=None: (
            max(0, dev.mem_total_bytes - dev.mem_reserved_bytes - ledger.allocated),
            dev.mem_total_bytes,
        ),
        "device": _device_ctx,
        "Stream": _Stream,
        "Event": _Event,
        "current_stream": lambda device=None: _Stream(),
        "default_stream": lambda device=None: _Stream(),
        "stream": lambda s=None: s if s is not None else _Stream(),
        "manual_seed": lambda seed: torch.manual_seed(seed),
        "manual_seed_all": lambda seed: torch.manual_seed(seed),
        "is_bf16_supported": lambda including_emulation=True: True,
    }
    for name, value in patches.items():
        try:
            setattr(torch.cuda, name, value)
        except Exception:
            pass

    def memory_summary(device=None, abbreviated=False):
        mib = 1024 * 1024
        return (
            "|===========================================================================|\n"
            "|                  PyTorch CUDA memory summary (gpuemu)                     |\n"
            "|---------------------------------------------------------------------------|\n"
            f"| Allocated memory      | {ledger.allocated // mib:>8} MiB                          |\n"
            f"| Peak allocated        | {ledger.peak // mib:>8} MiB                          |\n"
            f"| Device total          | {dev.mem_total_bytes // mib:>8} MiB                          |\n"
            "|===========================================================================|"
        )

    torch.cuda.memory_summary = memory_summary

    # torch.backends.cudnn is queried by plenty of tutorial code.
    try:
        torch.backends.cudnn.is_available = lambda: True
        torch.backends.cuda.is_built = lambda: True
    except Exception:
        pass


# --------------------------------------------------------------- placement


def _rewrite_device(device):
    """Map any CUDA device reference onto the CPU, leaving others alone."""
    if device is None:
        return None
    if isinstance(device, str):
        return "cpu" if device.startswith("cuda") else device
    type_ = getattr(device, "type", None)
    if type_ == "cuda":
        return "cpu"
    return device


def _patch_placement(torch, claim) -> None:
    ledger = torch.cuda._gpuemu_ledger

    def _is_cuda(device) -> bool:
        if isinstance(device, str):
            return device.startswith("cuda")
        return getattr(device, "type", None) == "cuda"

    # -- Tensor.to / .cuda -------------------------------------------
    # .cuda() is reimplemented in terms of .to("cpu") rather than wrapping the
    # original, which would try to initialise a CUDA context that is not there.
    orig_to = torch.Tensor.to

    def tensor_to(self, *args, **kwargs):
        moved_to_cuda = False
        if "device" in kwargs and _is_cuda(kwargs["device"]):
            moved_to_cuda = True
            kwargs["device"] = _rewrite_device(kwargs["device"])
        new_args = []
        for a in args:
            if _is_cuda(a):
                moved_to_cuda = True
                new_args.append(_rewrite_device(a))
            else:
                new_args.append(a)
        out = orig_to(self, *new_args, **kwargs)
        if moved_to_cuda:
            # Count it even when torch hands back the same object. The move is
            # a no-op here only because the data never left host memory; on a
            # real device it would be a fresh allocation, and the whole point
            # of the accounting is to reflect what the device would hold.
            ledger.add(_tensor_bytes(out))
        return out

    def tensor_cuda(self, device=None, non_blocking=False, **kwargs):
        out = orig_to(self, "cpu")
        ledger.add(_tensor_bytes(out))
        return out

    torch.Tensor.to = tensor_to
    torch.Tensor.cuda = tensor_cuda
    torch.Tensor.is_cuda = property(lambda self: _spoof_device())

    # -- Module.to / .cuda -------------------------------------------
    # Parameters are already in host memory, so .cuda() only has to account for
    # them; there is nothing to move.
    orig_module_to = torch.nn.Module.to

    def module_to(self, *args, **kwargs):
        moved = any(_is_cuda(a) for a in args) or _is_cuda(kwargs.get("device"))
        args = tuple(_rewrite_device(a) if _is_cuda(a) else a for a in args)
        if "device" in kwargs:
            kwargs["device"] = _rewrite_device(kwargs["device"])
        out = orig_module_to(self, *args, **kwargs)
        if moved:
            ledger.add(sum(_tensor_bytes(p) for p in self.parameters()))
            ledger.add(sum(_tensor_bytes(b) for b in self.buffers()))
        return out

    def module_cuda(self, device=None):
        ledger.add(sum(_tensor_bytes(p) for p in self.parameters()))
        ledger.add(sum(_tensor_bytes(b) for b in self.buffers()))
        return self

    torch.nn.Module.to = module_to
    torch.nn.Module.cuda = module_cuda

    # -- factory functions -------------------------------------------
    # Anything taking device= can be handed "cuda" directly.
    factories = [
        "tensor", "zeros", "ones", "empty", "full", "rand", "randn", "randint",
        "arange", "linspace", "logspace", "eye", "zeros_like", "ones_like",
        "empty_like", "full_like", "rand_like", "randn_like", "as_tensor",
        "from_numpy", "normal", "randperm",
    ]
    for fname in factories:
        fn = getattr(torch, fname, None)
        if fn is None:
            continue
        setattr(torch, fname, _wrap_factory(fn, ledger))

    # torch.load(map_location="cuda") is how saved models get restored.
    orig_load = torch.load

    def load(*args, **kwargs):
        if "map_location" in kwargs and _is_cuda(kwargs["map_location"]):
            kwargs["map_location"] = "cpu"
        return orig_load(*args, **kwargs)

    torch.load = load

    # autocast on "cuda" should behave, but bf16/fp16 on CPU is slow and
    # sometimes unsupported, so let it through as a no-op context.
    try:
        orig_autocast = torch.autocast

        class autocast(orig_autocast):
            def __init__(self, device_type="cuda", *a, **k):
                if device_type == "cuda":
                    device_type = "cpu"
                    k.pop("dtype", None)
                    k["enabled"] = False
                super().__init__(device_type, *a, **k)

        torch.autocast = autocast
        torch.cuda.amp.autocast = lambda *a, **k: autocast("cuda")
    except Exception:
        pass


def _estimate_bytes(fn_name: str, args, kwargs) -> int | None:
    """Best-effort size of the tensor a factory call is about to produce.

    Returns None when the shape cannot be worked out, in which case the caller
    falls back to allocating first and accounting afterwards.
    """
    import torch

    # The *_like family copies its argument's shape.
    if fn_name.endswith("_like"):
        if args and hasattr(args[0], "numel"):
            dtype = kwargs.get("dtype") or getattr(args[0], "dtype", None)
            try:
                elem = torch.empty(0, dtype=dtype).element_size()
            except Exception:
                return None
            return args[0].numel() * elem
        return None

    # Factories taking explicit sizes: either zeros(2, 3) or zeros((2, 3)).
    sizes = kwargs.get("size")
    if sizes is None:
        if len(args) == 1 and isinstance(args[0], (tuple, list)) and all(
            isinstance(v, int) for v in args[0]
        ):
            sizes = args[0]
        elif args and all(isinstance(a, int) for a in args):
            sizes = args
    if sizes is None:
        return None

    numel = 1
    for dim in sizes:
        if not isinstance(dim, int) or dim < 0:
            return None
        numel *= dim
    try:
        elem = torch.empty(0, dtype=kwargs.get("dtype")).element_size()
    except Exception:
        return None
    return numel * elem


def _wrap_factory(fn, ledger):
    fn_name = getattr(fn, "__name__", "")

    def wrapper(*args, **kwargs):
        device = kwargs.get("device")
        is_cuda = device is not None and (
            (isinstance(device, str) and device.startswith("cuda"))
            or getattr(device, "type", None) == "cuda"
        )
        if not is_cuda:
            return fn(*args, **kwargs)

        kwargs["device"] = "cpu"

        # Check the emulated limit *before* allocating. Allocating first would
        # mean a learner demonstrating an out-of-memory error really does ask
        # the host for 40 GB, and the container gets OOM-killed instead of
        # raising the error the exercise is about.
        estimated = _estimate_bytes(fn_name, args, kwargs)
        if estimated is not None:
            ledger.add(estimated)
            try:
                out = fn(*args, **kwargs)
            except Exception:
                ledger.remove(estimated)
                raise
            # Reconcile against what was really produced, in case the estimate
            # was off (an unusual dtype, say).
            actual = _tensor_bytes(out)
            if actual != estimated:
                ledger.remove(estimated)
                ledger.add(actual)
            return out

        out = fn(*args, **kwargs)
        ledger.add(_tensor_bytes(out))
        return out

    wrapper.__name__ = fn_name or "wrapped"
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def _patch_device_property(torch) -> None:
    """Make tensors claim to be on cuda:0.

    Off by default. It makes ``print(x.device)`` say ``cuda:0`` when the data is
    really in host memory, which is a lie that helps a demo look right and
    hinders anyone debugging. Opt in with GPUEMU_TORCH_SPOOF_DEVICE=1.
    """
    try:
        fake = torch.device("cuda", 0)
        torch.Tensor.device = property(lambda self: fake)
    except Exception as exc:  # pragma: no cover - depends on torch internals
        warnings.warn(
            f"gpuemu.torch_shim: could not spoof tensor.device ({exc}); "
            "tensors will report cpu",
            stacklevel=2,
        )


# Importing the module is enough; no need to call install() by hand.
if "torch" in sys.modules:
    install(sys.modules["torch"])
else:
    try:
        install()
    except ImportError:
        # torch is not installed. Harmless: nothing to patch.
        pass
