"""Wire a process into the emulator as soon as it touches a GPU library.

Enabled by ``GPUEMU_AUTOCLAIM=1`` (the session image sets it, through a .pth
file so it runs in every interpreter). Two things happen when ``torch`` or
``numba.cuda`` is imported:

1. The process claims the device, so it appears in ``nvidia-smi`` and its CPU
   use drives the emulated utilisation.
2. For ``torch``, the CUDA shim is installed, so ``torch.cuda.is_available()``
   is True and ``.to("cuda")`` works.

The second one is the reason this exists at all. Without it a researcher's
ordinary PyTorch script finds no GPU and silently falls back to the CPU - which
is precisely the failure the workshop teaches people to *diagnose*, so having
the environment produce it spontaneously would be worse than useless.

Neither happens in a process that never imports a GPU library, so the device
does not fill up with idle Python interpreters, and neither happens when
``CUDA_VISIBLE_DEVICES`` is empty, so a job that did not ask for a GPU still
does not get one.

How the hook works
------------------
A meta-path finder cannot do the work in ``find_spec``: at that point the
module has not been executed, so there is nothing to patch, and claiming the
device there would leave a claim behind if the import then failed. Instead the
finder locates the spec that would have been used anyway and wraps its
loader's ``exec_module``, so the callback runs immediately after the module
finishes importing successfully.
"""

from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    return os.environ.get("GPUEMU_AUTOCLAIM", "").strip().lower() in {"1", "true", "yes", "on"}


def _visible() -> bool:
    """Respect CUDA_VISIBLE_DEVICES, so a job without a GPU stays without one."""
    from .client import visible_devices

    vis = visible_devices()
    return vis is None or bool(vis)


def _do_claim() -> None:
    from . import client

    try:
        client.process_claim(device=0, util="auto")
    except Exception:
        # The emulator must never be the reason a learner's import fails.
        pass


def _on_imported(name: str, module) -> None:
    """Called once, right after a watched module has finished importing."""
    try:
        if name == "torch":
            from . import torch_shim

            torch_shim.install(module)
    except Exception:
        pass
    _do_claim()


class _ImportHook:
    """Watches for a GPU library and hooks the end of its import."""

    WATCHED = ("torch", "numba.cuda")

    def __init__(self):
        self._seen: set[str] = set()

    def find_module(self, fullname, path=None):  # legacy API, harmless
        return None

    def _delegate(self, fullname, path, target):
        """The spec Python would have used if this finder were not installed."""
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            try:
                spec = find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is not None:
                return spec
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.WATCHED or fullname in self._seen:
            return None
        # Mark it seen before delegating: _delegate walks the rest of
        # sys.meta_path, and anything that re-enters must not loop.
        self._seen.add(fullname)

        try:
            spec = self._delegate(fullname, path, target)
            loader = getattr(spec, "loader", None)
            if loader is None:
                return spec
            original = loader.exec_module

            def exec_module(module, _original=original, _name=fullname):
                _original(module)
                _on_imported(_name, module)

            loader.exec_module = exec_module
            return spec
        except Exception:
            # Returning None puts the import back on its normal path, so a
            # failure here costs the shim, not the import.
            return None


def install() -> None:
    if not _enabled() or not _visible():
        return

    # Already imported - common in a notebook kernel that preloads libraries,
    # and in anything that imports gpuemu after torch rather than before.
    already = [name for name in _ImportHook.WATCHED if name in sys.modules]
    if already:
        for name in already:
            _on_imported(name, sys.modules[name])
        return

    sys.meta_path.insert(0, _ImportHook())
