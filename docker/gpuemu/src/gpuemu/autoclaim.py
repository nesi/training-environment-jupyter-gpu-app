"""Register a process with the emulator as soon as it touches a GPU library.

Enabled by ``GPUEMU_AUTOCLAIM=1`` (the session image sets it). Without it, a
learner's notebook would only appear in nvidia-smi if they remembered to open a
``gpuemu.gpu()`` block, which is exactly the sort of environment-specific
boilerplate that gets in the way of teaching.

The hook watches for ``torch`` or ``numba.cuda`` being imported and, when one
appears, claims the device on this process's behalf. Utilisation then follows
the process's real CPU use, so a notebook running a training loop shows up as a
busy GPU with no extra ceremony.

A process that never imports a GPU library never claims anything, so the device
does not fill up with idle Python kernels.
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


class _ImportHook:
    """A meta-path finder that only watches; it never claims to load anything."""

    WATCHED = ("torch", "numba.cuda")

    def __init__(self):
        self._fired = False

    def find_module(self, fullname, path=None):  # legacy API, harmless
        return None

    def find_spec(self, fullname, path=None, target=None):
        if not self._fired and fullname in self.WATCHED:
            self._fired = True
            # Defer until the import finishes, otherwise we would claim the
            # device and then have the import fail.
            _schedule_claim(fullname)
        return None


_pending: list[str] = []


def _schedule_claim(module_name: str) -> None:
    _pending.append(module_name)


def _do_claim() -> None:
    from . import client

    try:
        client.process_claim(device=0, util="auto")
    except Exception:
        # The emulator must never be the reason a learner's import fails.
        pass


def install() -> None:
    if not _enabled() or not _visible():
        return

    hook = _ImportHook()
    sys.meta_path.insert(0, hook)

    # If the library is already imported (common in a notebook kernel that
    # preloads it), claim straight away.
    if any(name in sys.modules for name in _ImportHook.WATCHED):
        _do_claim()
        return

    # Otherwise attach to the first opportunity after an import completes.
    import atexit

    def _flush():
        if _pending:
            _do_claim()

    # Claim on the next interpreter event rather than inside find_spec.
    try:
        import threading

        def waiter():
            import time

            for _ in range(600):  # give a notebook 60s to import something
                if _pending:
                    _do_claim()
                    return
                time.sleep(0.1)

        t = threading.Thread(target=waiter, daemon=True, name="gpuemu-autoclaim")
        t.start()
    except Exception:
        atexit.register(_flush)
