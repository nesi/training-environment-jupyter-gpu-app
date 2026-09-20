"""Mark which parts of a script are GPU work and which are CPU work.

On real hardware nothing needs marking: the driver knows a kernel is running
because it launched it. There is no driver here, so a script whose utilisation
figures are meant to be worth reading has to say when the GPU is supposed to be
busy.

That requirement turns out to be the lesson rather than a workaround. Most
people asking "why is my GPU only 30% utilised?" have never worked out which
parts of their program run on the GPU at all. Writing the phases down is how
you find out, and the answer is usually that the GPU is waiting: a pipeline
spending four seconds reading and preparing data for every one second of
computation will leave *any* GPU idle 80% of the time, and buying a faster one
changes nothing.

    from gpuemu.phases import cpu_phase, gpu_phase, report

    for batch in batches:
        with cpu_phase("load and prepare"):
            data = prepare(batch)
        with gpu_phase("compute"):
            step(data)

    report()
"""

from __future__ import annotations

import atexit
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

__all__ = ["cpu_phase", "gpu_phase", "report", "totals", "reset"]


@dataclass
class _Totals:
    seconds: dict[str, float] = field(default_factory=dict)
    kind: dict[str, str] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)

    def add(self, name: str, kind: str, elapsed: float) -> None:
        if name not in self.seconds:
            self.seconds[name] = 0.0
            self.kind[name] = kind
            self.order.append(name)
        self.seconds[name] += elapsed


_TOTALS = _Totals()


def _claim():
    """This process's GPU claim, or None if there is no device to talk to.

    A script should still run, and still report its phase breakdown, in a job
    that was never given a GPU - that combination is one of the things the
    exercises deliberately produce.
    """
    try:
        from .client import process_claim

        return process_claim()
    except Exception:
        return None


@contextmanager
def _phase(name: str, kind: str, util: float):
    claim = _claim()
    if claim is not None:
        try:
            claim.set_util(util)
        except Exception:
            pass
    start = time.monotonic()
    try:
        yield
    finally:
        _TOTALS.add(name, kind, time.monotonic() - start)
        if claim is not None:
            try:
                claim.set_util(0)
            except Exception:
                pass


def cpu_phase(name: str = "cpu"):
    """A stretch of work the GPU is not doing. It reports as idle throughout."""
    return _phase(name, "CPU", 0)


def gpu_phase(name: str = "gpu", util: float = 95):
    """A stretch of work the GPU is doing.

    ``util`` is what a well-fed device would show while this runs. It is a
    statement about the work, not a measurement of it - nothing here executes
    on a GPU - so keep it honest: a memory-bound kernel that really does leave
    the device half idle should say 50.
    """
    return _phase(name, "GPU", util)


def totals() -> dict[str, float]:
    """Seconds spent in each named phase so far."""
    return dict(_TOTALS.seconds)


def reset() -> None:
    global _TOTALS
    _TOTALS = _Totals()


def report(speedup: float = 2.0) -> None:
    """Print where the time went, and what a faster GPU could do about it."""
    total = sum(_TOTALS.seconds.values())
    if total <= 0:
        print("No phases were recorded.")
        return

    print()
    print("Where the time went")
    print("-" * 46)
    for name in _TOTALS.order:
        secs = _TOTALS.seconds[name]
        print(f"  {_TOTALS.kind[name]:<4} {name:<22}{secs:>7.1f}s{secs / total * 100:>7.1f}%")
    print(f"  {'':<4} {'':<22}{'-' * 8:>8}{'-' * 8:>8}")
    print(f"  {'':<4} {'total':<22}{total:>7.1f}s{100.0:>7.1f}%")

    gpu_time = sum(s for n, s in _TOTALS.seconds.items() if _TOTALS.kind[n] == "GPU")
    share = gpu_time / total * 100
    print()
    print(f"The GPU was busy for {share:.0f}% of this run.")

    # Amdahl, in the only form that matters to someone choosing a GPU: the
    # ceiling on what a faster device can do for you is set by the work that
    # is not on it.
    if gpu_time > 0:
        saved = gpu_time - gpu_time / speedup
        print(
            f"A GPU {speedup:g}x faster would cut {gpu_time:.1f}s to "
            f"{gpu_time / speedup:.1f}s, saving {saved / total * 100:.0f}% "
            f"of the total runtime."
        )
        if share < 50:
            print(
                "Most of this job is not on the GPU. Speeding up the part that "
                "is cannot help much - look at the CPU phases first."
            )
    print()


@atexit.register
def _report_on_exit() -> None:
    """Release the claim's utilisation when the script ends.

    Without this a script that exits inside a GPU phase leaves the device
    reading as busy until the claim is reaped, and the next person to run
    nvidia-smi sees load that belongs to nobody.
    """
    claim = _claim()
    if claim is not None:
        try:
            claim.set_util(0)
        except Exception:
            pass
