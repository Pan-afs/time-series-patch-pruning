"""Device-agnostic measurements of inference cost.

``PeakMemoryTracker`` records the peak number of bytes held by live tensor
storages created inside its context. It works on CPU (where PyTorch exposes no
allocator statistics) and gives a deterministic number that does not depend on
allocator caching, which makes pruning levels directly comparable.
"""
import time
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.flop_counter import FlopCounterMode


class PeakMemoryTracker(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.live = {}
        self.current = 0
        self.peak = 0

    def _release(self, key):
        self.current -= self.live.pop(key, 0)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_leaves(out):
            if not isinstance(t, torch.Tensor):
                continue
            storage = t.untyped_storage()
            key = storage._cdata
            if key in self.live:
                continue
            nbytes = storage.nbytes()
            self.live[key] = nbytes
            self.current += nbytes
            self.peak = max(self.peak, self.current)
            weakref.finalize(storage, self._release, key)
        return out


@torch.inference_mode()
def peak_memory_bytes(fn) -> int:
    with PeakMemoryTracker() as tracker:
        out = fn()
        del out
    return tracker.peak


@torch.inference_mode()
def count_flops(fn) -> int:
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops()


@torch.inference_mode()
def measure_latency(fn, warmup: int = 10, repeats: int = 50, device=None):
    """Median and inter-quartile range of wall-clock latency in milliseconds."""
    sync = torch.cuda.synchronize if device is not None and device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    t = torch.tensor(times)
    return t.median().item(), (t.quantile(0.75) - t.quantile(0.25)).item()
