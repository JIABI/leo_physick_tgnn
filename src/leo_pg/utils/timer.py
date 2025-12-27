from __future__ import annotations
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str = "block"):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[TIMER] {name}: {dt:.3f}s")
