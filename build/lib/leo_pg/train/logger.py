from __future__ import annotations
class SimpleLogger:
    def __init__(self):
        self.step = 0
    def log(self, **kwargs):
        self.step += 1
        if kwargs:
            print("[LOG]", ", ".join(f"{k}={v}" for k,v in kwargs.items()))
