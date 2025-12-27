from __future__ import annotations
import torch

def save_ckpt(path: str, model, opt=None, meta=None):
    payload = {"model": model.state_dict(), "meta": meta or {}}
    if opt is not None:
        payload["opt"] = opt.state_dict()
    torch.save(payload, path)

def load_ckpt(path: str, model, opt=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if opt is not None and "opt" in payload:
        opt.load_state_dict(payload["opt"])
    return payload.get("meta", {})
