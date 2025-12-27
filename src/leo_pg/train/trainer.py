from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import mse_loss
from .logger import SimpleLogger

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, lr: float, weight_decay: float, clip_grad_norm: float, log_every: int = 20):
        self.model = model
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.clip = clip_grad_norm
        self.logger = SimpleLogger()
        self.log_every = log_every

    def train_one_step(self, dl: DataLoader, epochs: int) -> None:
        self.model.train()
        for ep in range(1, epochs+1):
            total = 0.0
            n = 0
            for i, episode in enumerate(dl):
                out = self.model.forward_episode(episode, device=self.device)
                # Use last step by default to avoid overly weighting long episodes
                pred = out["preds"][-1]
                y = out["ys"][-1]
                loss = mse_loss(pred, y)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                self.opt.step()
                total += float(loss.item())
                n += 1
                if (i+1) % self.log_every == 0:
                    self.logger.log(epoch=ep, it=i+1, loss=float(loss.item()))
            print(f"Epoch {ep:03d} | loss={total/max(1,n):.6f}")

    def train_rollout_teacher_forcing(self, dl: DataLoader, epochs: int, horizon: int = 30) -> None:
        self.model.train()
        for ep in range(1, epochs+1):
            total = 0.0
            n = 0
            for i, episode in enumerate(dl):
                out = self.model.forward_episode(episode, device=self.device)
                preds = out["preds"]
                ys = out["ys"]
                T = min(len(preds), horizon)
                loss = 0.0
                for t in range(T):
                    loss = loss + mse_loss(preds[t], ys[t])
                loss = loss / max(1, T)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                self.opt.step()
                total += float(loss.item())
                n += 1
            print(f"Epoch {ep:03d} | rollout_tf loss={total/max(1,n):.6f}")
