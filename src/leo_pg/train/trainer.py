from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import mse_loss
from .logger import SimpleLogger

def _build_multistep_weights(T: int, cfg: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Build non-negative weights w[0..T-1] for multi-step rollout loss.
    Supported schemes:
      - uniform: all ones
      - poly: w_t ∝ (t+1)^power
      - exp:  w_t ∝ exp(beta * t/(T-1))
      - milestones: sparse weights on selected 1-based steps, linear-interp elsewhere (fallback to uniform if empty)
    """
    cfg = cfg or {}
    scheme = str(cfg.get("scheme", "uniform")).lower()
    normalize = bool(cfg.get("normalize", True))

    if T <= 0:
        return torch.zeros((0,), device=device)

    t = torch.arange(T, device=device, dtype=torch.float32)

    if scheme == "uniform":
        w = torch.ones((T,), device=device, dtype=torch.float32)

    elif scheme == "poly":
        power = float(cfg.get("power", 1.0))
        w = (t + 1.0) ** power

    elif scheme == "exp":
        beta = float(cfg.get("exp_beta", 3.0))
        denom = max(1.0, float(T - 1))
        w = torch.exp(beta * (t / denom))

    elif scheme == "milestones":
        ms = cfg.get("milestones", []) or []
        # ms: list of dicts {t: int(1-based), w: float}
        # We'll construct a piecewise-linear weight curve over steps 1..T using provided anchors.
        anchors = []
        for a in ms:
            try:
                tt = int(a.get("t"))
                ww = float(a.get("w"))
                if 1 <= tt <= T and ww >= 0:
                    anchors.append((tt - 1, ww))  # convert to 0-based
            except Exception:
                pass
        anchors = sorted(set(anchors), key=lambda x: x[0])
        if len(anchors) == 0:
            w = torch.ones((T,), device=device, dtype=torch.float32)
        else:
            # if first/last not provided, extend with boundary values
            if anchors[0][0] != 0:
                anchors = [(0, anchors[0][1])] + anchors
            if anchors[-1][0] != T - 1:
                anchors = anchors + [(T - 1, anchors[-1][1])]
            w = torch.zeros((T,), device=device, dtype=torch.float32)
            for (i0, w0), (i1, w1) in zip(anchors[:-1], anchors[1:]):
                if i1 == i0:
                    w[i0] = w0
                    continue
                seg_t = torch.arange(i0, i1 + 1, device=device, dtype=torch.float32)
                alpha = (seg_t - float(i0)) / float(i1 - i0)
                w[i0:i1 + 1] = w0 * (1 - alpha) + w1 * alpha

    else:
        # unknown -> uniform
        w = torch.ones((T,), device=device, dtype=torch.float32)

    # Avoid zero-sum weights
    w = torch.clamp(w, min=0.0)
    if normalize:
        s = torch.sum(w)
        if float(s.item()) > 0:
            w = w / s
    return w


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
        for ep in range(1, epochs + 1):
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
                if (i + 1) % self.log_every == 0:
                    self.logger.log(epoch=ep, it=i + 1, loss=float(loss.item()))
            print(f"Epoch {ep:03d} | loss={total / max(1, n):.6f}")

    def train_rollout_teacher_forcing(

            self,

            dl: DataLoader,

            epochs: int,

            horizon: int = 30,

            multistep_loss_cfg: Optional[Dict[str, Any]] = None,

    ) -> None:

        """

        Teacher-forcing rollout training with Truncated BPTT (TBPTT).

        This prevents storing the full computation graph for long horizons.

        """

        self.model.train()

        # TBPTT length (default 20). You can add to cfg later; safe fallback here.

        tbptt_steps = 20

        if multistep_loss_cfg is not None:
            tbptt_steps = int(multistep_loss_cfg.get("tbptt_steps", tbptt_steps))

        use_amp = False

        if multistep_loss_cfg is not None:
            use_amp = bool(multistep_loss_cfg.get("amp", False))

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for ep in range(1, epochs + 1):

            total, n = 0.0, 0

            for i, episode in enumerate(dl):

                steps = episode["steps"]

                T = min(int(horizon), len(steps))

                w = _build_multistep_weights(T, multistep_loss_cfg, device=self.device)  # [T]

                self.opt.zero_grad(set_to_none=True)

                mem = None

                loss_acc = None  # tensor

                for t in range(T):
                    #if t % 1 ==0:
                    #    print(f"[HB] epoch={ep} t={t}/{horizon}", flush=True)

                    step = steps[t]

                    with torch.cuda.amp.autocast(enabled=use_amp):

                        pred, y, mem = self.model.forward_step(step, mem, device=self.device)

                        lt = w[t] * mse_loss(pred, y)

                        loss_acc = lt if loss_acc is None else (loss_acc + lt)

                    # TBPTT boundary or end

                    if ((t + 1) % tbptt_steps == 0) or (t == T - 1):

                        if use_amp:

                            scaler.scale(loss_acc).backward()

                            scaler.unscale_(self.opt)

                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)

                            scaler.step(self.opt)

                            scaler.update()

                        else:

                            loss_acc.backward()

                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)

                            self.opt.step()

                        self.opt.zero_grad(set_to_none=True)

                        # Detach memory to truncate graph

                        mem = mem.detach()

                        loss_acc = None

                # logging: use last computed scalar approx

                total += float(w.mean().item())  # placeholder scalar; optional

                n += 1

                if (i + 1) % self.log_every == 0:
                    # You can log a cheaper value: last step loss

                    self.logger.log(epoch=ep, it=i + 1, loss=float(lt.detach().item()))

            print(f"Epoch {ep:03d} | rollout_tf(tbptt={tbptt_steps},amp={use_amp}) done")


