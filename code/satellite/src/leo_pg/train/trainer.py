from __future__ import annotations
import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import satellite_target_mse
from .logger import SimpleLogger


def _make_grad_scaler(enabled: bool):
    """Use the non-deprecated scaler when available, with a PyTorch 2.1 fallback."""
    scaler_cls = getattr(torch.amp, "GradScaler", None)
    if scaler_cls is not None:
        return scaler_cls("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)

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
    valid_schemes = {"uniform", "poly", "exp", "milestones"}
    if scheme not in valid_schemes:
        raise ValueError(f"unknown multistep weight scheme {scheme!r}; expected {sorted(valid_schemes)}")

    if T <= 0:
        return torch.zeros((0,), device=device)

    t = torch.arange(T, device=device, dtype=torch.float32)

    if scheme == "uniform":
        w = torch.ones((T,), device=device, dtype=torch.float32)

    elif scheme == "poly":
        power = float(cfg.get("power", 1.0))
        if not math.isfinite(power):
            raise ValueError("multistep_loss.power must be finite")
        w = (t + 1.0) ** power

    elif scheme == "exp":
        beta = float(cfg.get("exp_beta", 3.0))
        if not math.isfinite(beta):
            raise ValueError("multistep_loss.exp_beta must be finite")
        denom = max(1.0, float(T - 1))
        w = torch.exp(beta * (t / denom))

    elif scheme == "milestones":
        ms = cfg.get("milestones", []) or []
        # ms: list of dicts {t: int(1-based), w: float}
        # We'll construct a piecewise-linear weight curve over steps 1..T using provided anchors.
        anchors = []
        seen_steps = set()
        for a in ms:
            if not isinstance(a, dict) or "t" not in a or "w" not in a:
                raise ValueError("each milestone must be a mapping with t and w")
            tt = int(a["t"])
            ww = float(a["w"])
            if not math.isfinite(ww) or ww < 0.0:
                raise ValueError("milestone weights must be finite and non-negative")
            if not 1 <= tt <= T:
                continue
            if tt in seen_steps:
                raise ValueError(f"duplicate milestone step: {tt}")
            seen_steps.add(tt)
            anchors.append((tt - 1, ww))
        anchors = sorted(anchors, key=lambda x: x[0])
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

    if not torch.isfinite(w).all():
        raise ValueError("multistep weights are non-finite; reduce power/exp_beta")
    w = torch.clamp(w, min=0.0)
    weight_sum = torch.sum(w)
    if float(weight_sum.item()) <= 0.0:
        raise ValueError("multistep weights must have a positive sum")
    if normalize:
        w = w / weight_sum
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
                if len(out["preds"]) != len(episode["steps"]):
                    raise ValueError("model output length does not match episode steps")
                step_losses = [
                    satellite_target_mse(pred, target, step)
                    for pred, target, step in zip(out["preds"], out["ys"], episode["steps"])
                ]
                if not step_losses:
                    raise ValueError("cannot train on an empty episode")
                loss = torch.stack(step_losses).mean()
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

        use_amp = bool(multistep_loss_cfg.get("amp", False)) if multistep_loss_cfg is not None else False
        use_amp = use_amp and self.device.type == "cuda"

        if int(horizon) <= 0:
            raise ValueError("rollout training horizon must be positive")
        if tbptt_steps <= 0:
            raise ValueError("multistep_loss.tbptt_steps must be positive")

        scaler = _make_grad_scaler(enabled=use_amp)

        for ep in range(1, epochs + 1):

            total, n = 0.0, 0

            for i, episode in enumerate(dl):

                steps = episode["steps"]

                T = min(int(horizon), len(steps))

                if T <= 0:
                    continue

                w = _build_multistep_weights(T, multistep_loss_cfg, device=self.device)  # [T]

                self.opt.zero_grad(set_to_none=True)

                mem = None

                loss_acc = None  # tensor
                episode_loss = 0.0

                for t in range(T):
                    #if t % 1 ==0:
                    #    print(f"[HB] epoch={ep} t={t}/{horizon}", flush=True)

                    step = steps[t]

                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):

                        pred, y, mem = self.model.forward_step(step, mem, device=self.device)

                        lt = w[t] * satellite_target_mse(pred, y, step)
                        episode_loss += float(lt.detach().item())

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

                total += episode_loss

                n += 1

                if (i + 1) % self.log_every == 0:
                    # You can log a cheaper value: last step loss

                    self.logger.log(epoch=ep, it=i + 1, loss=float(lt.detach().item()))

            print(
                f"Epoch {ep:03d} | rollout_tf_loss={total / max(1, n):.6f} "
                f"(tbptt={tbptt_steps},amp={use_amp})"
            )
