from __future__ import annotations

import math

import torch
import torch.nn as nn


class HarmonicTimeEncoder(nn.Module):
    """Fixed sinusoidal encoding for a scalar decision epoch.

    Frequencies are log-spaced and stored as a buffer, so the configured output
    width is an architectural contract without introducing an undocumented
    learned clock.  The following linear projection in :class:`TGN` learns how
    the encoded epoch modulates edge messages.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if isinstance(dimension, bool) or int(dimension) != dimension:
            raise TypeError("time-encoding dimension must be an integer")
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise ValueError("time-encoding dimension must be positive")
        frequency_count = (self.dimension + 1) // 2
        denominator = max(1, frequency_count - 1)
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(frequency_count, dtype=torch.float32)
            / float(denominator)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(
        self,
        epoch: int | float | torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = torch.as_tensor(epoch, device=device, dtype=dtype)
        if value.numel() != 1 or not bool(torch.isfinite(value).all()):
            raise ValueError("step.t must be one finite scalar")
        value = value.reshape(())
        if bool(value < 0):
            raise ValueError("step.t must be non-negative")
        angles = value * self.frequencies.to(device=device, dtype=dtype)
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=0)
        return encoded[: self.dimension]


__all__ = ["HarmonicTimeEncoder"]
