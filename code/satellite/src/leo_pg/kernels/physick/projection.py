from __future__ import annotations

import torch


def project_onto_l1_ball(weights: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Euclidean projection of the last dimension onto a signed L1 ball."""
    radius = float(radius)
    if radius < 0.0:
        raise ValueError("L1 projection radius must be non-negative")
    if weights.ndim == 0:
        raise ValueError("weights must have at least one dimension")
    if weights.size(-1) == 0:
        return weights
    if radius == 0.0:
        return torch.zeros_like(weights)

    original_shape = weights.shape
    flat = weights.reshape(-1, original_shape[-1])
    absolute = flat.abs()
    inside = absolute.sum(dim=-1, keepdim=True) <= radius

    sorted_absolute, _ = torch.sort(absolute, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_absolute, dim=-1)
    dimensions = torch.arange(
        1,
        sorted_absolute.size(-1) + 1,
        device=weights.device,
        dtype=weights.dtype,
    ).unsqueeze(0)
    active = sorted_absolute * dimensions > (cumulative - radius)
    rho = active.sum(dim=-1).clamp_min(1) - 1
    threshold = (cumulative.gather(1, rho.unsqueeze(1)) - radius) / (
        (rho + 1).to(weights.dtype).unsqueeze(1)
    )
    projected = flat.sign() * torch.clamp(absolute - threshold, min=0.0)
    output = torch.where(inside, flat, projected)
    return output.reshape(original_shape)

