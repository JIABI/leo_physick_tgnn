"""Message operators used in the UAV temporal graph world model.

The three implementations share the same input and output dimensions.  Only
the edge-message map changes, which is the comparison contract in the paper.
"""

from __future__ import annotations

import math

import torch
from torch import nn


SUPPORTED_OPERATORS = ("mlp", "kan", "physick")


def project_onto_l1_ball(values: torch.Tensor, radius: float) -> torch.Tensor:
    """Exact Euclidean projection along the last axis onto a signed L1 ball."""

    radius = float(radius)
    if radius < 0.0 or not math.isfinite(radius):
        raise ValueError("L1 projection radius must be finite and non-negative")
    if values.ndim == 0 or values.size(-1) == 0:
        raise ValueError("values must have a non-empty final dimension")
    if radius == 0.0:
        return torch.zeros_like(values)
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    magnitude = flat.abs()
    inside = magnitude.sum(dim=-1, keepdim=True) <= radius
    ordered, _ = torch.sort(magnitude, dim=-1, descending=True)
    cumulative = ordered.cumsum(dim=-1)
    index = torch.arange(
        1,
        ordered.size(-1) + 1,
        dtype=ordered.dtype,
        device=ordered.device,
    ).unsqueeze(0)
    active = ordered * index > cumulative - radius
    rho = active.sum(dim=-1).clamp_min(1) - 1
    threshold = (cumulative.gather(1, rho[:, None]) - radius) / (
        rho.to(ordered.dtype)[:, None] + 1.0
    )
    projected = flat.sign() * (magnitude - threshold).clamp_min(0.0)
    return torch.where(inside, flat, projected).reshape(shape)


class MLPMessageOperator(nn.Module):
    """Two-layer edge-message MLP used as the direct learned baseline."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, edge_state: torch.Tensor) -> torch.Tensor:
        return self.network(edge_state)


class KANLinear(nn.Module):
    """Per-coordinate piecewise-linear spline layer with a linear skip path."""

    def __init__(self, input_dim: int, output_dim: int, knots: int = 16) -> None:
        super().__init__()
        if knots < 2:
            raise ValueError("KAN requires at least two knots")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.knots = int(knots)
        self.register_buffer("grid", torch.linspace(-1.0, 1.0, self.knots))
        self.values = nn.Parameter(
            torch.empty(self.output_dim, self.input_dim, self.knots)
        )
        nn.init.uniform_(self.values, -0.05, 0.05)
        self.skip = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.size(1) != self.input_dim:
            raise ValueError(
                f"KAN input must have shape [E,{self.input_dim}]"
            )
        clipped = values.clamp(-1.0, 1.0).float()
        grid = self.grid.float()
        right = torch.bucketize(clipped, grid)
        left = (right - 1).clamp(0, self.knots - 2)
        right = left + 1
        lower = grid[left]
        upper = grid[right]
        fraction = (clipped - lower) / (upper - lower + 1e-12)
        result = torch.zeros(
            clipped.size(0),
            self.output_dim,
            dtype=torch.float32,
            device=clipped.device,
        )
        for coordinate in range(self.input_dim):
            basis = self.values[:, coordinate, :]
            lower_value = basis.index_select(1, left[:, coordinate])
            upper_value = basis.index_select(1, right[:, coordinate])
            weight = fraction[:, coordinate].unsqueeze(0)
            result += ((1.0 - weight) * lower_value + weight * upper_value).T
        return result + self.skip(values.clamp(-1.0, 1.0)).float()


class KANMessageOperator(nn.Module):
    """KAN edge-message map with the 16-basis comparison configuration."""

    def __init__(self, input_dim: int, output_dim: int, knots: int) -> None:
        super().__init__()
        self.kan = KANLinear(input_dim, output_dim, knots=knots)

    def forward(self, edge_state: torch.Tensor) -> torch.Tensor:
        return self.kan(torch.tanh(edge_state))


class PhysiCKMessageOperator(nn.Module):
    """Edge-conditioned signed mixture of a learned vector-kernel bank.

    This follows Supplementary Eqs. for ``phi_in``, ``phi_desc``, the shared
    vector-kernel bank, coefficient head, signed-L1 projection and output map.
    ``gain_control=False`` is the paper's no-gain-control ablation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        latent_dim: int,
        descriptor_dim: int,
        kernel_count: int,
        kernel_hidden_dim: int,
        coefficient_hidden_dim: int,
        dropout: float,
        projection_radius: float,
        operating_clip: float,
        gain_control: bool,
    ) -> None:
        super().__init__()
        if kernel_count <= 0:
            raise ValueError("PhysiCK kernel_count must be positive")
        if projection_radius <= 0.0:
            raise ValueError("PhysiCK projection_radius must be positive")
        if operating_clip <= 0.0:
            raise ValueError("PhysiCK operating_clip must be positive")
        self.kernel_count = int(kernel_count)
        self.output_dim = int(output_dim)
        self.projection_radius = float(projection_radius)
        self.operating_clip = float(operating_clip)
        self.gain_control = bool(gain_control)
        self.input_map = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(latent_dim),
        )
        self.descriptor_map = nn.Sequential(
            nn.Linear(latent_dim, descriptor_dim),
            nn.Tanh(),
            nn.LayerNorm(descriptor_dim),
        )
        self.kernel_bank = nn.Sequential(
            nn.Linear(descriptor_dim, kernel_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(kernel_hidden_dim, kernel_count * output_dim),
        )
        self.coefficient_head = nn.Sequential(
            nn.Linear(latent_dim, coefficient_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(coefficient_hidden_dim, kernel_count),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, edge_state: torch.Tensor) -> torch.Tensor:
        if edge_state.ndim != 2:
            raise ValueError("PhysiCK edge state must be two-dimensional")
        if not bool(torch.isfinite(edge_state).all()):
            raise ValueError("PhysiCK edge state contains NaN or Inf")
        latent = self.input_map(edge_state)
        descriptor = self.descriptor_map(
            latent.clamp(-self.operating_clip, self.operating_clip)
        )
        kernels = self.kernel_bank(descriptor).reshape(
            edge_state.size(0), self.kernel_count, self.output_dim
        )
        coefficients = self.coefficient_head(latent).float()
        if self.gain_control:
            coefficients = project_onto_l1_ball(
                coefficients, self.projection_radius
            )
        message = (coefficients.to(kernels.dtype).unsqueeze(-1) * kernels).sum(1)
        result = self.output_norm(message)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("PhysiCK emitted NaN or Inf")
        return result


def build_message_operator(
    kind: str,
    *,
    input_dim: int,
    output_dim: int,
    dropout: float,
    kan_knots: int,
    physick_latent_dim: int,
    physick_descriptor_dim: int,
    physick_kernel_count: int,
    physick_kernel_hidden_dim: int,
    physick_coefficient_hidden_dim: int,
    physick_projection_radius: float,
    physick_operating_clip: float,
    physick_gain_control: bool,
) -> nn.Module:
    """Build one operator from a fully resolved model configuration."""

    selected = str(kind).strip().lower()
    if selected == "mlp":
        return MLPMessageOperator(input_dim, output_dim, dropout)
    if selected == "kan":
        return KANMessageOperator(input_dim, output_dim, kan_knots)
    if selected == "physick":
        return PhysiCKMessageOperator(
            input_dim,
            output_dim,
            latent_dim=physick_latent_dim,
            descriptor_dim=physick_descriptor_dim,
            kernel_count=physick_kernel_count,
            kernel_hidden_dim=physick_kernel_hidden_dim,
            coefficient_hidden_dim=physick_coefficient_hidden_dim,
            dropout=dropout,
            projection_radius=physick_projection_radius,
            operating_clip=physick_operating_clip,
            gain_control=physick_gain_control,
        )
    raise ValueError(f"unsupported operator {kind!r}; choose {SUPPORTED_OPERATORS}")


__all__ = [
    "KANLinear",
    "KANMessageOperator",
    "MLPMessageOperator",
    "PhysiCKMessageOperator",
    "SUPPORTED_OPERATORS",
    "build_message_operator",
    "project_onto_l1_ball",
]
