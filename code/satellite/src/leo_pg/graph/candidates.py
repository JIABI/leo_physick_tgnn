from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ElevationCandidates:
    """Geometry-only Top-k candidate graph in a stable edge order."""

    edge_index: torch.Tensor
    edge_ids: torch.Tensor
    elevation_deg: torch.Tensor
    distance: torch.Tensor


def pairwise_elevation_deg(user_pos: torch.Tensor, sat_pos: torch.Tensor) -> torch.Tensor:
    """Return local-horizon elevation angles for every user/satellite pair.

    Positions must share an Earth-centred Cartesian frame. The user radial
    direction defines the local zenith; an elevation of zero lies on the local
    horizon.
    """
    if user_pos.ndim != 2 or user_pos.size(1) != 3:
        raise ValueError("user_pos must have shape [K,3]")
    if sat_pos.ndim != 2 or sat_pos.size(1) != 3:
        raise ValueError("sat_pos must have shape [S,3]")
    if user_pos.size(0) == 0 or sat_pos.size(0) == 0:
        return torch.empty(
            (user_pos.size(0), sat_pos.size(0)),
            dtype=user_pos.dtype,
            device=user_pos.device,
        )
    if user_pos.device != sat_pos.device:
        raise ValueError("user_pos and sat_pos must use the same device")
    if not torch.isfinite(user_pos).all() or not torch.isfinite(sat_pos).all():
        raise ValueError("positions must be finite")
    user_radius = torch.linalg.vector_norm(user_pos, dim=1)
    if torch.any(user_radius <= 0):
        raise ValueError("user positions must have a non-zero Earth-centred radius")
    zenith = user_pos / user_radius[:, None]
    line_of_sight = sat_pos[None, :, :] - user_pos[:, None, :]
    distance = torch.linalg.vector_norm(line_of_sight, dim=-1)
    if torch.any(distance <= 0):
        raise ValueError("user and satellite positions must not coincide")
    los_unit = line_of_sight / distance[..., None]
    sin_elevation = torch.einsum("ksd,kd->ks", los_unit, zenith).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.asin(sin_elevation))


def elevation_for_edges(
    user_pos: torch.Tensor,
    sat_pos: torch.Tensor,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    """Evaluate elevation for stable local ``(user, satellite)`` edge ids."""
    if edge_ids.ndim != 2 or edge_ids.size(1) != 2 or edge_ids.dtype != torch.long:
        raise ValueError("edge_ids must have shape [E,2] and dtype long")
    if edge_ids.numel() == 0:
        return torch.empty(0, dtype=user_pos.dtype, device=user_pos.device)
    if int(edge_ids[:, 0].min()) < 0 or int(edge_ids[:, 0].max()) >= user_pos.size(0):
        raise ValueError("edge_ids contain an invalid user id")
    if int(edge_ids[:, 1].min()) < 0 or int(edge_ids[:, 1].max()) >= sat_pos.size(0):
        raise ValueError("edge_ids contain an invalid satellite id")
    elevation = pairwise_elevation_deg(user_pos, sat_pos)
    return elevation[edge_ids[:, 0], edge_ids[:, 1]]


def build_elevation_candidates(
    user_pos: torch.Tensor,
    sat_pos: torch.Tensor,
    *,
    minimum_elevation_deg: float = 10.0,
    topk: int = 6,
    user_offset: int = 0,
    satellite_offset: int | None = None,
) -> ElevationCandidates:
    """Apply the hard elevation mask, then stable Top-k-by-elevation ranking.

    Candidate construction never uses SINR, load, intensity, or a learned
    quantity. Equal elevations are resolved by smaller local satellite id.
    """
    if topk <= 0:
        raise ValueError("topk must be positive")
    if not torch.isfinite(torch.tensor(float(minimum_elevation_deg))):
        raise ValueError("minimum_elevation_deg must be finite")
    K, S = int(user_pos.size(0)), int(sat_pos.size(0))
    satellite_offset = K if satellite_offset is None else int(satellite_offset)
    elevation = pairwise_elevation_deg(user_pos, sat_pos)
    distance_matrix = torch.cdist(user_pos, sat_pos)
    user_parts = []
    satellite_parts = []
    elevation_parts = []
    distance_parts = []
    for user in range(K):
        visible = torch.nonzero(
            elevation[user] >= float(minimum_elevation_deg), as_tuple=False
        ).flatten()
        if visible.numel() == 0:
            continue
        # visible is satellite-id ordered. Stable sorting therefore supplies
        # the manuscript's smaller-id tie break for equal elevation.
        order = torch.argsort(elevation[user, visible], descending=True, stable=True)
        selected = visible[order[: min(topk, visible.numel())]]
        user_parts.append(torch.full_like(selected, user))
        satellite_parts.append(selected)
        elevation_parts.append(elevation[user, selected])
        distance_parts.append(distance_matrix[user, selected])
    if not user_parts:
        empty_ids = torch.empty((0, 2), dtype=torch.long, device=user_pos.device)
        return ElevationCandidates(
            edge_index=torch.empty((2, 0), dtype=torch.long, device=user_pos.device),
            edge_ids=empty_ids,
            elevation_deg=torch.empty(0, dtype=user_pos.dtype, device=user_pos.device),
            distance=torch.empty(0, dtype=user_pos.dtype, device=user_pos.device),
        )
    local_users = torch.cat(user_parts)
    local_satellites = torch.cat(satellite_parts)
    edge_ids = torch.stack((local_users, local_satellites), dim=1)
    edge_index = torch.stack(
        (local_users + int(user_offset), local_satellites + satellite_offset), dim=0
    )
    return ElevationCandidates(
        edge_index=edge_index.long(),
        edge_ids=edge_ids.long(),
        elevation_deg=torch.cat(elevation_parts),
        distance=torch.cat(distance_parts),
    )
