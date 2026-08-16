from __future__ import annotations

import hashlib

import torch


def keyed_seed(episode_seed: int, epoch: int, stream: str) -> int:
    """Return a stable 63-bit seed for a named exogenous random stream."""
    if episode_seed < 0 or epoch < 0:
        raise ValueError("episode_seed and epoch must be non-negative")
    if not stream:
        raise ValueError("stream must be non-empty")
    payload = f"leo-pg-v1:{int(episode_seed)}:{int(epoch)}:{stream}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def keyed_generator(episode_seed: int, epoch: int, stream: str) -> torch.Generator:
    """Create an independent CPU generator without touching global RNG state."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(keyed_seed(episode_seed, epoch, stream))
    return generator


def keyed_user_order(
    user_count: int,
    episode_seed: int,
    epoch: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if user_count <= 0:
        raise ValueError("user_count must be positive")
    order = torch.randperm(
        user_count,
        generator=keyed_generator(episode_seed, epoch, "user-order"),
        device="cpu",
    )
    return order.to(device=device)


def keyed_standard_normal(
    shape: tuple[int, ...],
    episode_seed: int,
    epoch: int,
    stream: str,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if any(size < 0 for size in shape):
        raise ValueError("shape dimensions must be non-negative")
    values = torch.randn(
        shape,
        generator=keyed_generator(episode_seed, epoch, stream),
        dtype=dtype,
        device="cpu",
    )
    return values.to(device=device)
