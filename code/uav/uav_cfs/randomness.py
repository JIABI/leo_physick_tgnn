from __future__ import annotations

import hashlib
from typing import Iterable, Sequence


_KEY_NAMESPACE = "leo-pg-uav-shared-v1"


def keyed_u64(
    episode_seed: int,
    epoch: int,
    stream: str,
    *entity_keys: int | str,
) -> int:
    """Return an order-independent deterministic random key.

    Random draws are addressed by semantic keys rather than consumed from a
    mutable generator.  Forked rollouts therefore receive the same exogenous
    draws for the same ``(seed, epoch, stream, entity_keys)`` tuple even when
    users are visited in a different Python iteration order.
    """

    if isinstance(episode_seed, bool) or int(episode_seed) != episode_seed:
        raise TypeError("episode_seed must be an integer")
    if isinstance(epoch, bool) or int(epoch) != epoch:
        raise TypeError("epoch must be an integer")
    if episode_seed < 0 or epoch < 0:
        raise ValueError("episode_seed and epoch must be non-negative")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a non-empty string")
    payload = ":".join(
        (_KEY_NAMESPACE, str(int(episode_seed)), str(int(epoch)), stream)
        + tuple(str(key) for key in entity_keys)
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        byteorder="little",
        signed=False,
    )


def keyed_uniform(
    episode_seed: int,
    epoch: int,
    stream: str,
    *entity_keys: int | str,
) -> float:
    """Map a semantic key to a reproducible value in the open interval (0, 1)."""

    value = keyed_u64(episode_seed, epoch, stream, *entity_keys)
    return (value + 0.5) / float(1 << 64)


def keyed_signed_uniform(
    episode_seed: int,
    epoch: int,
    stream: str,
    *entity_keys: int | str,
) -> float:
    return 2.0 * keyed_uniform(
        episode_seed, epoch, stream, *entity_keys
    ) - 1.0


def keyed_order(
    values: Iterable[int],
    episode_seed: int,
    epoch: int,
    stream: str,
) -> tuple[int, ...]:
    """Return a deterministic permutation whose result ignores input order."""

    canonical = sorted(int(value) for value in values)
    if len(canonical) != len(set(canonical)):
        raise ValueError("values passed to keyed_order must be unique")
    return tuple(
        sorted(
            canonical,
            key=lambda value: (
                keyed_u64(episode_seed, epoch, stream, value),
                value,
            ),
        )
    )


def keyed_choice_index(
    size: int,
    episode_seed: int,
    epoch: int,
    stream: str,
    keys: Sequence[int | str] = (),
) -> int:
    if isinstance(size, bool) or int(size) != size or size <= 0:
        raise ValueError("size must be a positive integer")
    return keyed_u64(episode_seed, epoch, stream, *keys) % int(size)
