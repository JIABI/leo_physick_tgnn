"""Temporal backbones used by the controlled paper baselines.

All backbones in this module implement the same streaming contract::

    embedding_t, state = backbone.forward_step(token_t, state)

``token_t`` has shape ``[num_nodes, d_model]`` and ``state`` is either ``None``
or a tensor containing the episode-local history.  Node identities and node
count must remain fixed within an episode; callers must pass ``state=None`` at
an episode boundary.  The returned embedding has shape
``[num_nodes, d_model]``.  The history length is bounded by
``context_length`` so evaluation memory cannot grow with rollout duration.

These classes implement temporal mixing only.  The paper S4/Mamba2/Conformer
methods are assembled by ``PhysiCKTemporalMixerWorldModel``, which supplies the
shared 128-dimensional Paper PhysiCK message stream and Intensity--Flow
readout.  ``build_temporal_backbone`` must not be treated as a complete graph
world-model factory.

Dependency assumptions
----------------------
``TransformerTemporalBackbone`` uses only the ``torch`` distribution.

``OfficialS4TemporalBackbone`` uses the upstream `state-spaces/s4` repository
(https://github.com/state-spaces/s4), whose canonical Python import is
``models.s4.s4.S4Block``.  Upstream S4 does not publish a stable PyPI package
with that import path; install the official repository and place its root on
``PYTHONPATH``.  No local approximation is substituted when it is unavailable.

``Mamba2TemporalBackbone`` uses distribution ``mamba-ssm`` and imports
``mamba_ssm.Mamba2``.  The installed wheel/source build must match the active
PyTorch/CUDA toolchain.

``TorchaudioConformerTemporalBackbone`` uses distribution ``torchaudio`` and
imports ``torchaudio.models.Conformer``.  Its version/build must match the
installed ``torch`` distribution.

The optional imports are deliberately lazy.  A repository user can use the
Transformer or GAT baselines without installing every long-context backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


TemporalState = torch.Tensor | None


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _dropout(value: Any) -> float:
    result = float(value)
    if not 0.0 <= result < 1.0:
        raise ValueError("dropout must lie in [0, 1)")
    return result


class StreamingTemporalBackbone(nn.Module, ABC):
    """Abstract fixed-width temporal backbone with bounded raw-token history."""

    def __init__(self, *, d_model: int, context_length: int) -> None:
        super().__init__()
        self.d_model = _positive_int("d_model", d_model)
        self.context_length = _positive_int("context_length", context_length)

    def _append_history(
        self,
        token_t: torch.Tensor,
        state: TemporalState,
    ) -> torch.Tensor:
        if not isinstance(token_t, torch.Tensor):
            raise TypeError("token_t must be a torch.Tensor")
        if token_t.ndim != 2 or token_t.size(1) != self.d_model:
            raise ValueError(
                f"token_t must have shape [num_nodes,{self.d_model}], "
                f"got {tuple(token_t.shape)}"
            )
        if token_t.size(0) == 0:
            raise ValueError("token_t requires at least one node")
        if not token_t.is_floating_point() or not bool(torch.isfinite(token_t).all()):
            raise ValueError("token_t must be a finite floating-point tensor")

        if state is None:
            history = token_t.unsqueeze(1)
        else:
            if not isinstance(state, torch.Tensor):
                raise TypeError("temporal state must be a torch.Tensor or None")
            if state.ndim != 3:
                raise ValueError(
                    "temporal state must have shape "
                    f"[num_nodes,history,{self.d_model}] with stable node identity"
                )
            expected = (token_t.size(0), state.size(1), self.d_model)
            if tuple(state.shape) != expected:
                raise ValueError(
                    "temporal state must have shape "
                    f"[num_nodes,history,{self.d_model}] with stable node identity"
                )
            if state.device != token_t.device or state.dtype != token_t.dtype:
                raise ValueError("temporal state and token_t must share device and dtype")
            if state.size(1) == 0 or state.size(1) > self.context_length:
                raise ValueError("temporal state has an invalid history length")
            if not bool(torch.isfinite(state).all()):
                raise ValueError("temporal state contains NaN or Inf")
            history = torch.cat((state, token_t.unsqueeze(1)), dim=1)

        if history.size(1) > self.context_length:
            history = history[:, -self.context_length :, :]
        return history

    @abstractmethod
    def forward_step(
        self,
        token_t: torch.Tensor,
        state: TemporalState = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current node embedding and bounded raw-token history."""


class TransformerTemporalBackbone(StreamingTemporalBackbone):
    """Causal PyTorch Transformer encoder evaluated on a rolling context.

    The LTT-R default factory uses ``d_model=384``, six layers, eight heads and
    ``ffn_dim=1536``.  Together with the controlled graph front-end and
    Intensity--Flow readout this is intended to be the paper's approximately
    12-million-parameter capacity class; the exact parameter count depends on
    node/edge feature widths and must be recorded by the experiment runner.
    """

    def __init__(
        self,
        *,
        d_model: int,
        context_length: int = 200,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_model=d_model, context_length=context_length)
        self.num_layers = _positive_int("num_layers", num_layers)
        self.num_heads = _positive_int("num_heads", num_heads)
        self.ffn_dim = _positive_int("ffn_dim", ffn_dim)
        self.dropout = _dropout(dropout)
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model),
        )
        self.position = nn.Parameter(torch.empty(self.context_length, self.d_model))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward_step(
        self,
        token_t: torch.Tensor,
        state: TemporalState = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = self._append_history(token_t, state)
        length = int(history.size(1))
        sequence = history + self.position[:length].to(
            device=history.device,
            dtype=history.dtype,
        ).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=history.device),
            diagonal=1,
        )
        encoded = self.encoder(sequence, mask=causal_mask)
        return encoded[:, -1, :], history


class OfficialS4TemporalBackbone(StreamingTemporalBackbone):
    """Adapter around the official ``state-spaces/s4`` ``S4Block``.

    The adapter recomputes the bounded rolling context rather than depending on
    an undocumented recurrent-state layout.  ``s4_kwargs`` is forwarded to the
    upstream ``S4Block`` constructor; kernel options therefore remain explicit
    in the run configuration.
    """

    def __init__(
        self,
        *,
        d_model: int,
        context_length: int = 200,
        num_layers: int = 4,
        dropout: float = 0.1,
        s4_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(d_model=d_model, context_length=context_length)
        self.num_layers = _positive_int("num_layers", num_layers)
        self.dropout = _dropout(dropout)
        try:
            from models.s4.s4 import S4Block
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Official S4 requested but `models.s4.s4.S4Block` is unavailable. "
                "Install the upstream `state-spaces/s4` repository from "
                "https://github.com/state-spaces/s4 and add its repository root "
                "to PYTHONPATH; no fallback S4 approximation is used."
            ) from exc

        kwargs = dict(s4_kwargs or {})
        forbidden = {"d_model", "dropout", "transposed"}.intersection(kwargs)
        if forbidden:
            raise ValueError(
                "s4_kwargs must not override controlled arguments: "
                + ", ".join(sorted(forbidden))
            )
        self.blocks = nn.ModuleList(
            [
                S4Block(
                    d_model=self.d_model,
                    dropout=self.dropout,
                    transposed=True,
                    **kwargs,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(self.num_layers)]
        )
        self.residual_dropout = nn.Dropout(self.dropout)
        self.output_norm = nn.LayerNorm(self.d_model)

    def forward_step(
        self,
        token_t: torch.Tensor,
        state: TemporalState = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = self._append_history(token_t, state)
        sequence = history
        for norm, block in zip(self.norms, self.blocks):
            upstream_result = block(norm(sequence).transpose(1, 2))
            transformed = (
                upstream_result[0]
                if isinstance(upstream_result, tuple)
                else upstream_result
            )
            if transformed.ndim != 3:
                raise RuntimeError("official S4Block returned a non-sequence tensor")
            transformed = transformed.transpose(1, 2)
            if transformed.shape != sequence.shape:
                raise RuntimeError(
                    "official S4Block output shape is incompatible with the configured "
                    "d_model/context"
                )
            sequence = sequence + self.residual_dropout(transformed)
        encoded = self.output_norm(sequence)
        return encoded[:, -1, :], history


class Mamba2TemporalBackbone(StreamingTemporalBackbone):
    """Adapter around ``mamba_ssm.Mamba2`` from distribution ``mamba-ssm``."""

    def __init__(
        self,
        *,
        d_model: int,
        context_length: int = 200,
        num_layers: int = 4,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_model=d_model, context_length=context_length)
        self.num_layers = _positive_int("num_layers", num_layers)
        self.d_state = _positive_int("d_state", d_state)
        self.d_conv = _positive_int("d_conv", d_conv)
        self.expand = _positive_int("expand", expand)
        self.headdim = _positive_int("headdim", headdim)
        self.dropout = _dropout(dropout)
        if (self.d_model * self.expand) % self.headdim != 0:
            raise ValueError("d_model * expand must be divisible by headdim for Mamba2")
        try:
            from mamba_ssm import Mamba2
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Mamba2 requested but `mamba_ssm.Mamba2` is unavailable. Install "
                "the `mamba-ssm` distribution built for the active PyTorch/CUDA "
                "toolchain; no Transformer or local approximation is substituted."
            ) from exc

        self.blocks = nn.ModuleList(
            [
                Mamba2(
                    d_model=self.d_model,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    headdim=self.headdim,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(self.num_layers)]
        )
        self.residual_dropout = nn.Dropout(self.dropout)
        self.output_norm = nn.LayerNorm(self.d_model)

    def forward_step(
        self,
        token_t: torch.Tensor,
        state: TemporalState = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = self._append_history(token_t, state)
        sequence = history
        for norm, block in zip(self.norms, self.blocks):
            transformed = block(norm(sequence))
            if transformed.shape != sequence.shape:
                raise RuntimeError("Mamba2 output shape is incompatible with d_model")
            sequence = sequence + self.residual_dropout(transformed)
        encoded = self.output_norm(sequence)
        return encoded[:, -1, :], history


class TorchaudioConformerTemporalBackbone(StreamingTemporalBackbone):
    """Adapter around ``torchaudio.models.Conformer``.

    A complete rolling context is passed to the official Conformer module on
    each decision epoch.  This keeps the public state contract independent of
    torchaudio-private cache formats.
    """

    def __init__(
        self,
        *,
        d_model: int,
        context_length: int = 200,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_model=d_model, context_length=context_length)
        self.num_layers = _positive_int("num_layers", num_layers)
        self.num_heads = _positive_int("num_heads", num_heads)
        self.ffn_dim = _positive_int("ffn_dim", ffn_dim)
        self.depthwise_conv_kernel_size = _positive_int(
            "depthwise_conv_kernel_size", depthwise_conv_kernel_size
        )
        self.dropout = _dropout(dropout)
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.depthwise_conv_kernel_size % 2 == 0:
            raise ValueError("Conformer depthwise_conv_kernel_size must be odd")
        try:
            from torchaudio.models import Conformer
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            raise ImportError(
                "Conformer requested but `torchaudio.models.Conformer` is unavailable. "
                "Install the `torchaudio` distribution whose version and CPU/CUDA "
                "build match the active `torch` installation; no local Conformer "
                "approximation is used."
            ) from exc

        self.conformer = Conformer(
            input_dim=self.d_model,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
            depthwise_conv_kernel_size=self.depthwise_conv_kernel_size,
            dropout=self.dropout,
        )

    def forward_step(
        self,
        token_t: torch.Tensor,
        state: TemporalState = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = self._append_history(token_t, state)
        lengths = torch.full(
            (history.size(0),),
            history.size(1),
            dtype=torch.long,
            device=history.device,
        )
        encoded, output_lengths = self.conformer(history, lengths)
        if encoded.shape != history.shape or not torch.equal(output_lengths, lengths):
            raise RuntimeError("torchaudio Conformer changed the controlled sequence shape")
        return encoded[:, -1, :], history


def build_temporal_backbone(
    config: Mapping[str, Any],
    *,
    d_model: int,
) -> StreamingTemporalBackbone:
    """Build one real temporal backend from an explicit configuration mapping.

    Accepted ``type`` values are ``transformer``, ``s4``, ``mamba2`` and
    ``conformer``.  Optional backends never fall back to Transformer when their
    dependency is missing.
    """

    if not isinstance(config, Mapping):
        raise TypeError("temporal backbone config must be a mapping")
    kind = str(config.get("type", "")).strip().lower()
    if not kind:
        raise ValueError("temporal backbone config requires an explicit `type`")
    context_length = config.get("context_length", 200)
    dropout = config.get("dropout", 0.1)

    if kind == "transformer":
        return TransformerTemporalBackbone(
            d_model=d_model,
            context_length=context_length,
            num_layers=config.get("num_layers", 6),
            num_heads=config.get("num_heads", 8),
            ffn_dim=config.get("ffn_dim", 4 * int(d_model)),
            dropout=dropout,
        )
    if kind == "s4":
        return OfficialS4TemporalBackbone(
            d_model=d_model,
            context_length=context_length,
            num_layers=config.get("num_layers", 4),
            dropout=dropout,
            s4_kwargs=config.get("s4_kwargs"),
        )
    if kind == "mamba2":
        return Mamba2TemporalBackbone(
            d_model=d_model,
            context_length=context_length,
            num_layers=config.get("num_layers", 4),
            d_state=config.get("d_state", 128),
            d_conv=config.get("d_conv", 4),
            expand=config.get("expand", 2),
            headdim=config.get("headdim", 64),
            dropout=dropout,
        )
    if kind == "conformer":
        return TorchaudioConformerTemporalBackbone(
            d_model=d_model,
            context_length=context_length,
            num_layers=config.get("num_layers", 4),
            num_heads=config.get("num_heads", 8),
            ffn_dim=config.get("ffn_dim", 4 * int(d_model)),
            depthwise_conv_kernel_size=config.get(
                "depthwise_conv_kernel_size", 31
            ),
            dropout=dropout,
        )
    raise ValueError(
        f"unknown temporal backbone {kind!r}; expected transformer, s4, mamba2, "
        "or conformer"
    )
