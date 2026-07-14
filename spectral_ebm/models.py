"""Scalar energy networks and structured architectural variants."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from .layers import BlockCirculantLinear, CirculantLinear
from .permutations import DifferentiablePermutation


def _activation() -> nn.Module:
    return nn.SiLU()


def _noncyclic_permutation(dim: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(dim, generator=generator)
    identity = torch.arange(dim)
    if any(torch.equal(permutation, torch.roll(identity, shift)) for shift in range(dim)):
        permutation = permutation.clone()
        permutation[0], permutation[1] = permutation[1], permutation[0]
    return permutation


class SpectralEBM(nn.Module):
    """Scalar energy network built from FFT-parameterized layers."""

    def __init__(self, dim: int, hidden_layers: int = 3, *, bias: bool = True) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        blocks: list[nn.Module] = []
        for index in range(hidden_layers):
            blocks.append(CirculantLinear(dim, bias=bias))
            if index + 1 < hidden_layers:
                blocks.append(_activation())
        self.network = nn.Sequential(*blocks)
        self.energy_projection = nn.Linear(dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.energy_projection(self.network(x)).squeeze(-1)


class DenseEBM(nn.Module):
    """Matched dense scalar energy baseline with the same depth and activation."""

    def __init__(self, dim: int, hidden_layers: int = 3, *, bias: bool = True) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        blocks: list[nn.Module] = []
        for index in range(hidden_layers):
            blocks.append(nn.Linear(dim, dim, bias=bias))
            if index + 1 < hidden_layers:
                blocks.append(_activation())
        self.network = nn.Sequential(*blocks)
        self.energy_projection = nn.Linear(dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.energy_projection(self.network(x)).squeeze(-1)


class BlockSpectralEBM(nn.Module):
    """Multi-channel block-circulant scalar energy network.

    Inputs have shape ``(..., channels, dim)``. Every hidden layer mixes
    channels densely while using circulant feature-axis blocks. The output is
    a scalar energy per leading batch element.
    """

    def __init__(
        self,
        channels: int,
        dim: int,
        hidden_layers: int = 3,
        *,
        hidden_channels: int | None = None,
        bias: bool = True,
        backend: Literal["torch", "triton"] = "torch",
    ) -> None:
        super().__init__()
        if channels < 1 or dim < 1:
            raise ValueError("channels and dim must be positive")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        hidden_channels = channels if hidden_channels is None else int(hidden_channels)
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        self.channels = int(channels)
        self.dim = int(dim)
        self.hidden_channels = hidden_channels
        channel_sizes = [self.channels] + [hidden_channels] * hidden_layers
        self.layers = nn.ModuleList(
            [
                BlockCirculantLinear(
                    channel_sizes[index],
                    channel_sizes[index + 1],
                    dim,
                    bias=bias,
                    backend=backend,
                )
                for index in range(hidden_layers)
            ]
        )
        self.activation = _activation()
        self.energy_projection = nn.Linear(hidden_channels * dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 3 or x.shape[-2:] != (self.channels, self.dim):
            raise ValueError(
                f"expected trailing shape ({self.channels}, {self.dim}), "
                f"got {tuple(x.shape[-2:]) if x.ndim >= 2 else tuple(x.shape)}"
            )
        hidden = x
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index + 1 < len(self.layers):
                hidden = self.activation(hidden)
        return self.energy_projection(hidden.flatten(start_dim=-2)).squeeze(-1)


class PermutedSpectralEBM(nn.Module):
    """Spectral EBM with fixed or differentiable coordinate permutations.

    ``permutation_mode="fixed"`` uses deterministic non-cyclic buffers and
    adds no trainable parameters. ``permutation_mode="differentiable"`` uses
    Sinkhorn-normalized learnable matrices, optionally with straight-through
    hard assignments. Both modes break the shared coordinate symmetry between
    spectral layers; only the fixed mode is exactly reproducible without a
    training seed.
    """

    def __init__(
        self,
        dim: int,
        hidden_layers: int = 3,
        *,
        bias: bool = True,
        seed: int = 42,
        permutation_mode: Literal["fixed", "differentiable"] = "fixed",
        permutation_temperature: float = 0.5,
        permutation_hard: bool = False,
        permutation_noise_scale: float = 0.0,
        permutation_sinkhorn_iterations: int = 20,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        if permutation_mode not in ("fixed", "differentiable"):
            raise ValueError("permutation_mode must be fixed or differentiable")
        if permutation_mode == "fixed" and hidden_layers > 1 and dim < 3:
            raise ValueError("dim must be at least 3 for non-cyclic fixed permutations")
        self.dim = int(dim)
        self.hidden_layers = int(hidden_layers)
        self.seed = int(seed)
        self.permutation_mode = permutation_mode
        self.layers = nn.ModuleList(
            [CirculantLinear(dim, bias=bias) for _ in range(hidden_layers)]
        )
        self._permutation_names: list[str] = []
        self.permutation_layers = nn.ModuleList()
        for index in range(hidden_layers - 1):
            if permutation_mode == "fixed":
                name = f"permutation_{index}"
                self.register_buffer(
                    name,
                    _noncyclic_permutation(dim, self.seed + index),
                    persistent=True,
                )
                self._permutation_names.append(name)
            else:
                self.permutation_layers.append(
                    DifferentiablePermutation(
                        dim,
                        temperature=permutation_temperature,
                        sinkhorn_iterations=permutation_sinkhorn_iterations,
                        hard=permutation_hard,
                        noise_scale=permutation_noise_scale,
                    )
                )
        self.activation = _activation()
        self.energy_projection = nn.Linear(dim, 1, bias=False)

    @property
    def permutations(self) -> tuple[Tensor, ...]:
        """Return current fixed buffers or differentiable matrices."""

        if self.permutation_mode == "differentiable":
            return tuple(layer.permutation_matrix() for layer in self.permutation_layers)
        return tuple(getattr(self, name) for name in self._permutation_names)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        hidden = x
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index + 1 < len(self.layers):
                hidden = self.activation(hidden)
                if self.permutation_mode == "differentiable":
                    hidden = self.permutation_layers[index](hidden)
                else:
                    hidden = hidden.index_select(-1, self.permutations[index])
        return self.energy_projection(hidden).squeeze(-1)


def parameter_count(module: nn.Module) -> int:
    """Return the number of trainable and frozen parameters in ``module``."""

    return sum(parameter.numel() for parameter in module.parameters())
