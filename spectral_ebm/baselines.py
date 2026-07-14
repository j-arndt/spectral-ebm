"""Additional low-parameter structured baseline for controlled comparisons."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DiagonalLinear(nn.Module):
    """Elementwise affine map with ``O(D)`` parameters."""

    def __init__(self, dim: int, *, bias: bool = True) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        result = x * self.scale
        return result if self.bias is None else result + self.bias


class DiagonalEBM(nn.Module):
    """Scalar energy network using diagonal structured maps."""

    def __init__(self, dim: int, hidden_layers: int = 3, *, bias: bool = True) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        blocks: list[nn.Module] = []
        for index in range(hidden_layers):
            blocks.append(DiagonalLinear(dim, bias=bias))
            if index + 1 < hidden_layers:
                blocks.append(nn.SiLU())
        self.network = nn.Sequential(*blocks)
        self.energy_projection = nn.Linear(dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.energy_projection(self.network(x)).squeeze(-1)
