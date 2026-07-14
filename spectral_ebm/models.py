"""Scalar energy networks used by the POC and its matched baseline."""

from __future__ import annotations

from torch import Tensor, nn

from .layers import CirculantLinear


def _activation() -> nn.Module:
    return nn.SiLU()


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


def parameter_count(module: nn.Module) -> int:
    """Return the number of trainable and frozen parameters in ``module``."""

    return sum(parameter.numel() for parameter in module.parameters())
