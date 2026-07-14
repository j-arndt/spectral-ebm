"""FFT-parameterized linear layers with explicit circulant conventions."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


def circulant_matrix(generator: Tensor) -> Tensor:
    """Materialize the circulant matrix whose first column is ``generator``.

    For ``c = generator`` and ``W = circulant_matrix(c)``,
    ``W[i, j] = c[(i - j) mod D]``. Consequently, ``W @ x`` is circular
    convolution and is computed by ``irfft(rfft(x) * rfft(c))``.
    """

    if generator.ndim != 1:
        raise ValueError("generator must be a one-dimensional tensor")
    return torch.stack(
        [torch.roll(generator, shifts=j, dims=0) for j in range(generator.numel())], dim=1
    )


def block_circulant_matrix(weight: Tensor) -> Tensor:
    """Materialize a block-circulant operator from ``[C_out, C_in, D]`` weights.

    Each channel block uses the same first-column convention as
    :func:`circulant_matrix`. The returned matrix has shape
    ``[C_out * D, C_in * D]`` and acts on tensors flattened from ``[C_in, D]``.
    This intentionally slow construction is a reference oracle for tests and
    small-scale diagnostics, not a production execution path.
    """

    if weight.ndim != 3:
        raise ValueError("weight must have shape [out_channels, in_channels, dim]")
    blocks = [
        [circulant_matrix(weight[out_channel, in_channel]) for in_channel in range(weight.shape[1])]
        for out_channel in range(weight.shape[0])
    ]
    return torch.cat([torch.cat(row, dim=1) for row in blocks], dim=0)


class CirculantLinear(nn.Module):
    """A real square circulant linear map implemented with an FFT.

    The layer has ``D`` generator parameters and, when enabled, ``D`` bias
    parameters. It is not an arbitrary dense ``D x D`` map: it is equivariant
    to cyclic shifts of the feature index.
    """

    def __init__(self, dim: int, *, bias: bool = True) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.generator = nn.Parameter(torch.empty(self.dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.generator, mean=0.0, std=self.dim**-0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        x_fft = torch.fft.rfft(x, dim=-1)
        c_fft = torch.fft.rfft(self.generator, dim=0)
        y = torch.fft.irfft(x_fft * c_fft, n=self.dim, dim=-1)
        return y if self.bias is None else y + self.bias

    @property
    def eigenvalues(self) -> Tensor:
        """Eigenvalues of the linear map under the documented convention."""

        return torch.fft.fft(self.generator, dim=0)

    def spectral_norm(self) -> Tensor:
        """Exact operator 2-norm of the real circulant map."""

        return self.eigenvalues.abs().amax()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, bias={self.bias is not None}"


class BlockCirculantLinear(nn.Module):
    """Multi-channel block-circulant linear map implemented in the FFT domain.

    Input tensors have shape ``(..., C_in, D)`` and outputs have shape
    ``(..., C_out, D)``. Each ``(out_channel, in_channel)`` pair owns one
    length-``D`` generator, so the layer has ``C_out * C_in * D`` weight
    parameters plus ``C_out * D`` bias parameters when enabled. The channel
    mixing is dense while the feature-axis map in every block is circulant.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        *,
        bias: bool = True,
        backend: Literal["torch", "triton"] = "torch",

    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1 or dim < 1:
            raise ValueError("in_channels, out_channels, and dim must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dim = int(dim)
        if backend not in ("torch", "triton"):
            raise ValueError("backend must be torch or triton")
        self.backend = backend
        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels, self.dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, self.dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = (self.dim * self.in_channels) ** -0.5
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError("input must have shape (..., in_channels, dim)")
        if x.shape[-2:] != (self.in_channels, self.dim):
            raise ValueError(
                f"expected trailing shape ({self.in_channels}, {self.dim}), "
                f"got {tuple(x.shape[-2:])}"
            )
        x_fft = torch.fft.rfft(x, dim=-1)
        weight_fft = torch.fft.rfft(self.weight, dim=-1)
        if self.backend == "triton":
            from .triton_backend import block_circulant_frequency_mix
            output_fft = block_circulant_frequency_mix(x_fft, weight_fft)
        else:
            output_fft = torch.einsum("...if,oif->...of", x_fft, weight_fft)
        output = torch.fft.irfft(output_fft, n=self.dim, dim=-1)
        return output if self.bias is None else output + self.bias

    def materialize(self) -> Tensor:
        """Return the slow dense reference matrix for small diagnostics."""

        return block_circulant_matrix(self.weight)

    def spectral_norm(self) -> Tensor:
        """Exact operator 2-norm from the largest Fourier-symbol singular value."""

        symbols = torch.fft.fft(self.weight, dim=-1).permute(2, 0, 1)
        return torch.linalg.matrix_norm(symbols, ord=2, dim=(-2, -1)).amax()

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dim={self.dim}, bias={self.bias is not None}, backend={self.backend}"
        )
