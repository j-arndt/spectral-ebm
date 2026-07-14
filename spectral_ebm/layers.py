"""FFT-parameterized linear layers with an explicit circulant convention."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def circulant_matrix(generator: Tensor) -> Tensor:
    """Materialize the circulant matrix whose first column is ``generator``.

    For ``c = generator`` and ``W = circulant_matrix(c)``,

    ``W[i, j] = c[(i - j) mod D]``.

    Consequently, ``W @ x`` is the circular convolution of ``c`` and ``x``
    and is exactly computed by ``irfft(rfft(x) * rfft(c))`` under PyTorch's
    default FFT normalization.
    """

    if generator.ndim != 1:
        raise ValueError("generator must be a one-dimensional tensor")
    return torch.stack(
        [torch.roll(generator, shifts=j, dims=0) for j in range(generator.numel())], dim=1
    )


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
