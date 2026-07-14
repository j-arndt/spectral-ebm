"""Differentiable coordinate permutations for structured spectral networks."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DifferentiablePermutation(nn.Module):
    """Learn a soft permutation relaxation or straight-through hard row assignment.

    The learnable logits produce a doubly-stochastic matrix ``P``. Inputs with
    shape ``(..., D)`` are transformed as ``x @ P.T``. With ``hard=True``, the
    forward value uses one-hot row assignments while the backward pass uses the
    soft Sinkhorn matrix. The soft matrix is doubly stochastic; row-wise hard
    projection is intentionally lightweight and is not an exact assignment solver.
    This layer is differentiable and adds ``D^2`` trainable logits.
    """

    def __init__(
        self,
        dim: int,
        *,
        temperature: float = 0.5,
        sinkhorn_iterations: int = 20,
        hard: bool = False,
        noise_scale: float = 0.0,
        identity_bias: float = 2.0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be positive")
        if noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")
        self.dim = int(dim)
        self.temperature = float(temperature)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.hard = bool(hard)
        self.noise_scale = float(noise_scale)
        initial = torch.eye(self.dim) * float(identity_bias)
        self.logits = nn.Parameter(initial)

    def permutation_matrix(self) -> Tensor:
        """Return the current differentiable doubly-stochastic matrix."""

        logits = self.logits
        if self.noise_scale:
            uniform = torch.rand_like(logits).clamp_(min=1e-6, max=1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            logits = logits + self.noise_scale * gumbel
        log_probability = logits / self.temperature
        for _ in range(self.sinkhorn_iterations):
            log_probability = log_probability - torch.logsumexp(
                log_probability, dim=-1, keepdim=True
            )
            log_probability = log_probability - torch.logsumexp(
                log_probability, dim=-2, keepdim=True
            )
        probability = log_probability.exp()
        if self.hard:
            indices = probability.argmax(dim=-1, keepdim=True)
            hard_probability = torch.zeros_like(probability).scatter_(-1, indices, 1.0)
            probability = hard_probability - probability.detach() + probability
        return probability

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        return torch.matmul(x, self.permutation_matrix().transpose(-1, -2))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, temperature={self.temperature}, "
            f"iterations={self.sinkhorn_iterations}, hard={self.hard}"
        )


class AmortizedHouseholderPermutation(nn.Module):
    """Compose K trainable Householder reflections with O(KD) parameters.

    The result is an orthogonal coordinate transformation, preserving the
    Euclidean norm up to floating-point error. It is a low-rank orthogonal
    mixer rather than a strict one-hot coordinate permutation.
    """

    def __init__(self, dim: int, reflections: int = 4, *, eps: float = 1e-8) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if reflections < 0:
            raise ValueError("reflections must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.dim = int(dim)
        self.reflections = int(reflections)
        self.eps = float(eps)
        self.vectors = nn.Parameter(torch.empty(self.reflections, self.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.reflections:
            nn.init.normal_(self.vectors, mean=0.0, std=1.0)

    def _unit_vectors(self) -> Tensor:
        return F.normalize(self.vectors, dim=-1, eps=self.eps)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        output = x
        for vector in self._unit_vectors():
            projection = torch.matmul(output, vector)
            output = output - 2.0 * projection.unsqueeze(-1) * vector
        return output

    def matrix(self) -> Tensor:
        """Materialize the dense orthogonal operator for small diagnostics only."""

        operator = torch.eye(self.dim, device=self.vectors.device, dtype=self.vectors.dtype)
        for vector in self._unit_vectors():
            reflector = torch.eye(self.dim, device=operator.device, dtype=operator.dtype) - 2.0 * torch.outer(vector, vector)
            operator = operator @ reflector
        return operator

    def extra_repr(self) -> str:
        return f"dim={self.dim}, reflections={self.reflections}, parameters={self.reflections * self.dim}"
