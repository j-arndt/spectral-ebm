"""Training losses with explicit conventions."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def contrastive_divergence_loss(
    model: nn.Module,
    positive: Tensor,
    negative: Tensor,
) -> Tensor:
    """Short-run CD loss with a detached negative chain.

    Minimizing ``E(positive) - E(negative)`` lowers data energy and raises
    negative-sample energy. The negative chain is intentionally detached; this
    is the conventional short-run CD approximation, not exact maximum
    likelihood through the sampler.
    """

    return model(positive).mean() - model(negative.detach()).mean()


def denoising_score_matching_loss(
    model: nn.Module,
    clean: Tensor,
    *,
    noise_std: float,
    temperature: float = 1.0,
) -> Tensor:
    """Denoising score matching loss for an energy whose score is ``-grad(E)/T``."""

    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    noise = torch.randn_like(clean) * noise_std
    noisy = (clean + noise).detach().requires_grad_(True)
    energy = model(noisy).sum()
    grad_energy = torch.autograd.grad(energy, noisy, create_graph=True)[0]
    predicted_score = -grad_energy / temperature
    target_score = -noise / noise_std**2
    return (predicted_score - target_score).square().flatten(start_dim=1).mean()
