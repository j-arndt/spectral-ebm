"""Langevin sampling and deterministic energy relaxation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def _energy_gradient(model: nn.Module, x: Tensor) -> Tensor:
    x = x.detach().requires_grad_(True)
    energy = model(x)
    if energy.ndim != 1:
        energy = energy.reshape(-1)
    gradient = torch.autograd.grad(energy.sum(), x, create_graph=False)[0]
    return gradient


def _apply_bounds(x: Tensor, bounds: tuple[float, float] | None) -> Tensor:
    if bounds is None:
        return x
    low, high = bounds
    if not low < high:
        raise ValueError("bounds must satisfy low < high")
    return x.clamp(min=low, max=high)


def ula_step(
    model: nn.Module,
    x: Tensor,
    *,
    step_size: float,
    temperature: float = 1.0,
    noise: Tensor | None = None,
    bounds: tuple[float, float] | None = None,
    noise_scale: float = 1.0,
) -> Tensor:
    """Perform one ULA step for ``p(x) ∝ exp(-E(x) / temperature)``.

    With time step ``h`` and standard normal ``epsilon``, the update is

    ``x' = x - h/(2T) * grad(E(x)) + sqrt(h) * epsilon``.

    ``bounds`` applies a documented projected-ULA variant and therefore is not
    claimed to preserve the unconstrained target exactly.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")
    gradient = _energy_gradient(model, x)
    if noise is None:
        noise = torch.randn_like(x)
    if noise.shape != x.shape:
        raise ValueError("noise must have the same shape as x")
    updated = x.detach() - (step_size / (2.0 * temperature)) * gradient
    updated = updated + noise_scale * step_size**0.5 * noise
    return _apply_bounds(updated, bounds).detach()


def langevin_sample(
    model: nn.Module,
    x_init: Tensor,
    *,
    steps: int,
    step_size: float,
    temperature: float = 1.0,
    bounds: tuple[float, float] | None = None,
    noise_scale: float = 1.0,
    callback: Callable[[int, Tensor], None] | None = None,
) -> Tensor:
    """Run a short ULA chain, returning a detached final state."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    state = x_init.detach()
    for index in range(steps):
        state = ula_step(
            model,
            state,
            step_size=step_size,
            temperature=temperature,
            bounds=bounds,
            noise_scale=noise_scale,
        )
        if callback is not None:
            callback(index, state)
    return state


def relax(
    model: nn.Module,
    x_init: Tensor,
    *,
    steps: int,
    step_size: float,
    temperature: float = 1.0,
    bounds: tuple[float, float] | None = None,
) -> Tensor:
    """Run deterministic gradient descent on energy, not an MCMC sampler."""

    return langevin_sample(
        model,
        x_init,
        steps=steps,
        step_size=2.0 * step_size,
        temperature=temperature,
        bounds=bounds,
        noise_scale=0.0,
    )
