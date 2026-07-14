"""Langevin sampling and deterministic energy relaxation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def _energy_gradient(model: nn.Module, x: Tensor) -> Tensor:
    variable = x.detach().requires_grad_(True)
    energy = model(variable)
    if energy.ndim != 1:
        energy = energy.reshape(-1)
    return torch.autograd.grad(energy.sum(), variable, create_graph=False)[0]


def _validate_bounds(bounds: tuple[float, float] | None) -> None:
    if bounds is not None and not bounds[0] < bounds[1]:
        raise ValueError("bounds must satisfy low < high")


def _apply_bounds(x: Tensor, bounds: tuple[float, float] | None) -> Tensor:
    _validate_bounds(bounds)
    if bounds is None:
        return x
    low, high = bounds
    return x.clamp(min=low, max=high)


def _validate_langevin(step_size: float, temperature: float, noise_scale: float) -> None:
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")


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
    ``x' = x - h/(2T) * grad(E(x)) + sqrt(h) * epsilon``. ``bounds`` applies
    a documented projected-ULA variant and is not claimed to preserve the
    unconstrained target exactly.
    """

    _validate_langevin(step_size, temperature, noise_scale)
    _validate_bounds(bounds)
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


def vectorized_langevin_chain(
    model: nn.Module,
    x_init: Tensor,
    *,
    steps: int,
    step_size: float,
    temperature: float = 1.0,
    bounds: tuple[float, float] | None = None,
    noise_scale: float = 1.0,
    noise_sequence: Tensor | None = None,
    callback: Callable[[int, Tensor], None] | None = None,
) -> Tensor:
    """Run ULA while reusing one detached state buffer between gradient steps.

    This is numerically the same update as repeated :func:`ula_step` calls,
    but performs the state update in-place under ``no_grad`` after extracting
    each input gradient. It reduces tensor allocation churn in long CD or
    relaxation chains without attempting unsafe autograd-graph caching. For
    deterministic tests and benchmarks, ``noise_sequence`` may have shape
    ``(steps, *x_init.shape)``.
    """

    if steps < 0:
        raise ValueError("steps must be non-negative")
    _validate_langevin(step_size, temperature, noise_scale)
    _validate_bounds(bounds)
    if noise_sequence is not None and noise_sequence.shape != (steps, *x_init.shape):
        raise ValueError("noise_sequence must have shape (steps, *x_init.shape)")

    state = x_init.detach().clone()
    stochastic_coefficient = noise_scale * step_size**0.5
    drift_coefficient = step_size / (2.0 * temperature)
    for index in range(steps):
        state.requires_grad_(True)
        with torch.enable_grad():
            energy = model(state)
            if energy.ndim != 1:
                energy = energy.reshape(-1)
            gradient = torch.autograd.grad(energy.sum(), state, create_graph=False)[0]
        with torch.no_grad():
            state.add_(gradient, alpha=-drift_coefficient)
            if stochastic_coefficient:
                noise = (
                    noise_sequence[index]
                    if noise_sequence is not None
                    else torch.randn_like(state)
                )
                state.add_(noise, alpha=stochastic_coefficient)
            if bounds is not None:
                state.clamp_(min=bounds[0], max=bounds[1])
            state.requires_grad_(False)
        if callback is not None:
            callback(index, state)
    return state.detach()


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