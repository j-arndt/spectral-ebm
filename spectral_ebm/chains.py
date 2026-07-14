"""Optional persistent chains for short-run contrastive-divergence experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .sampler import vectorized_langevin_chain


class PersistentLangevin:
    """Keep a detached ULA state between optimization steps.

    This is a practical CD device, not a guarantee that the state is at
    equilibrium. ``reset`` should be called when the target domain or model
    semantics change.
    """

    def __init__(self, *, bounds: tuple[float, float] | None = None) -> None:
        self.state: Tensor | None = None
        self.bounds = bounds

    def reset(self) -> None:
        self.state = None

    def _run(
        self,
        model: nn.Module,
        *,
        steps: int,
        step_size: float,
        temperature: float,
        noise_scale: float = 1.0,
    ) -> Tensor:
        if self.state is None:
            raise RuntimeError("persistent state is not initialized")
        self.state = vectorized_langevin_chain(
            model,
            self.state,
            steps=steps,
            step_size=step_size,
            temperature=temperature,
            bounds=self.bounds,
            noise_scale=noise_scale,
        )
        return self.state

    def sample(
        self,
        model: nn.Module,
        *,
        shape: tuple[int, ...],
        steps: int,
        step_size: float,
        temperature: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Initialize or reuse a random state, then run optimized ULA."""

        parameter = next(model.parameters(), None)
        if parameter is not None:
            device = parameter.device if device is None else device
            dtype = parameter.dtype if dtype is None else dtype
        if self.state is None or self.state.shape != shape:
            self.state = torch.empty(shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        else:
            if device is not None:
                self.state = self.state.to(device=device)
            if dtype is not None:
                self.state = self.state.to(dtype=dtype)
        return self._run(
            model,
            steps=steps,
            step_size=step_size,
            temperature=temperature,
        )

    def sample_from(
        self,
        model: nn.Module,
        initial_state: Tensor,
        *,
        steps: int,
        step_size: float,
        temperature: float = 1.0,
        bounds: tuple[float, float] | None = None,
        noise_scale: float = 1.0,
    ) -> Tensor:
        """Start a persistent optimized chain from an explicit continuous state."""

        if bounds is not None:
            self.bounds = bounds
        self.state = initial_state.detach().clone()
        return self._run(
            model,
            steps=steps,
            step_size=step_size,
            temperature=temperature,
            noise_scale=noise_scale,
        )
