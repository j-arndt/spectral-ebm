"""Optional persistent chains for short-run contrastive-divergence experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .sampler import langevin_sample


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
        self.state = langevin_sample(
            model,
            self.state,
            steps=steps,
            step_size=step_size,
            temperature=temperature,
            bounds=self.bounds,
        )
        return self.state
