import torch
from torch import nn

from spectral_ebm.sampler import ula_step


class QuadraticEnergy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x.square().sum(dim=-1)


def test_ula_step_matches_explicit_update_with_fixed_noise() -> None:
    model = QuadraticEnergy()
    x = torch.tensor([[1.0, -2.0]])
    noise = torch.tensor([[0.5, -0.25]])
    result = ula_step(model, x, step_size=0.2, temperature=2.0, noise=noise)
    expected = x - 0.2 / (2.0 * 2.0) * x + 0.2**0.5 * noise
    torch.testing.assert_close(result, expected)


def test_ula_bounds_are_applied_and_output_is_detached() -> None:
    result = ula_step(
        QuadraticEnergy(),
        torch.tensor([[2.0]], requires_grad=True),
        step_size=0.1,
        bounds=(-1.0, 1.0),
        noise_scale=0.0,
    )
    assert result.requires_grad is False
    assert result.item() <= 1.0
