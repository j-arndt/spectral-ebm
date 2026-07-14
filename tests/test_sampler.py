import torch
from torch import nn

from spectral_ebm.sampler import ula_step, vectorized_langevin_chain


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


def test_vectorized_chain_matches_repeated_ula_steps_with_fixed_noise() -> None:
    model = QuadraticEnergy()
    initial = torch.tensor([[1.0, -2.0]])
    noise = torch.tensor(
        [
            [[0.5, -0.25]],
            [[-0.1, 0.2]],
            [[0.3, 0.7]],
        ]
    )
    expected = initial
    for noise_step in noise:
        expected = ula_step(
            model,
            expected,
            step_size=0.2,
            temperature=2.0,
            noise=noise_step,
        )
    result = vectorized_langevin_chain(
        model,
        initial,
        steps=3,
        step_size=0.2,
        temperature=2.0,
        noise_sequence=noise,
    )
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(initial, torch.tensor([[1.0, -2.0]]))
    assert result.requires_grad is False


def test_vectorized_chain_applies_bounds_in_place() -> None:
    result = vectorized_langevin_chain(
        QuadraticEnergy(),
        torch.tensor([[2.0]], requires_grad=True),
        steps=3,
        step_size=0.1,
        bounds=(-1.0, 1.0),
        noise_scale=0.0,
    )
    assert result.item() <= 1.0
    assert result.requires_grad is False