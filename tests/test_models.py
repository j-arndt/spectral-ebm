import torch

from spectral_ebm.models import (
    BlockSpectralEBM,
    DenseEBM,
    PermutedSpectralEBM,
    SpectralEBM,
    parameter_count,
)


def test_models_return_scalar_energy_per_example() -> None:
    model = SpectralEBM(11, hidden_layers=3)
    assert model(torch.randn(4, 11)).shape == (4,)


def test_parameter_counts_are_explicitly_different() -> None:
    spectral = parameter_count(SpectralEBM(16, hidden_layers=3))
    dense = parameter_count(DenseEBM(16, hidden_layers=3))
    assert spectral == 7 * 16
    assert dense == 3 * 16 * 16 + 4 * 16


def test_block_spectral_ebm_mixes_channels_and_returns_scalar_energy() -> None:
    model = BlockSpectralEBM(channels=2, dim=7, hidden_layers=2, hidden_channels=3)
    assert model(torch.randn(5, 2, 7)).shape == (5,)
    assert parameter_count(model) == (3 * 2 * 7 + 3 * 7) + (3 * 3 * 7 + 3 * 7) + 3 * 7


def test_permuted_model_is_reproducible_and_serializes_permutations() -> None:
    torch.manual_seed(10)
    first = PermutedSpectralEBM(13, hidden_layers=3, seed=99)
    torch.manual_seed(10)
    second = PermutedSpectralEBM(13, hidden_layers=3, seed=99)
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter)
    for first_permutation, second_permutation in zip(first.permutations, second.permutations):
        torch.testing.assert_close(first_permutation, second_permutation)
    assert all(name.startswith("permutation_") for name in first.state_dict() if "permutation" in name)
    assert first(torch.randn(4, 13)).shape == (4,)


def test_permuted_model_uses_no_trainable_parameters_for_shuffles() -> None:
    model = PermutedSpectralEBM(16, hidden_layers=3)
    assert len(model.permutations) == 2
    assert all(not permutation.requires_grad for permutation in model.permutations)
    assert parameter_count(model) == parameter_count(SpectralEBM(16, hidden_layers=3))

def test_permutations_are_not_cyclic_shifts() -> None:
    model = PermutedSpectralEBM(13, hidden_layers=4, seed=7)
    identity = torch.arange(13)
    for permutation in model.permutations:
        assert all(not torch.equal(permutation, torch.roll(identity, shift)) for shift in range(13))