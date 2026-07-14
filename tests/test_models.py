from spectral_ebm.models import DenseEBM, SpectralEBM, parameter_count


def test_models_return_scalar_energy_per_example() -> None:
    model = SpectralEBM(11, hidden_layers=3)
    assert model.__class__(11)(  # noqa: B018
        __import__("torch").randn(4, 11)
    ).shape == (4,)


def test_parameter_counts_are_explicitly_different() -> None:
    spectral = parameter_count(SpectralEBM(16, hidden_layers=3))
    dense = parameter_count(DenseEBM(16, hidden_layers=3))
    assert spectral == 7 * 16
    assert dense == 3 * 16 * 16 + 4 * 16
