import torch

from spectral_ebm.baselines import DiagonalEBM, DiagonalLinear
from spectral_ebm.models import parameter_count


def test_diagonal_layer_is_elementwise() -> None:
    layer = DiagonalLinear(4)
    x = torch.randn(3, 4)
    torch.testing.assert_close(layer(x), x * layer.scale + layer.bias)


def test_diagonal_ebm_has_linear_parameter_scaling() -> None:
    model = DiagonalEBM(16, hidden_layers=3)
    assert model(torch.randn(2, 16)).shape == (2,)
    assert parameter_count(model) == 7 * 16
