import torch

from spectral_ebm.layers import CirculantLinear, circulant_matrix


def test_materialized_matrix_matches_fft_forward_for_odd_and_even_dims() -> None:
    generator = torch.tensor([0.2, -0.5, 1.3, 0.7, -0.1, 0.4])
    for dim in (1, 2, 5, 6):
        c = generator[:dim].clone()
        x = torch.linspace(-1.0, 1.0, dim)
        layer = CirculantLinear(dim, bias=False)
        with torch.no_grad():
            layer.generator.copy_(c)
        expected = circulant_matrix(c) @ x
        torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)


def test_layer_supports_leading_batch_dimensions_and_bias() -> None:
    layer = CirculantLinear(7, bias=True)
    x = torch.randn(2, 3, 7)
    assert layer(x).shape == x.shape


def test_gradients_match_materialized_reference() -> None:
    dim = 5
    layer = CirculantLinear(dim, bias=True).double()
    x = torch.randn(3, dim, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_spectral_norm_is_exact_for_circulant_map() -> None:
    generator = torch.randn(8)
    layer = CirculantLinear(8, bias=False)
    with torch.no_grad():
        layer.generator.copy_(generator)
    expected = torch.linalg.matrix_norm(circulant_matrix(generator), ord=2)
    torch.testing.assert_close(layer.spectral_norm(), expected, rtol=1e-5, atol=1e-5)
