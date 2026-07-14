import torch

from spectral_ebm.layers import (
    BlockCirculantLinear,
    CirculantLinear,
    block_circulant_matrix,
    circulant_matrix,
)


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


def test_block_circulant_reference_matches_fft_for_leading_batches() -> None:
    torch.manual_seed(4)
    layer = BlockCirculantLinear(2, 3, 5, bias=True)
    x = torch.randn(4, 2, 5)
    expected_matrix = block_circulant_matrix(layer.weight)
    expected = torch.stack(
        [
            (expected_matrix @ sample.flatten()).reshape(3, 5) + layer.bias
            for sample in x
        ]
    )
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)


def test_block_circulant_supports_extra_leading_dimensions() -> None:
    layer = BlockCirculantLinear(2, 4, 7, bias=False)
    assert layer(torch.randn(3, 5, 2, 7)).shape == (3, 5, 4, 7)


def test_block_circulant_parameter_count_and_gradients() -> None:
    layer = BlockCirculantLinear(2, 3, 4, bias=True).double()
    assert sum(parameter.numel() for parameter in layer.parameters()) == 3 * 2 * 4 + 3 * 4
    x = torch.randn(2, 2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_block_circulant_spectral_norm_matches_materialized_operator() -> None:
    layer = BlockCirculantLinear(2, 3, 5, bias=False)
    expected = torch.linalg.matrix_norm(layer.materialize(), ord=2)
    torch.testing.assert_close(layer.spectral_norm(), expected, rtol=1e-5, atol=1e-5)