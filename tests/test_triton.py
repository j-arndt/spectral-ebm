import pytest
import torch

from spectral_ebm.layers import BlockCirculantLinear
from spectral_ebm.triton_backend import triton_runtime_available


@pytest.mark.skipif(not triton_runtime_available(), reason="CUDA, Triton, and a C compiler are required")
def test_triton_block_circulant_matches_torch_forward_and_gradients() -> None:
    torch.manual_seed(11)
    torch_layer = BlockCirculantLinear(2, 3, 16, bias=True, backend="torch").cuda()
    triton_layer = BlockCirculantLinear(2, 3, 16, bias=True, backend="triton").cuda()
    triton_layer.load_state_dict(torch_layer.state_dict())
    x_torch = torch.randn(4, 2, 16, device="cuda", requires_grad=True)
    x_triton = x_torch.detach().clone().requires_grad_(True)
    torch_output = torch_layer(x_torch)
    triton_output = triton_layer(x_triton)
    torch.testing.assert_close(triton_output, torch_output, rtol=2e-4, atol=2e-4)
    torch_output.square().sum().backward()
    triton_output.square().sum().backward()
    torch.testing.assert_close(x_triton.grad, x_torch.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        triton_layer.weight.grad,
        torch_layer.weight.grad,
        rtol=2e-3,
        atol=2e-3,
    )
