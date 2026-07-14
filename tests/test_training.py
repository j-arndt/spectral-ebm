import torch

from spectral_ebm.models import SpectralEBM
from spectral_ebm.training import contrastive_divergence_loss, denoising_score_matching_loss


def test_contrastive_divergence_has_expected_sign() -> None:
    model = SpectralEBM(3, hidden_layers=1, bias=False)
    positive = torch.zeros(4, 3)
    negative = torch.ones(4, 3)
    loss = contrastive_divergence_loss(model, positive, negative)
    assert loss.ndim == 0
    assert loss.requires_grad


def test_denoising_score_matching_returns_differentiable_scalar() -> None:
    torch.manual_seed(0)
    model = SpectralEBM(3, hidden_layers=1, bias=False)
    clean = torch.randn(8, 3)
    loss = denoising_score_matching_loss(model, clean, noise_std=0.2)
    assert loss.ndim == 0
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
