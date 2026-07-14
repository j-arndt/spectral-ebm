import torch
from torch import nn

from spectral_ebm import (
    AmortizedHouseholderPermutation,
    FormalProofSearchAdapter,
    HRREncoder,
    SpectralEBM,
    explicit_spherical_langevin_step,
)


def test_householder_mixer_is_norm_preserving_and_linear_parameterized() -> None:
    torch.manual_seed(5)
    layer = AmortizedHouseholderPermutation(dim=17, reflections=3)
    x = torch.randn(4, 17)
    y = layer(x)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=2e-5, rtol=2e-5)
    assert sum(parameter.numel() for parameter in layer.parameters()) == 3 * 17
    matrix = layer.matrix()
    torch.testing.assert_close(matrix.T @ matrix, torch.eye(17), atol=2e-5, rtol=2e-5)


def test_spherical_langevin_step_preserves_radius_and_has_finite_gradient() -> None:
    model = nn.Sequential(nn.Linear(13, 13), nn.SiLU(), nn.Linear(13, 1))
    x = torch.randn(4, 13)
    noise = torch.randn_like(x)
    y = explicit_spherical_langevin_step(
        model,
        x,
        step_size=0.02,
        noise=noise,
        noise_scale=1.0,
        radius=2.5,
    )
    torch.testing.assert_close(y.norm(dim=-1), torch.full((4,), 2.5), atol=2e-5, rtol=2e-5)
    assert torch.isfinite(y).all()


def test_formal_adapter_defaults_to_spherical_refinement() -> None:
    encoder = HRREncoder(["by", "exact", "h"], 20, max_length=8, seed=9)
    adapter = FormalProofSearchAdapter(encoder, SpectralEBM(20, hidden_layers=1))
    result = adapter.refine(
        [["by", "exact", "h"], ["exact", "h"]],
        steps=2,
        step_size=0.01,
        noise_scale=0.0,
    )
    torch.testing.assert_close(
        result.refined_embeddings.norm(dim=-1),
        torch.ones(2),
        atol=2e-5,
        rtol=2e-5,
    )


def test_permuted_spectral_ebm_supports_householder_mode() -> None:
    from spectral_ebm.models import PermutedSpectralEBM

    model = PermutedSpectralEBM(
        dim=13,
        hidden_layers=3,
        permutation_mode="householder",
        householder_reflections=2,
    )
    energy = model(torch.randn(4, 13))
    assert energy.shape == (4,)
    assert len(model.permutation_layers) == 2
    assert all(layer.vectors.numel() == 2 * 13 for layer in model.permutation_layers)
