import torch

from spectral_ebm.formal_search import ASTNode, FormalProofSearchAdapter, HRREncoder, hrr_bind
from spectral_ebm.models import SpectralEBM


def test_hrr_bind_and_encoder_are_shape_correct_and_deterministic() -> None:
    left = torch.randn(2, 16)
    right = torch.randn(2, 16)
    assert hrr_bind(left, right).shape == left.shape
    first = HRREncoder(["by", "intro", "exact"], 16, max_length=8, seed=5)
    second = HRREncoder(["by", "intro", "exact"], 16, max_length=8, seed=5)
    states = [["by", "intro", "h"], ["exact", "h"]]
    torch.testing.assert_close(first.encode(states), second.encode(states))
    torch.testing.assert_close(first.encode(states).norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_ast_tokens_and_nearest_token_diagnostic() -> None:
    ast = ASTNode("app", (ASTNode("identifier", ("foo",)), "bar"))
    assert ast.tokens() == ("app", "identifier", "foo", "bar")
    encoder = HRREncoder(["app", "identifier", "foo", "bar"], 12, max_length=8)
    nearest = encoder.nearest_tokens(encoder.encode([ast]), top_k=3)
    assert len(nearest) == 1
    assert len(nearest[0]) == 3


def test_formal_proof_adapter_runs_hrr_to_persistent_chain() -> None:
    encoder = HRREncoder(["by", "intro", "exact", "h"], 16, max_length=16, seed=3)
    adapter = FormalProofSearchAdapter(encoder, SpectralEBM(16, hidden_layers=1))
    result = adapter.refine(
        [["by", "intro", "h"], ["exact", "h"]],
        steps=2,
        step_size=0.01,
        noise_scale=0.0,
    )
    assert result.states == (("by", "intro", "h"), ("exact", "h"))
    assert result.initial_embeddings.shape == (2, 16)
    assert result.refined_embeddings.shape == (2, 16)
    assert result.initial_energy.shape == (2,)
    assert result.refined_energy.shape == (2,)
    assert torch.isfinite(result.refined_embeddings).all()
    assert adapter.chain.state is result.refined_embeddings
