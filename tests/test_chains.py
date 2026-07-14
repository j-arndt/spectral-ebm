import torch

from spectral_ebm.chains import PersistentLangevin
from spectral_ebm.models import SpectralEBM


def test_persistent_chain_reuses_state_and_can_reset() -> None:
    chain = PersistentLangevin(bounds=(-2.0, 2.0))
    model = SpectralEBM(4, hidden_layers=1)

    second = chain.sample(model, shape=(3, 4), steps=1, step_size=0.01)
    assert chain.state is second
    assert chain.state.device == next(model.parameters()).device
    assert torch.all(second.abs() <= 2.0)
    chain.reset()
    assert chain.state is None
