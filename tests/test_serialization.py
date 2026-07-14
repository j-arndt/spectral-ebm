import io

import torch

from spectral_ebm.layers import CirculantLinear


def test_state_dict_round_trip_preserves_output() -> None:
    source = CirculantLinear(9)
    restored = CirculantLinear(9)
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    x = torch.randn(4, 9)
    torch.testing.assert_close(source(x), restored(x))
