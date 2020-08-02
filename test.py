"""Test contiguous parameter functions."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from contiguous_params import ContiguousParams


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_equal_optimizer_update(device):
    """Verify that the parameters are the same after a few updates."""
    x = torch.randn(1, 8).to(device)

    model_ref = nn.Sequential(*[nn.Linear(8, 8) for i in range(10)])
    model_ref = model_ref.to(device)
    optimizer = torch.optim.Adam(model_ref.parameters())
    
    model_c = deepcopy(model_ref)
    parameters_c = ContiguousParams(model_c.parameters())
    optimizer_c = torch.optim.Adam(parameters_c.contiguous())

    for model, optimizer in zip([model_ref, model_c], [optimizer, optimizer_c]):
        for step in range(5):
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # Verify that the model/optimizer did not modify the data or grad handle.
    parameters_c.assert_buffer_is_valid()

    # Verify that both models applied the same parameter updates.
    for p1, p2 in zip(model_ref.parameters(), model_c.parameters()):
        assert torch.allclose(p1.data, p2.data, atol=1e-06)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_buffer_invalidation_detection(device):
    """Verify that we recognize an invalidated buffer."""
    model = nn.Linear(8, 8)
    parameters = ContiguousParams(model.parameters())
    assert parameters.buffer_is_valid()
    # Invalidate the buffer.
    model.weight.data = model.weight + 4
    assert not parameters.buffer_is_valid()
    with pytest.raises(ValueError):
      parameters.assert_buffer_is_valid()
