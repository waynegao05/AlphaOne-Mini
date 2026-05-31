from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_small_resnet_forward_shapes_and_value_range():
    from model.resnet_policy_value_net import SmallResNetPolicyValueNet

    model = SmallResNetPolicyValueNet(blocks=2, channels=32)
    x = torch.randn(2, 4, 15, 15)
    policy_logits, value = model(x)

    assert policy_logits.shape == (2, 225)
    assert value.shape == (2, 1)
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)
    assert torch.any((policy_logits.exp().sum(dim=1) - 1.0).abs() > 1e-3)


def test_small_resnet_backward():
    from model.resnet_policy_value_net import SmallResNetPolicyValueNet

    model = SmallResNetPolicyValueNet(blocks=2, channels=16)
    x = torch.randn(2, 4, 15, 15)
    policy_logits, value = model(x)
    loss = policy_logits.mean() + value.mean()
    loss.backward()

    assert model.stem_conv.weight.grad is not None
