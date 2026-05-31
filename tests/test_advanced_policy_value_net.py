from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_advanced_forward_tuple_and_aux_shapes():
    from model.advanced_policy_value_net import AdvancedPolicyValueNet

    model = AdvancedPolicyValueNet(blocks=2, channels=32)
    x = torch.randn(2, 4, 15, 15)

    policy_logits, value = model(x)
    outputs = model(x, return_aux=True)

    assert policy_logits.shape == (2, 225)
    assert value.shape == (2, 1)
    assert outputs["policy_logits"].shape == (2, 225)
    assert outputs["value"].shape == (2, 1)
    assert outputs["threat_logits"].shape == (2, 12, 15, 15)
    assert outputs["forbidden_logits"].shape == (2, 1, 15, 15)
    assert outputs["tactical_score"].shape == (2, 225)
    assert torch.all(outputs["value"] <= 1.0)
    assert torch.all(outputs["value"] >= -1.0)


def test_advanced_backward_and_is_larger_than_cnn():
    from model.advanced_policy_value_net import AdvancedPolicyValueNet
    from model.policy_value_net import PolicyValueNet

    cnn = PolicyValueNet()
    advanced = AdvancedPolicyValueNet(blocks=2, channels=32)
    x = torch.randn(2, 4, 15, 15)
    outputs = advanced(x, return_aux=True)
    loss = (
        outputs["policy_logits"].mean()
        + outputs["value"].mean()
        + outputs["threat_logits"].mean()
        + outputs["forbidden_logits"].mean()
        + outputs["tactical_score"].mean()
    )
    loss.backward()

    assert advanced.stem_conv.weight.grad is not None
    assert sum(p.numel() for p in advanced.parameters()) > sum(
        p.numel() for p in cnn.parameters()
    )
