from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_advanced_loss_works_without_auxiliary_labels():
    from train.advanced_loss import advanced_policy_value_loss

    outputs = {
        "policy_logits": torch.randn(2, 225, requires_grad=True),
        "value": torch.tanh(torch.randn(2, 1, requires_grad=True)),
    }
    target_policy = torch.softmax(torch.randn(2, 225), dim=1)
    target_value = torch.zeros(2, 1)

    losses = advanced_policy_value_loss(outputs, target_policy, target_value)
    losses["total_loss"].backward()

    assert losses["policy_loss"].item() > 0
    assert losses["value_loss"].item() >= 0
    assert losses["threat_loss"].item() == 0


def test_advanced_loss_uses_auxiliary_labels_and_backpropagates():
    from train.advanced_loss import advanced_policy_value_loss

    outputs = {
        "policy_logits": torch.randn(2, 225, requires_grad=True),
        "value": torch.tanh(torch.randn(2, 1, requires_grad=True)),
        "threat_logits": torch.randn(2, 12, 15, 15, requires_grad=True),
        "forbidden_logits": torch.randn(2, 1, 15, 15, requires_grad=True),
        "tactical_score": torch.randn(2, 225, requires_grad=True),
    }
    target_policy = torch.softmax(torch.randn(2, 225), dim=1)
    target_value = torch.zeros(2, 1)
    threat_labels = torch.zeros(2, 12, 15, 15)
    forbidden_labels = torch.zeros(2, 1, 15, 15)
    tactical_scores = torch.zeros(2, 225)

    losses = advanced_policy_value_loss(
        outputs,
        target_policy,
        target_value,
        threat_labels=threat_labels,
        forbidden_labels=forbidden_labels,
        tactical_score_labels=tactical_scores,
    )
    losses["total_loss"].backward()

    assert losses["threat_loss"].item() > 0
    assert losses["forbidden_loss"].item() > 0
    assert losses["tactical_score_loss"].item() > 0
