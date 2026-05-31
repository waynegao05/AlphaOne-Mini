"""train/loss.py 的单元测试。"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from train.loss import alphazero_loss, policy_loss_fn, value_loss_fn  # noqa: E402


# ---- policy_loss_fn --------------------------------------------------------
class TestPolicyLoss:
    def test_uniform_logits_uniform_target_equals_log_n(self):
        # logits=0, target=均匀 -> loss = log(N)
        n = 225
        logits = torch.zeros(2, n)
        target = torch.full((2, n), 1.0 / n)
        loss = policy_loss_fn(logits, target)
        expected = math.log(n)
        assert loss.item() == pytest.approx(expected, abs=1e-5)

    def test_one_hot_target_matches_cross_entropy_index(self):
        # 用 1-hot 目标作 cross-entropy，等价于直接取 -log_softmax 的对应位置
        torch.manual_seed(0)
        logits = torch.randn(3, 5)
        target_idx = torch.tensor([0, 2, 4])
        target = F.one_hot(target_idx, num_classes=5).float()
        loss = policy_loss_fn(logits, target)
        expected = F.cross_entropy(logits, target_idx)
        assert loss.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_handles_soft_target_distribution(self):
        # 任意合法概率分布，不会出现非有限值
        torch.manual_seed(1)
        logits = torch.randn(4, 225)
        soft_target = F.softmax(torch.randn(4, 225), dim=1)
        loss = policy_loss_fn(logits, soft_target)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_does_not_require_pre_softmax_logits(self):
        # 故意把 logits 放大 / 缩小，loss 仍然有限可计算
        target = F.softmax(torch.randn(2, 225), dim=1)
        for scale in (0.0, 1e-3, 1.0, 100.0):
            logits = torch.randn(2, 225) * scale
            loss = policy_loss_fn(logits, target)
            assert torch.isfinite(loss)

    def test_value_loss_lifts_1d_prediction(self):
        pred = torch.tensor([0.5, -0.5])  # shape (2,)
        target = torch.tensor([[1.0], [-1.0]])
        loss = value_loss_fn(pred, target)
        loss_2d = value_loss_fn(pred.unsqueeze(-1), target)
        assert loss.item() == pytest.approx(loss_2d.item(), abs=1e-7)

    def test_shape_mismatch_raises(self):
        logits = torch.zeros(2, 225)
        target = torch.zeros(2, 100)
        with pytest.raises(ValueError):
            policy_loss_fn(logits, target)


# ---- value_loss_fn ---------------------------------------------------------
class TestValueLoss:
    def test_value_loss_is_nonneg(self):
        pred = torch.tensor([[0.5], [-0.3], [0.0]])
        target = torch.tensor([[1.0], [-1.0], [0.0]])
        loss = value_loss_fn(pred, target)
        assert loss.item() >= 0.0

    def test_value_loss_zero_when_match(self):
        pred = torch.tensor([[1.0], [-1.0], [0.0]])
        target = torch.tensor([[1.0], [-1.0], [0.0]])
        loss = value_loss_fn(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_value_loss_lifts_1d_target(self):
        # target 是 [batch] 形状时也能算
        pred = torch.tensor([[0.5], [-0.5]])
        target = torch.tensor([1.0, -1.0])  # shape (2,)
        loss = value_loss_fn(pred, target)
        # 等价于把 target 升到 [2,1]
        loss_2d = value_loss_fn(pred, target.unsqueeze(-1))
        assert loss.item() == pytest.approx(loss_2d.item(), abs=1e-7)

    def test_shape_mismatch_raises(self):
        pred = torch.zeros(2, 1)
        target = torch.zeros(3, 1)
        with pytest.raises(Exception):  # ValueError or runtime broadcast error
            value_loss_fn(pred, target)


# ---- alphazero_loss --------------------------------------------------------
class TestAlphaZeroLoss:
    def test_returns_three_components_and_sum(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 225, requires_grad=True)
        pred_value = torch.tanh(torch.randn(4, 1, requires_grad=True))
        target_policy = F.softmax(torch.randn(4, 225), dim=1)
        target_value = torch.tensor([[1.0], [-1.0], [0.0], [1.0]])

        total, p, v = alphazero_loss(logits, pred_value, target_policy, target_value)
        assert total.item() == pytest.approx(p.item() + v.item(), abs=1e-5)
        assert p.item() >= 0.0
        assert v.item() >= 0.0

    def test_can_backward(self):
        logits = torch.randn(2, 225, requires_grad=True)
        pred_value = torch.randn(2, 1, requires_grad=True)
        target_policy = F.softmax(torch.randn(2, 225), dim=1)
        target_value = torch.zeros(2, 1)

        total, _, _ = alphazero_loss(logits, pred_value, target_policy, target_value)
        total.backward()
        assert logits.grad is not None
        assert pred_value.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert torch.isfinite(pred_value.grad).all()

    def test_accepts_1d_target_value(self):
        logits = torch.zeros(3, 225)
        pred_value = torch.zeros(3, 1)
        target_policy = torch.full((3, 225), 1.0 / 225)
        target_value = torch.tensor([1.0, -1.0, 0.0])  # shape (3,)
        total, p, v = alphazero_loss(logits, pred_value, target_policy, target_value)
        assert torch.isfinite(total)
