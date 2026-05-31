"""model/policy_value_net.py 与 model/checkpoint.py 的测试。

如果当前环境没有装 PyTorch，整个文件会被自动跳过(本批阶段不做硬依赖)。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from model.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from model.policy_value_net import PolicyValueNet  # noqa: E402


BOARD_SIZE = 15
CELLS = BOARD_SIZE * BOARD_SIZE


class CustomMetadata:
    def __init__(self, value):
        self.value = value


# ---- 实例化与 forward ------------------------------------------------------
def test_model_can_be_instantiated_with_defaults():
    model = PolicyValueNet()
    assert model.board_size == 15
    assert model.input_channels == 4
    assert model.hidden_channels == 64


def test_model_forward_shapes():
    model = PolicyValueNet()
    model.eval()
    x = torch.zeros(2, 4, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        policy_logits, value = model(x)
    assert policy_logits.shape == (2, CELLS)
    assert value.shape == (2, 1)


def test_model_forward_supports_custom_dimensions():
    model = PolicyValueNet(board_size=9, input_channels=5, hidden_channels=16)
    model.eval()
    x = torch.zeros(2, 5, 9, 9)
    with torch.no_grad():
        policy_logits, value = model(x)
    assert policy_logits.shape == (2, 81)
    assert value.shape == (2, 1)


def test_value_in_range_minus_one_to_one():
    model = PolicyValueNet()
    model.eval()
    x = torch.randn(4, 4, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        _, value = model(x)
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_policy_logits_are_unnormalized():
    """policy 端不应做 softmax，因此 ``exp(logits).sum()`` 通常 != 1。"""
    model = PolicyValueNet()
    model.eval()
    x = torch.randn(1, 4, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        policy_logits, _ = model(x)
    sums = policy_logits.exp().sum(dim=1)
    # 不强制要求 sum != 1，但要求 logits 没有被强行归一化成概率分布。
    # 用一个非常宽松的判断：至少一种输入下 sum 与 1 差距明显。
    assert torch.any((sums - 1.0).abs() > 1e-3)


# ---- backward --------------------------------------------------------------
def test_model_backward_step():
    model = PolicyValueNet()
    model.train()
    x = torch.randn(2, 4, BOARD_SIZE, BOARD_SIZE, requires_grad=False)
    policy_logits, value = model(x)

    target_policy = torch.randn(2, CELLS)
    target_value = torch.randn(2, 1).clamp(-1.0, 1.0)
    loss = F.mse_loss(policy_logits, target_policy) + F.mse_loss(value, target_value)
    loss.backward()

    # 关键参数应当有梯度
    assert model.conv1.weight.grad is not None
    assert model.policy_fc.weight.grad is not None
    assert model.value_fc2.weight.grad is not None


# ---- checkpoint ------------------------------------------------------------
def _params_equal(model_a, model_b) -> bool:
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    if sd_a.keys() != sd_b.keys():
        return False
    for k in sd_a:
        if not torch.equal(sd_a[k], sd_b[k]):
            return False
    return True


def test_save_checkpoint_creates_file(tmp_path):
    model = PolicyValueNet()
    path = tmp_path / "subdir" / "ckpt.pt"  # 父目录不存在，应自动创建
    save_checkpoint(model, str(path))
    assert path.exists()


def test_save_and_load_round_trip(tmp_path):
    model = PolicyValueNet()
    model.eval()  # 用 eval 模式让 BN 用 running stats，避免 batch 抖动
    x = torch.randn(2, 4, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        p_before, v_before = model(x)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, str(path))

    fresh = PolicyValueNet()
    # 未加载前参数应不一致
    assert not _params_equal(model, fresh)

    state = load_checkpoint(fresh, str(path))
    assert "model_state_dict" in state
    assert _params_equal(model, fresh)

    fresh.eval()
    with torch.no_grad():
        p_after, v_after = fresh(x)

    assert torch.allclose(p_before, p_after, atol=1e-6)
    assert torch.allclose(v_before, v_after, atol=1e-6)


def test_save_and_load_with_optimizer_and_metadata(tmp_path):
    model = PolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 跑一次假训练让 optimizer 有 state
    x = torch.randn(2, 4, BOARD_SIZE, BOARD_SIZE)
    policy_logits, value = model(x)
    loss = policy_logits.sum() + value.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    metadata = {"epoch": 3, "note": "demo"}
    path = tmp_path / "ckpt.pt"
    saved = save_checkpoint(model, str(path), optimizer=optimizer, metadata=metadata)
    assert "optimizer_state_dict" in saved
    assert saved["metadata"] == metadata

    # 用一个新模型 / 新 optimizer 加载
    fresh = PolicyValueNet()
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    state = load_checkpoint(fresh, str(path), optimizer=fresh_opt)

    assert _params_equal(model, fresh)
    assert state.get("metadata") == metadata
    # optimizer 状态应被恢复(至少包含 param_groups + state)
    fresh_opt_state = fresh_opt.state_dict()
    assert "param_groups" in fresh_opt_state
    assert len(fresh_opt_state["state"]) > 0


def test_load_checkpoint_preserves_custom_metadata_object(tmp_path):
    model = PolicyValueNet()
    metadata = {"custom": CustomMetadata(value=7)}
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, str(path), metadata=metadata)

    fresh = PolicyValueNet()
    state = load_checkpoint(fresh, str(path), device="cpu")

    assert isinstance(state["metadata"]["custom"], CustomMetadata)
    assert state["metadata"]["custom"].value == 7


def test_load_checkpoint_missing_file_raises(tmp_path):
    model = PolicyValueNet()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(model, str(tmp_path / "does_not_exist.pt"))
