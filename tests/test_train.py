"""train/train.py 的单元 + 集成测试。"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model.checkpoint import load_checkpoint  # noqa: E402
from model.policy_value_net import PolicyValueNet  # noqa: E402
from train.train import (  # noqa: E402
    SelfPlayDataset,
    create_dataloader,
    load_selfplay_npz,
    train_model,
    train_one_epoch,
)


# ---------------------------------------------------------------------------
# 工具：构造小批量假数据 / 写 npz
# ---------------------------------------------------------------------------
def _make_data(n: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    states = rng.random((n, 4, 15, 15), dtype=np.float64).astype(np.float32)
    raw_pi = rng.random((n, 225), dtype=np.float64)
    policies = (raw_pi / raw_pi.sum(axis=1, keepdims=True)).astype(np.float32)
    values = rng.choice(np.array([-1.0, 0.0, 1.0]), size=(n, 1)).astype(np.float32)
    return states, policies, values


def _save_npz(path: str, states, policies, values):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(path, states=states, policies=policies, values=values)


# ===========================================================================
# SelfPlayDataset
# ===========================================================================
class TestSelfPlayDataset:
    def test_length(self):
        states, policies, values = _make_data(n=8)
        ds = SelfPlayDataset(states, policies, values)
        assert len(ds) == 8

    def test_item_shapes_and_dtypes(self):
        states, policies, values = _make_data(n=4)
        ds = SelfPlayDataset(states, policies, values)
        s, p, v = ds[0]
        assert s.shape == (4, 15, 15)
        assert p.shape == (225,)
        assert v.shape == (1,)
        assert s.dtype == torch.float32
        assert p.dtype == torch.float32
        assert v.dtype == torch.float32

    def test_handles_1d_values(self):
        states, policies, _ = _make_data(n=4)
        values_1d = np.array([-1.0, 0.0, 1.0, 1.0], dtype=np.float32)  # shape (4,)
        ds = SelfPlayDataset(states, policies, values_1d)
        _, _, v = ds[0]
        assert v.shape == (1,)

    def test_invalid_state_shape_raises(self):
        states, policies, values = _make_data(n=4)
        bad = states.reshape(4, 4 * 15 * 15)
        with pytest.raises(ValueError):
            SelfPlayDataset(bad, policies, values)

    def test_invalid_policy_shape_raises(self):
        states, _, values = _make_data(n=4)
        bad_policies = np.zeros((4, 100), dtype=np.float32)
        with pytest.raises(ValueError):
            SelfPlayDataset(states, bad_policies, values)


# ===========================================================================
# load_selfplay_npz
# ===========================================================================
class TestLoadNPZ:
    def test_round_trip(self, tmp_path):
        states, policies, values = _make_data(n=5)
        path = tmp_path / "data.npz"
        _save_npz(str(path), states, policies, values)
        loaded_s, loaded_p, loaded_v = load_selfplay_npz(str(path))
        np.testing.assert_array_equal(loaded_s, states)
        np.testing.assert_array_equal(loaded_p, policies)
        np.testing.assert_array_equal(loaded_v, values)
        assert loaded_v.shape == (5, 1)
        assert loaded_v.dtype == np.float32

    def test_handles_1d_values_in_npz(self, tmp_path):
        states, policies, _ = _make_data(n=4)
        values_1d = np.array([-1.0, 0.0, 1.0, 1.0], dtype=np.float32)
        path = tmp_path / "data.npz"
        _save_npz(str(path), states, policies, values_1d)
        _, _, loaded_v = load_selfplay_npz(str(path))
        assert loaded_v.shape == (4, 1)
        assert loaded_v.dtype == np.float32

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_selfplay_npz(str(tmp_path / "nope.npz"))

    def test_bad_state_shape_raises(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez_compressed(
            str(path),
            states=np.zeros((4, 3, 15, 15), dtype=np.float32),  # 通道数错
            policies=np.zeros((4, 225), dtype=np.float32),
            values=np.zeros((4, 1), dtype=np.float32),
        )
        with pytest.raises(ValueError):
            load_selfplay_npz(str(path))

    def test_bad_policy_shape_raises(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez_compressed(
            str(path),
            states=np.zeros((4, 4, 15, 15), dtype=np.float32),
            policies=np.zeros((4, 100), dtype=np.float32),
            values=np.zeros((4, 1), dtype=np.float32),
        )
        with pytest.raises(ValueError):
            load_selfplay_npz(str(path))

    def test_size_mismatch_raises(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez_compressed(
            str(path),
            states=np.zeros((4, 4, 15, 15), dtype=np.float32),
            policies=np.zeros((3, 225), dtype=np.float32),
            values=np.zeros((4, 1), dtype=np.float32),
        )
        with pytest.raises(ValueError):
            load_selfplay_npz(str(path))


# ===========================================================================
# create_dataloader
# ===========================================================================
class TestCreateDataloader:
    def test_yields_batches_with_correct_shapes(self):
        states, policies, values = _make_data(n=8)
        loader = create_dataloader(
            states, policies, values, batch_size=4, shuffle=False
        )
        batches = list(loader)
        assert len(batches) == 2
        s, p, v = batches[0]
        assert s.shape == (4, 4, 15, 15)
        assert p.shape == (4, 225)
        assert v.shape == (4, 1)
        assert s.dtype == torch.float32


# ===========================================================================
# train_one_epoch
# ===========================================================================
class TestTrainOneEpoch:
    def test_returns_loss_stats(self):
        states, policies, values = _make_data(n=8)
        loader = create_dataloader(states, policies, values, batch_size=4, shuffle=False)

        torch.manual_seed(0)
        model = PolicyValueNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        stats = train_one_epoch(model, loader, optimizer, device="cpu")
        for key in ("total_loss", "policy_loss", "value_loss", "num_batches", "num_samples"):
            assert key in stats
        assert stats["num_samples"] == 8
        assert stats["num_batches"] == 2
        assert stats["total_loss"] >= 0.0

    def test_changes_some_parameters(self):
        states, policies, values = _make_data(n=8)
        loader = create_dataloader(states, policies, values, batch_size=4, shuffle=False)

        torch.manual_seed(0)
        model = PolicyValueNet()
        before = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        train_one_epoch(model, loader, optimizer, device="cpu", grad_clip=5.0)

        any_changed = False
        for name, param in model.named_parameters():
            if param.requires_grad and not torch.equal(before[name], param.detach()):
                any_changed = True
                break
        assert any_changed, "训练一轮后没有任何参数发生变化"


# ===========================================================================
# train_model
# ===========================================================================
class TestTrainModel:
    def test_runs_one_epoch_and_saves_checkpoint(self, tmp_path):
        states, policies, values = _make_data(n=8)
        data_path = tmp_path / "data.npz"
        _save_npz(str(data_path), states, policies, values)

        ckpt_dir = tmp_path / "checkpoints"

        torch.manual_seed(0)
        model = PolicyValueNet()
        history = train_model(
            model=model,
            data_path=str(data_path),
            checkpoint_dir=str(ckpt_dir),
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=1e-4,
            device="cpu",
            grad_clip=5.0,
        )
        assert isinstance(history, list)
        assert len(history) == 1
        for key in ("epoch", "loss", "policy_loss", "value_loss"):
            assert key in history[0]

        ckpt_path = ckpt_dir / "latest.pt"
        assert ckpt_path.exists()

    def test_runs_multiple_epochs(self, tmp_path):
        states, policies, values = _make_data(n=8)
        data_path = tmp_path / "data.npz"
        _save_npz(str(data_path), states, policies, values)

        ckpt_dir = tmp_path / "checkpoints"

        torch.manual_seed(0)
        model = PolicyValueNet()
        history = train_model(
            model=model,
            data_path=str(data_path),
            checkpoint_dir=str(ckpt_dir),
            epochs=3,
            batch_size=4,
        )
        assert len(history) == 3
        assert [h["epoch"] for h in history] == [1, 2, 3]

    def test_saved_checkpoint_can_be_loaded_and_used(self, tmp_path):
        states, policies, values = _make_data(n=8)
        data_path = tmp_path / "data.npz"
        _save_npz(str(data_path), states, policies, values)

        ckpt_dir = tmp_path / "checkpoints"
        torch.manual_seed(0)
        model = PolicyValueNet()
        train_model(
            model=model,
            data_path=str(data_path),
            checkpoint_dir=str(ckpt_dir),
            epochs=1,
            batch_size=4,
        )

        # 加载到一个新模型与新 optimizer
        fresh_model = PolicyValueNet()
        fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
        state = load_checkpoint(
            fresh_model, str(ckpt_dir / "latest.pt"), optimizer=fresh_opt
        )

        # metadata 应包含 spec 要求的字段
        assert "metadata" in state
        meta = state["metadata"]
        for key in ("epoch", "loss", "policy_loss", "value_loss", "data_path"):
            assert key in meta, f"metadata 缺失字段 {key}"
        assert meta["epoch"] == 1

        # 新模型应能 forward
        fresh_model.eval()
        x = torch.zeros(2, 4, 15, 15)
        with torch.no_grad():
            logits, value = fresh_model(x)
        assert logits.shape == (2, 225)
        assert value.shape == (2, 1)

    def test_checkpoint_round_trip_parameter_equivalence(self, tmp_path):
        """保存后加载，对相同输入，eval 模式下输出应该一致。"""
        states, policies, values = _make_data(n=8)
        data_path = tmp_path / "data.npz"
        _save_npz(str(data_path), states, policies, values)

        ckpt_dir = tmp_path / "checkpoints"
        torch.manual_seed(0)
        model = PolicyValueNet()
        train_model(
            model=model,
            data_path=str(data_path),
            checkpoint_dir=str(ckpt_dir),
            epochs=1,
            batch_size=4,
        )

        x = torch.randn(2, 4, 15, 15)
        model.eval()
        with torch.no_grad():
            logits_before, value_before = model(x)

        fresh = PolicyValueNet()
        load_checkpoint(fresh, str(ckpt_dir / "latest.pt"))
        fresh.eval()
        with torch.no_grad():
            logits_after, value_after = fresh(x)

        assert torch.allclose(logits_before, logits_after, atol=1e-5)
        assert torch.allclose(value_before, value_after, atol=1e-5)

    def test_missing_data_raises(self, tmp_path):
        torch.manual_seed(0)
        model = PolicyValueNet()
        with pytest.raises(FileNotFoundError):
            train_model(
                model=model,
                data_path=str(tmp_path / "nope.npz"),
                checkpoint_dir=str(tmp_path / "ckpt"),
                epochs=1,
                batch_size=4,
            )

    def test_train_model_uses_auxiliary_labels_when_present(self, tmp_path):
        from model.advanced_policy_value_net import AdvancedPolicyValueNet

        states, policies, values = _make_data(n=8)
        data_path = tmp_path / "advanced_data.npz"
        np.savez_compressed(
            str(data_path),
            states=states,
            policies=policies,
            values=values,
            threat_labels=np.zeros((8, 12, 15, 15), dtype=np.float32),
            forbidden_labels=np.zeros((8, 1, 15, 15), dtype=np.float32),
            tactical_scores=np.zeros((8, 225), dtype=np.float32),
        )

        model = AdvancedPolicyValueNet(blocks=1, channels=16)
        history = train_model(
            model=model,
            data_path=str(data_path),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            epochs=1,
            batch_size=4,
            device="cpu",
            use_auxiliary_loss=True,
        )

        assert history[0]["threat_loss"] > 0
        state = load_checkpoint(model, str(tmp_path / "checkpoints" / "latest.pt"))
        assert state["metadata"]["used_auxiliary_data"] is True
