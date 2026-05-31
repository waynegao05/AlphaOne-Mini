"""Tests for record-based supervised pretraining helpers."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from game.board import BOARD_SIZE
from game.encoder import action_to_index
from model.policy_value_net import PolicyValueNet


def test_build_supervised_samples_from_record_extracts_policy_targets():
    from train.supervised_pretrain import build_supervised_samples_from_record

    record = "{C5;B(H,8);W(H,9);B(I,8)}"
    states, policies, values = build_supervised_samples_from_record(record)

    assert states.shape == (3, 4, BOARD_SIZE, BOARD_SIZE)
    assert policies.shape == (3, BOARD_SIZE * BOARD_SIZE)
    assert values.shape == (3, 1)
    assert int(np.argmax(policies[0])) == action_to_index(7, 7)
    assert policies[0, action_to_index(7, 7)] == np.float32(1.0)
    assert values.dtype == np.float32


def test_build_supervised_dataset_from_records_round_trip(tmp_path):
    from train.supervised_pretrain import (
        build_supervised_dataset_from_records,
        load_supervised_npz,
    )

    path = tmp_path / "records.npz"
    states, policies, values = build_supervised_dataset_from_records(
        ["B(H,8);W(H,9)", "B(J,10);W(H,8)"],
        output_path=str(path),
    )

    assert path.exists()
    assert states.shape[0] == 4
    np.testing.assert_allclose(policies.sum(axis=1), np.ones(4))
    loaded = load_supervised_npz(str(path))
    np.testing.assert_array_equal(loaded[0], states)
    np.testing.assert_array_equal(loaded[1], policies)
    np.testing.assert_array_equal(loaded[2], values)


def test_invalid_record_raises_clear_error():
    from train.supervised_pretrain import build_supervised_samples_from_record

    with pytest.raises(ValueError, match="record"):
        build_supervised_samples_from_record("B(H,8);B(I,8)")


def test_train_policy_pretrain_updates_model_and_saves_checkpoint(tmp_path, capsys):
    from model.checkpoint import load_checkpoint
    from train.supervised_pretrain import (
        build_supervised_dataset_from_records,
        train_policy_pretrain,
    )

    data_path = tmp_path / "records.npz"
    build_supervised_dataset_from_records(
        ["B(H,8);W(H,9);B(I,8);W(I,9)"],
        output_path=str(data_path),
    )

    torch.manual_seed(0)
    model = PolicyValueNet()
    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    history = train_policy_pretrain(
        model,
        data_path=str(data_path),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device="cpu",
    )

    assert len(history) == 1
    assert "policy_loss" in history[0]
    output = capsys.readouterr().out
    assert "[train] START supervised_pretrain" in output
    assert "[train] epoch 1/1 complete" in output
    assert "[train] DONE supervised_pretrain" in output
    ckpt_path = tmp_path / "checkpoints" / "pretrained.pt"
    assert ckpt_path.exists()
    assert any(
        not torch.equal(before[name], param.detach())
        for name, param in model.named_parameters()
        if param.requires_grad
    )

    fresh = PolicyValueNet()
    state = load_checkpoint(fresh, str(ckpt_path), device="cpu")
    assert state["metadata"]["pretrain_type"] == "supervised_policy"


def test_pretrain_dataset_can_augment_on_the_fly():
    from train.supervised_pretrain import PretrainDataset

    arrays = {
        "states": np.zeros((2, 4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        "policies": np.zeros((2, BOARD_SIZE * BOARD_SIZE), dtype=np.float32),
        "values": np.zeros((2, 1), dtype=np.float32),
        "threat_labels": np.zeros((2, 12, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        "forbidden_labels": np.zeros((2, 1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        "tactical_scores": np.zeros((2, BOARD_SIZE * BOARD_SIZE), dtype=np.float32),
    }
    arrays["states"][0, 0, 7, 7] = 1.0
    arrays["policies"][0, action_to_index(7, 7)] = 1.0

    dataset = PretrainDataset(arrays, augment=True)

    assert len(dataset) == 16
    sample = dataset[0]
    assert sample["states"].shape == (4, BOARD_SIZE, BOARD_SIZE)
    assert sample["policies"].shape == (BOARD_SIZE * BOARD_SIZE,)
    assert sample["values"].shape == (1,)
    assert sample["threat_labels"].shape == (12, BOARD_SIZE, BOARD_SIZE)
    assert sample["forbidden_labels"].shape == (1, BOARD_SIZE, BOARD_SIZE)
    assert sample["tactical_scores"].shape == (BOARD_SIZE * BOARD_SIZE,)
