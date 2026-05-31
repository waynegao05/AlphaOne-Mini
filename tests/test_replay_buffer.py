"""selfplay/replay_buffer.py 测试。"""

from __future__ import annotations

import numpy as np
import pytest

from selfplay.replay_buffer import ReplayBuffer
from selfplay.self_play import SelfPlaySample


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _make_sample(seed: int, z: float = 0.0) -> SelfPlaySample:
    rng = np.random.default_rng(seed)
    state = rng.random((4, 15, 15), dtype=np.float64).astype(np.float32)
    pi = rng.random(225, dtype=np.float64).astype(np.float32)
    pi = pi / pi.sum()
    return SelfPlaySample(state=state, pi=pi, z=float(z))


# ---------------------------------------------------------------------------
# 基本属性
# ---------------------------------------------------------------------------
class TestReplayBufferBasic:
    def test_initial_length_is_zero(self):
        buf = ReplayBuffer(capacity=8)
        assert len(buf) == 0

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=0)
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=-1)

    def test_add_increments_length(self):
        buf = ReplayBuffer(capacity=8)
        buf.add(_make_sample(0))
        assert len(buf) == 1
        buf.add(_make_sample(1))
        assert len(buf) == 2

    def test_extend_increments_length(self):
        buf = ReplayBuffer(capacity=8)
        buf.extend([_make_sample(i) for i in range(3)])
        assert len(buf) == 3

    def test_clear(self):
        buf = ReplayBuffer(capacity=8)
        buf.extend([_make_sample(i) for i in range(3)])
        buf.clear()
        assert len(buf) == 0


# ---------------------------------------------------------------------------
# 容量与 FIFO
# ---------------------------------------------------------------------------
class TestReplayBufferCapacity:
    def test_overflow_drops_oldest(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            sample = SelfPlaySample(
                state=np.full((4, 15, 15), float(i), dtype=np.float32),
                pi=np.zeros(225, dtype=np.float32),
                z=float(i),
            )
            buf.add(sample)
        assert len(buf) == 3
        states, _, values = buf.to_arrays()
        # 应该保留最近的 i=2,3,4
        np.testing.assert_array_equal(states[0], np.full((4, 15, 15), 2.0))
        np.testing.assert_array_equal(states[1], np.full((4, 15, 15), 3.0))
        np.testing.assert_array_equal(states[2], np.full((4, 15, 15), 4.0))
        np.testing.assert_array_equal(values.flatten(), np.array([2.0, 3.0, 4.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------
class TestReplayBufferSample:
    def test_sample_shapes_and_dtypes(self):
        buf = ReplayBuffer(capacity=10, seed=0)
        for i in range(8):
            buf.add(_make_sample(i, z=1.0 if i % 2 == 0 else -1.0))

        states, policies, values = buf.sample(4)
        assert states.shape == (4, 4, 15, 15)
        assert states.dtype == np.float32
        assert policies.shape == (4, 225)
        assert policies.dtype == np.float32
        assert values.shape == (4, 1)
        assert values.dtype == np.float32

    def test_sample_batch_too_large_raises(self):
        buf = ReplayBuffer(capacity=10, seed=0)
        buf.extend([_make_sample(i) for i in range(3)])
        with pytest.raises(ValueError):
            buf.sample(4)

    def test_sample_seed_reproducible(self):
        buf = ReplayBuffer(capacity=10, seed=42)
        for i in range(8):
            buf.add(_make_sample(i, z=float(i)))
        states_a, _, values_a = buf.sample(3)

        buf2 = ReplayBuffer(capacity=10, seed=42)
        for i in range(8):
            buf2.add(_make_sample(i, z=float(i)))
        states_b, _, values_b = buf2.sample(3)

        np.testing.assert_array_equal(states_a, states_b)
        np.testing.assert_array_equal(values_a, values_b)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------
class TestReplayBufferSaveLoad:
    def test_save_creates_file(self, tmp_path):
        buf = ReplayBuffer(capacity=10)
        for i in range(3):
            buf.add(_make_sample(i, z=float(i - 1)))
        path = tmp_path / "data" / "buf.npz"  # 子目录不存在，应自动创建
        buf.save(str(path))
        assert path.exists()

    def test_save_then_load_round_trip(self, tmp_path):
        buf = ReplayBuffer(capacity=10)
        for i in range(5):
            buf.add(_make_sample(i, z=float(i % 3 - 1)))
        states_before, policies_before, values_before = buf.to_arrays()

        path = tmp_path / "buf.npz"
        buf.save(str(path))

        buf2 = ReplayBuffer(capacity=10)
        buf2.load(str(path))
        assert len(buf2) == len(buf)

        states_after, policies_after, values_after = buf2.to_arrays()
        np.testing.assert_array_equal(states_before, states_after)
        np.testing.assert_array_equal(policies_before, policies_after)
        np.testing.assert_array_equal(values_before, values_after)

    def test_load_missing_file_raises(self, tmp_path):
        buf = ReplayBuffer(capacity=10)
        with pytest.raises(FileNotFoundError):
            buf.load(str(tmp_path / "nope.npz"))

    def test_load_clears_existing_content(self, tmp_path):
        # buf1 写 3 条
        buf1 = ReplayBuffer(capacity=10)
        for i in range(3):
            buf1.add(_make_sample(i))
        path = tmp_path / "buf.npz"
        buf1.save(str(path))

        # buf2 先放 5 条，然后 load 应只剩下 3 条
        buf2 = ReplayBuffer(capacity=10)
        for i in range(5):
            buf2.add(_make_sample(100 + i))
        assert len(buf2) == 5

        buf2.load(str(path))
        assert len(buf2) == 3


# ---------------------------------------------------------------------------
# from_arrays / to_arrays
# ---------------------------------------------------------------------------
class TestArrayInterop:
    def test_from_arrays_then_to_arrays(self):
        states = np.random.rand(4, 4, 15, 15).astype(np.float32)
        policies = np.random.rand(4, 225).astype(np.float32)
        # 测试 (N,) 形状的 values 也能被吃掉
        values = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32)

        buf = ReplayBuffer(capacity=10)
        buf.from_arrays(states, policies, values)
        assert len(buf) == 4

        s2, p2, v2 = buf.to_arrays()
        assert s2.shape == (4, 4, 15, 15)
        assert p2.shape == (4, 225)
        assert v2.shape == (4, 1)
        np.testing.assert_array_equal(s2, states)
        np.testing.assert_array_equal(p2, policies)
        np.testing.assert_array_equal(v2.flatten(), values)

    def test_from_arrays_validates_shapes(self):
        buf = ReplayBuffer(capacity=10)
        with pytest.raises(ValueError):
            buf.from_arrays(
                np.zeros((2, 3, 15, 15), dtype=np.float32),  # channel 错
                np.zeros((2, 225), dtype=np.float32),
                np.zeros((2, 1), dtype=np.float32),
            )
        with pytest.raises(ValueError):
            buf.from_arrays(
                np.zeros((2, 4, 15, 15), dtype=np.float32),
                np.zeros((2, 100), dtype=np.float32),  # policy 长度错
                np.zeros((2, 1), dtype=np.float32),
            )
        with pytest.raises(ValueError):
            buf.from_arrays(
                np.zeros((3, 4, 15, 15), dtype=np.float32),  # 数量不一致
                np.zeros((2, 225), dtype=np.float32),
                np.zeros((2, 1), dtype=np.float32),
            )
