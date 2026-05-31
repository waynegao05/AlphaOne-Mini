"""selfplay/self_play.py 的单元测试。

不依赖真正训练好的网络：用 ``FakeMCTSScripted`` / ``FakeMCTSStochastic`` 注入
确定性的 MCTS 输出，从而稳定地验证 z 标签、温度调度、轨迹合法性等。

PolicyValueNet 的真实 smoke 测试放在 ``test_mcts.py`` / ``test_model.py``，
本文件不强依赖 PyTorch。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest

from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import action_to_index, index_to_action, legal_moves_to_mask
from selfplay.self_play import (
    SelfPlayGame,
    SelfPlaySample,
    TrajectoryStep,
)


ACTION_SIZE = BOARD_SIZE * BOARD_SIZE


def _idx(x: int, y: int) -> int:
    return action_to_index(x, y, BOARD_SIZE)


# ---------------------------------------------------------------------------
# 假 MCTS：让测试结果完全确定
# ---------------------------------------------------------------------------
class FakeMCTSScripted:
    """按预设 action 序列依次输出 one-hot 概率。

    序列耗尽后退化为"第一个合法点"策略，保证 play_game 不死循环。
    """

    def __init__(self, action_sequence: Optional[List[int]] = None) -> None:
        self.action_sequence = list(action_sequence) if action_sequence else []
        self.call_count = 0
        self.temperature = 1.0
        self.last_pi: Optional[np.ndarray] = None

    def run(self, board: Board) -> np.ndarray:
        probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        if self.call_count < len(self.action_sequence):
            target = self.action_sequence[self.call_count]
            tx, ty = index_to_action(target, BOARD_SIZE)
            if board.is_legal_move(tx, ty):
                probs[target] = 1.0
                self.call_count += 1
                self.last_pi = probs
                return probs
        # 兜底：第一个合法点
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board.is_empty(x, y):
                    probs[_idx(x, y)] = 1.0
                    self.call_count += 1
                    self.last_pi = probs
                    return probs
        self.call_count += 1
        self.last_pi = probs
        return probs


class FakeMCTSStochastic:
    """返回一个非 one-hot 的合法概率，用于检验温度调度。"""

    def __init__(self) -> None:
        self.temperature = 1.0

    def run(self, board: Board) -> np.ndarray:
        mask = legal_moves_to_mask(board, BOARD_SIZE).astype(np.float64)
        if mask.sum() == 0:
            return np.zeros(ACTION_SIZE, dtype=np.float32)
        legal_indices = np.flatnonzero(mask)
        probs = np.zeros(ACTION_SIZE, dtype=np.float64)
        # 把概率集中在前两个合法点：60% / 40%
        first = legal_indices[0]
        probs[first] = 0.6
        if len(legal_indices) > 1:
            second = legal_indices[1]
            probs[second] = 0.4
        else:
            probs[first] = 1.0
        # 数值稳健归一
        probs = probs / probs.sum()
        return probs.astype(np.float32)


class FakeMCTSBadPolicy:
    """Return invalid policies so SelfPlayGame must sanitize them before use."""

    def __init__(self) -> None:
        self.temperature = 1.0

    def run(self, board: Board) -> np.ndarray:
        probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        probs[_idx(0, 0)] = 10.0
        probs[_idx(1, 0)] = -2.0
        probs[_idx(2, 0)] = np.nan
        return probs


# ---------------------------------------------------------------------------
# 实例化与基本属性
# ---------------------------------------------------------------------------
class TestSelfPlayGameBasic:
    def test_can_be_instantiated_with_mcts(self):
        game = SelfPlayGame(mcts=FakeMCTSScripted(), max_moves=8)
        assert game.board_size == BOARD_SIZE
        assert game.action_size == ACTION_SIZE
        assert game.max_moves == 8

    def test_requires_model_or_mcts(self):
        with pytest.raises(ValueError):
            SelfPlayGame()

    def test_temperature_schedule(self):
        game = SelfPlayGame(
            mcts=FakeMCTSScripted(),
            temperature=1.0,
            temperature_drop_step=5,
        )
        for i in range(5):
            assert game.get_temperature(i) == pytest.approx(1.0)
        for i in range(5, 10):
            assert game.get_temperature(i) <= 1e-3


# ---------------------------------------------------------------------------
# play_game：样本格式与合法性
# ---------------------------------------------------------------------------
class TestPlayGameOutput:
    def test_play_game_returns_list_of_samples(self):
        game = SelfPlayGame(
            mcts=FakeMCTSScripted(),
            temperature=0.0,
            temperature_drop_step=0,
            max_moves=4,
        )
        samples = game.play_game()
        assert isinstance(samples, list)
        assert all(isinstance(s, SelfPlaySample) for s in samples)
        assert 1 <= len(samples) <= 4

    def test_each_sample_state_and_pi_shape(self):
        game = SelfPlayGame(
            mcts=FakeMCTSScripted(),
            temperature=0.0,
            temperature_drop_step=0,
            max_moves=6,
        )
        samples = game.play_game()
        for s in samples:
            assert s.state.shape == (4, BOARD_SIZE, BOARD_SIZE)
            assert s.state.dtype == np.float32
            assert s.pi.shape == (ACTION_SIZE,)
            assert s.pi.dtype == np.float32
            assert s.z in (-1.0, 0.0, 1.0)
            assert (s.pi >= 0).all()
            assert s.pi.sum() == pytest.approx(1.0, abs=1e-5)

    def test_no_illegal_actions_in_trajectory(self):
        game = SelfPlayGame(
            mcts=FakeMCTSScripted(),
            temperature=0.0,
            temperature_drop_step=0,
            max_moves=10,
        )
        game.play_game()
        # 每一步的 action 都必须是落子前棋盘上合法的位置 ->
        # 通过重放轨迹来验证：依次用一个干净 board 落子，不应抛异常。
        replay = Board()
        for step in game.last_trajectory:
            x, y = index_to_action(step.action, BOARD_SIZE)
            assert replay.is_legal_move(x, y), f"轨迹中出现非法落子: ({x},{y})"
            replay.place_stone(x, y)

    def test_play_game_does_not_exceed_max_moves(self):
        game = SelfPlayGame(
            mcts=FakeMCTSScripted(),
            temperature=0.0,
            temperature_drop_step=0,
            max_moves=3,
        )
        samples = game.play_game()
        assert len(samples) <= 3
        assert game.last_move_count <= 3


    def test_play_game_sanitizes_mcts_policy_before_saving(self):
        game = SelfPlayGame(
            mcts=FakeMCTSBadPolicy(),
            temperature=1.0,
            temperature_drop_step=99,
            max_moves=2,
            rng=np.random.default_rng(seed=0),
        )
        samples = game.play_game()
        assert len(samples) == 2

        replay = Board()
        for sample, step in zip(samples, game.last_trajectory):
            assert sample.pi.shape == (ACTION_SIZE,)
            assert sample.pi.dtype == np.float32
            assert np.isfinite(sample.pi).all()
            assert (sample.pi >= 0.0).all()
            assert sample.pi.sum() == pytest.approx(1.0, abs=1e-5)

            x, y = index_to_action(step.action, BOARD_SIZE)
            assert replay.is_legal_move(x, y)
            for occupied_x in range(BOARD_SIZE):
                for occupied_y in range(BOARD_SIZE):
                    if not replay.is_empty(occupied_x, occupied_y):
                        assert sample.pi[_idx(occupied_x, occupied_y)] == 0.0
            replay.place_stone(x, y)


# ---------------------------------------------------------------------------
# z 标签的三种胜负场景
# ---------------------------------------------------------------------------
class TestZLabel:
    def test_black_wins_z_labels(self):
        # 黑：(0..4, 0) 五连胜；白：(0..3, 14) 安全落子
        actions = [
            _idx(0, 0),    # 1 B
            _idx(0, 14),   # 2 W
            _idx(1, 0),    # 3 B
            _idx(1, 14),   # 4 W
            _idx(2, 0),    # 5 B
            _idx(2, 14),   # 6 W
            _idx(3, 0),    # 7 B
            _idx(3, 14),   # 8 W
            _idx(4, 0),    # 9 B  -> 5 连
        ]
        game = SelfPlayGame(
            mcts=FakeMCTSScripted(actions),
            temperature=0.0,
            temperature_drop_step=0,
            max_moves=20,
        )
        samples = game.play_game()
        assert game.last_winner == BLACK
        assert len(samples) == 9

        # 黑回合: 偶数下标 0,2,4,6,8 -> z=+1
        for i in (0, 2, 4, 6, 8):
            assert samples[i].z == 1.0
        # 白回合: 奇数下标 1,3,5,7 -> z=-1
        for i in (1, 3, 5, 7):
            assert samples[i].z == -1.0

    def test_white_wins_z_labels(self):
        # 白：(0..4, 0) 5 连胜；黑用零散点防御失败
        actions = [
            _idx(7, 7),   # 1 B
            _idx(0, 0),   # 2 W
            _idx(0, 1),   # 3 B
            _idx(1, 0),   # 4 W
            _idx(0, 2),   # 5 B
            _idx(2, 0),   # 6 W
            _idx(0, 3),   # 7 B
            _idx(3, 0),   # 8 W
            _idx(0, 4),   # 9 B
            _idx(4, 0),   # 10 W -> 5 连
        ]
        game = SelfPlayGame(
            mcts=FakeMCTSScripted(actions),
            temperature=0.0,
            temperature_drop_step=0,
            max_moves=20,
        )
        samples = game.play_game()
        assert game.last_winner == WHITE
        assert len(samples) == 10

        # 黑回合: 0, 2, 4, 6, 8 -> z=-1
        for i in (0, 2, 4, 6, 8):
            assert samples[i].z == -1.0
        # 白回合: 1, 3, 5, 7, 9 -> z=+1
        for i in (1, 3, 5, 7, 9):
            assert samples[i].z == 1.0

    def test_draw_z_labels_via_max_moves(self):
        # 4 步无 5 连后被 max_moves 截断 -> 平局
        actions = [
            _idx(0, 0),    # 1 B
            _idx(14, 14),  # 2 W
            _idx(14, 0),   # 3 B
            _idx(0, 14),   # 4 W
        ]
        game = SelfPlayGame(
            mcts=FakeMCTSScripted(actions),
            temperature=0.0,
            temperature_drop_step=0,
            max_moves=4,
        )
        samples = game.play_game()
        assert game.last_winner == 0
        assert len(samples) == 4
        for s in samples:
            assert s.z == 0.0


# ---------------------------------------------------------------------------
# 温度调度的整盘验证
# ---------------------------------------------------------------------------
class TestTemperatureBehavior:
    def test_actions_after_drop_step_are_argmax(self):
        # 用一个非 one-hot 的 FakeMCTS：在阈值前可能采样不一定取最大，
        # 但阈值 *后* 的每一步一定等于 argmax(pi)。
        game = SelfPlayGame(
            mcts=FakeMCTSStochastic(),
            temperature=1.0,
            temperature_drop_step=2,
            max_moves=6,
            rng=np.random.default_rng(seed=0),
        )
        game.play_game()
        for i, step in enumerate(game.last_trajectory):
            if i >= 2:
                assert step.action == int(np.argmax(step.pi)), (
                    f"第 {i} 手在 drop_step 之后但选了非 argmax 动作"
                )


# ---------------------------------------------------------------------------
# play_games：多盘拼接
# ---------------------------------------------------------------------------
class TestPlayGames:
    def test_play_games_concatenates_samples(self):
        actions = [
            _idx(0, 0), _idx(0, 14), _idx(1, 0), _idx(1, 14),
            _idx(2, 0), _idx(2, 14), _idx(3, 0), _idx(3, 14),
            _idx(4, 0),  # B 5 连
        ]
        game = SelfPlayGame(
            mcts=FakeMCTSScripted(actions * 3),  # 让脚本足够长跑多盘
            temperature=0.0,
            temperature_drop_step=0,
            max_moves=20,
        )
        all_samples = game.play_games(2)
        # 第一盘脚本只够 9 步；第二盘脚本继续往下，但因为两盘共用一个 mcts，
        # 第二盘从脚本第 10 个动作开始，在新棋盘上很可能不再合法 -> 退化策略接管。
        # 重点验证：返回是 list、每条样本结构合法、总数 > 第一盘。
        assert isinstance(all_samples, list)
        for s in all_samples:
            assert s.state.shape == (4, BOARD_SIZE, BOARD_SIZE)
            assert s.pi.shape == (ACTION_SIZE,)
            assert s.z in (-1.0, 0.0, 1.0)
        assert len(all_samples) >= 9


# ---------------------------------------------------------------------------
# 与真实 PolicyValueNet + MCTS 的最小 smoke 测试(若装了 torch)
# ---------------------------------------------------------------------------
def test_smoke_with_real_pvnet_and_mcts():
    torch = pytest.importorskip("torch")  # 没装 torch 就跳过
    from mcts.mcts import MCTS  # noqa: WPS433
    from model.policy_value_net import PolicyValueNet  # noqa: WPS433

    torch.manual_seed(0)
    model = PolicyValueNet()
    model.eval()
    mcts = MCTS(model, num_simulations=4)
    game = SelfPlayGame(
        mcts=mcts,
        temperature=1.0,
        temperature_drop_step=0,
        max_moves=4,
        rng=np.random.default_rng(seed=0),
    )
    samples = game.play_game()
    assert len(samples) > 0
    for s in samples:
        assert s.state.shape == (4, BOARD_SIZE, BOARD_SIZE)
        assert s.pi.shape == (ACTION_SIZE,)
        assert s.z in (-1.0, 0.0, 1.0)
