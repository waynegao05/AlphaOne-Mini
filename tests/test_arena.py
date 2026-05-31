"""evaluate/arena.py 的单元 + 集成测试。

主体测试用 ScriptedPlayer / FirstLegalPlayer / IllegalPlayer 三种轻量假玩家，
保证测试稳定不依赖真实模型棋力。结尾有一个对真实 ModelMCTSPlayer 的 smoke
test(若没装 torch 自动跳过)。
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from evaluate.arena import Arena, GameResult, run_match
from evaluate.players import RandomPlayer
from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import action_to_index, index_to_action


def _idx(x: int, y: int) -> int:
    return action_to_index(x, y, BOARD_SIZE)


# ---------------------------------------------------------------------------
# 假玩家
# ---------------------------------------------------------------------------
class ScriptedPlayer:
    """按预设 action_index 序列依次落子；序列耗尽后退化为第一个合法点。"""

    def __init__(self, actions: Optional[List[int]] = None, name: str = "scripted") -> None:
        self.actions = list(actions) if actions else []
        self.call_count = 0
        self.name = name

    def select_action(self, board: Board) -> Optional[int]:
        if self.call_count < len(self.actions):
            target = self.actions[self.call_count]
            tx, ty = index_to_action(target, BOARD_SIZE)
            if board.is_legal_move(tx, ty):
                self.call_count += 1
                return target
        # 兜底：第一个合法点
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board.is_empty(x, y):
                    self.call_count += 1
                    return _idx(x, y)
        return None


class FirstLegalPlayer:
    """无状态：始终选第一个合法落子(行优先扫描)。"""

    def __init__(self, name: str = "first_legal") -> None:
        self.name = name

    def select_action(self, board: Board) -> Optional[int]:
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board.is_empty(x, y):
                    return _idx(x, y)
        return None


class IllegalPlayer:
    """始终返回 action 0；第二次轮到自己时一定非法。"""

    def __init__(self, name: str = "illegal") -> None:
        self.name = name

    def select_action(self, board: Board) -> Optional[int]:
        return 0


# ---------------------------------------------------------------------------
# 基本对局
# ---------------------------------------------------------------------------
class TestArenaBasics:
    def test_can_be_instantiated(self):
        arena = Arena(FirstLegalPlayer("A"), FirstLegalPlayer("B"), max_moves=10)
        assert arena.board_size == BOARD_SIZE
        assert arena.max_moves == 10

    def test_random_vs_random_completes_one_game(self):
        arena = Arena(
            RandomPlayer(seed=0, name="rand_a"),
            RandomPlayer(seed=1, name="rand_b"),
        )
        result = arena.play_one_game()
        assert isinstance(result, GameResult)
        assert result.winner in (BLACK, WHITE, 0)
        assert result.num_moves >= 1
        assert result.black_player_name == "rand_a"
        assert result.white_player_name == "rand_b"

    def test_play_many_games_runs_all(self):
        arena = Arena(
            RandomPlayer(seed=0, name="A"),
            RandomPlayer(seed=1, name="B"),
        )
        results = arena.play_many_games(3, alternate_sides=False)
        assert len(results) == 3
        for r in results:
            assert r.winner in (BLACK, WHITE, 0)

    def test_moves_are_action_indices(self):
        arena = Arena(FirstLegalPlayer("A"), FirstLegalPlayer("B"), max_moves=4)
        result = arena.play_one_game()
        for action in result.moves:
            assert isinstance(action, int)
            assert 0 <= action < BOARD_SIZE * BOARD_SIZE


# ---------------------------------------------------------------------------
# alternate_sides
# ---------------------------------------------------------------------------
class TestAlternateSides:
    def test_alternate_sides_swaps_seats(self):
        arena = Arena(FirstLegalPlayer("A"), FirstLegalPlayer("B"), max_moves=2)
        results = arena.play_many_games(2, alternate_sides=True)
        assert len(results) == 2
        assert results[0].black_player_name == "A"
        assert results[0].white_player_name == "B"
        assert results[1].black_player_name == "B"
        assert results[1].white_player_name == "A"

    def test_no_alternate_keeps_seats(self):
        arena = Arena(FirstLegalPlayer("A"), FirstLegalPlayer("B"), max_moves=2)
        results = arena.play_many_games(3, alternate_sides=False)
        for r in results:
            assert r.black_player_name == "A"
            assert r.white_player_name == "B"


# ---------------------------------------------------------------------------
# 受控胜负场景
# ---------------------------------------------------------------------------
class TestControlledOutcomes:
    def test_black_wins_via_scripted_player(self):
        # A 执黑：(0,0),(1,0),(2,0),(3,0),(4,0) -> 5 连胜
        # B 执白：(0,14),(1,14),(2,14),(3,14)
        a_actions = [_idx(0, 0), _idx(1, 0), _idx(2, 0), _idx(3, 0), _idx(4, 0)]
        b_actions = [_idx(0, 14), _idx(1, 14), _idx(2, 14), _idx(3, 14)]
        arena = Arena(
            ScriptedPlayer(a_actions, name="A"),
            ScriptedPlayer(b_actions, name="B"),
        )
        result = arena.play_one_game()
        assert result.winner == BLACK
        assert result.black_player_name == "A"
        assert result.white_player_name == "B"
        assert result.reason == "black_win"
        assert result.num_moves == 9

    def test_white_wins_via_scripted_player(self):
        # A 执黑随便下，B 执白：(0,0)..(4,0) 行 0 五连胜
        a_actions = [
            _idx(7, 7),    # 1 B
            _idx(0, 1),    # 3 B
            _idx(0, 2),    # 5 B
            _idx(0, 3),    # 7 B
            _idx(0, 4),    # 9 B
        ]
        b_actions = [
            _idx(0, 0),    # 2 W
            _idx(1, 0),    # 4 W
            _idx(2, 0),    # 6 W
            _idx(3, 0),    # 8 W
            _idx(4, 0),    # 10 W -> 5 连
        ]
        arena = Arena(
            ScriptedPlayer(a_actions, name="A"),
            ScriptedPlayer(b_actions, name="B"),
        )
        result = arena.play_one_game()
        assert result.winner == WHITE
        assert result.reason == "white_win"
        assert result.num_moves == 10

    def test_max_moves_cap_returns_safe_result(self):
        # 两个 FirstLegalPlayer 在 max_moves=4 下不会出现 5 连
        arena = Arena(FirstLegalPlayer("A"), FirstLegalPlayer("B"), max_moves=4)
        result = arena.play_one_game()
        assert result.num_moves <= 4
        assert result.winner == 0
        assert result.reason in ("max_moves", "draw")


# ---------------------------------------------------------------------------
# 异常路径
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_illegal_action_raises(self):
        # 第一手 A 落 (0,0)；第二手 B 仍返回 0 -> 已被占用 -> 抛错
        arena = Arena(IllegalPlayer("A"), IllegalPlayer("B"))
        with pytest.raises(RuntimeError):
            arena.play_one_game()

    def test_out_of_range_action_raises(self):
        class BadIndexPlayer:
            def __init__(self, name="bad"):
                self.name = name

            def select_action(self, board):
                return BOARD_SIZE * BOARD_SIZE + 100  # 越界

        arena = Arena(BadIndexPlayer("A"), FirstLegalPlayer("B"))
        with pytest.raises(RuntimeError):
            arena.play_one_game()


# ---------------------------------------------------------------------------
# run_match：跨多盘的统计
# ---------------------------------------------------------------------------
class TestRunMatch:
    def test_run_match_returns_summary_distinguishing_players(self):
        # FirstLegal 行扫描决定的胜负是确定的，但更重要是验证 summary 字段
        a = FirstLegalPlayer("A")
        b = FirstLegalPlayer("B")
        summary, results = run_match(a, b, num_games=4, alternate_sides=True)
        assert len(results) == 4
        assert summary["total_games"] == 4
        assert summary["player_a_name"] == "A"
        assert summary["player_b_name"] == "B"
        assert summary["player_a_wins"] + summary["player_b_wins"] + summary["draws"] == 4

    def test_run_match_alternate_sides_assigns_seats(self):
        a = FirstLegalPlayer("A")
        b = FirstLegalPlayer("B")
        _summary, results = run_match(a, b, num_games=2, alternate_sides=True)
        assert results[0].black_player_name == "A"
        assert results[1].black_player_name == "B"


# ---------------------------------------------------------------------------
# 真实 ModelMCTSPlayer smoke(若装了 torch)
# ---------------------------------------------------------------------------
def test_smoke_model_mcts_player_runs():
    torch = pytest.importorskip("torch")  # 没装 torch 就跳过
    from evaluate.players import ModelMCTSPlayer  # noqa: WPS433
    from model.policy_value_net import PolicyValueNet  # noqa: WPS433

    torch.manual_seed(0)
    model = PolicyValueNet()
    player = ModelMCTSPlayer(
        model=model, num_simulations=2, c_puct=5.0, device="cpu", name="model"
    )
    arena = Arena(player, RandomPlayer(seed=0, name="rand"), max_moves=4)
    result = arena.play_one_game()
    assert isinstance(result, GameResult)
    assert result.winner in (BLACK, WHITE, 0)
    assert result.num_moves <= 4
